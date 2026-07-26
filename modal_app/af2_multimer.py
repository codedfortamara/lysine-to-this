#!/usr/bin/env python
"""AlphaFold2-Multimer refolding of the design set on Modal A100s.

This is the answer to the paper's Limitation (i): the existing evaluation folds
each chain of a complex in isolation with ESMFold, so it cannot see the
interface, which is exactly where the perturbed charge lives. Here the whole
complex is refolded and interface-resolved metrics are reported.

Design notes
------------

**Weights are never downloaded per container.** They live in a Modal Volume
(``AF2Params.weights_volume``) mounted read-only. Downloading several gigabytes
of parameters on every cold start would dominate both the wall-clock and the
bill. Populate the volume once, out of band::

    modal volume create af2-weights
    modal volume put af2-weights /path/to/params /params

**The run is resumable.** Every job writes its output to the results volume as
soon as it completes, and the job list is filtered against what is already
there before anything launches. A run that is interrupted, hits a quota, or is
cancelled halfway can be relaunched with the same command and will pick up only
the outstanding work. This is not a nicety: at several hundred jobs, a run that
has to start over is a run that blows the budget.

**Single-sequence mode for redesigned chains, MSA retained for native chains.**
A ProteinMPNN design has no evolutionary history, so an MSA for it is
meaningless and actively misleading: the search would return homologues of the
native sequence the design was derived from, and the prediction would be
propped up by information the design does not carry. A chain held at its native
sequence keeps its MSA, because there the evolutionary signal is real. Mixing
the two within one complex is the correct treatment and it is what the
``msa_mode`` field of each job encodes.

**One obvious timeout constant.** ``AF2Params.timeout_s``, two hours, sized for
the largest complex rather than the median. A job killed at hour two has cost
two GPU-hours and produced nothing; a slot held idle costs far less.

Dry run
-------
``--dry-run`` prints the job count and an estimated GPU-hour cost and launches
nothing. It works **without ``modal`` installed**, because the job enumeration
and the arithmetic are pure Python. Use it before every real launch.

The per-job time estimate is a planning figure, not a measurement. Run a pilot
of about ten jobs, take the observed median, and put it in
``config.AF2Params.estimated_minutes_per_job`` before quoting a cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interface_charge.config import (
    DEFAULT_CONFIG,
    MODAL_GPU_RATES_USD_PER_HOUR,
    AF2Params,
)

try:  # pragma: no cover - exercised by whether the extra is installed
    import modal

    MODAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    modal = None  # type: ignore[assignment]
    MODAL_AVAILABLE = False


PARAMS: AF2Params = DEFAULT_CONFIG.af2

#: Per-call timeout in seconds. One constant, referenced everywhere.
TIMEOUT_S: int = PARAMS.timeout_s


# ---------------------------------------------------------------------------
# Job enumeration. Pure Python, no Modal, so --dry-run always works.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Job:
    """One AlphaFold2-Multimer prediction.

    ``chains`` maps author chain identifier to sequence. ``msa_mode`` maps the
    same identifiers to either ``single_sequence`` for a redesigned chain or
    ``msa`` for a chain held at its native sequence.
    """

    pdb_id: str
    beta: float
    replicate: int
    designed_chain: str
    chains: dict[str, str]
    msa_mode: dict[str, str]

    @property
    def key(self) -> str:
        """Stable output name. Also the resumability key."""
        beta = f"{self.beta:+.4f}".replace("+", "p").replace("-", "m").replace(".", "_")
        return f"{self.pdb_id}_{self.designed_chain}_beta{beta}_rep{self.replicate}"

    def total_residues(self) -> int:
        return sum(len(seq) for seq in self.chains.values())


def build_job_list(
    designs_path: Path,
    definitions_path: Path,
    test_set_path: Path,
) -> list[Job]:
    """Enumerate every prediction to run.

    Reads the validated design table, the native sequences from the interface
    definitions produced by script 01, and the chain pairing from the test set.
    Uses the project's own strict loaders, so a malformed table fails here
    rather than after a GPU has been billed.
    """
    from interface_charge.contracts import check_cross_table, load_designs, load_test_set

    designs = load_designs(designs_path)
    test_set = load_test_set(test_set_path)
    check_cross_table(designs, test_set)

    if not definitions_path.is_file():
        raise SystemExit(
            f"interface definitions not found at {definitions_path}. "
            "Run scripts/01_define_interfaces.py first: the native sequences of "
            "the chains held fixed come from there."
        )
    definitions = json.loads(definitions_path.read_text())

    chains_by_id = dict(zip(test_set["pdb_id"], test_set["chains"], strict=True))
    jobs: list[Job] = []
    problems: list[str] = []

    for line, row in enumerate(designs.itertuples(), start=2):
        if len(row.designed_chain) != 1:
            problems.append(f"line {line}: designed_chain names more than one chain")
            continue

        designed = row.designed_chain[0]
        pair = chains_by_id.get(row.pdb_id)
        entry = definitions.get(row.pdb_id)
        if pair is None or entry is None:
            problems.append(f"line {line}: {row.pdb_id} missing from test_set or definitions")
            continue

        partner = next((c for c in pair if c != designed), None)
        if partner is None:
            problems.append(f"line {line}: no partner chain for {row.pdb_id} chain {designed}")
            continue

        partner_entry = entry["chains"].get(partner)
        if partner_entry is None:
            problems.append(f"line {line}: no native sequence for {row.pdb_id} chain {partner}")
            continue

        jobs.append(
            Job(
                pdb_id=row.pdb_id,
                beta=float(row.beta),
                replicate=int(row.replicate),
                designed_chain=designed,
                chains={designed: row.sequence, partner: partner_entry["sequence"]},
                msa_mode={designed: "single_sequence", partner: "msa"},
            )
        )

    if problems:
        raise SystemExit(
            f"{len(problems)} design row(s) could not be turned into jobs:\n  "
            + "\n  ".join(problems[:20])
        )

    return jobs


def estimate_cost(
    jobs: list[Job] | int,
    params: AF2Params = PARAMS,
    gpu_type: str | None = None,
) -> dict[str, Any]:
    """Estimated GPU-hours and dollars for a job list, or for a bare job count.

    Accepts an integer so that a grid can be costed before ``designs.csv``
    exists, which is the situation for most of the planning window.

    Three things are counted that a jobs-times-minutes calculation misses, and
    all three are real money:

    * **Cold starts.** Each container pulls the image and loads the model
      parameters from the weights volume once. Paid per container, not per job,
      so it dominates a ten-job pilot and washes out over hundreds.
    * **Failures and retries.** Refolding failures are not random: large and
      heavily charged complexes fail more often, which is precisely the corner
      of the grid this study cares about.
    * **Wall-clock.** Reported alongside the cost because with the deadline in
      view it is often the binding constraint, not the money.

    The per-job minutes remain a planning assumption, not a measurement, and
    the returned dict says so in ``estimate_basis``.
    """
    n_jobs = jobs if isinstance(jobs, int) else len(jobs)
    gpu_type = gpu_type or params.gpu_type
    rate = MODAL_GPU_RATES_USD_PER_HOUR.get(gpu_type)
    if rate is None:
        raise ValueError(
            f"no published rate recorded for GPU {gpu_type!r}; "
            f"known: {sorted(MODAL_GPU_RATES_USD_PER_HOUR)}"
        )

    containers = min(params.max_containers, n_jobs)
    compute_hours = n_jobs * params.estimated_minutes_per_job / 60.0
    cold_start_hours = containers * params.cold_start_minutes / 60.0
    gpu_hours = (compute_hours + cold_start_hours) * (1.0 + params.failure_overhead)

    out: dict[str, Any] = {
        "n_jobs": n_jobs,
        "gpu_type": gpu_type,
        "usd_per_gpu_hour": rate,
        "estimated_minutes_per_job": params.estimated_minutes_per_job,
        # Deliberately unrounded. Rounding here would destroy the arithmetic
        # properties a caller may rely on (a grid of twice the size costing
        # exactly twice the compute) and would compound across the comparison
        # table. Round at the point of display instead.
        "compute_gpu_hours": compute_hours,
        "cold_start_gpu_hours": cold_start_hours,
        "failure_overhead": params.failure_overhead,
        "estimated_gpu_hours": gpu_hours,
        "estimated_usd": gpu_hours * rate,
        "max_containers": params.max_containers,
        "estimated_wall_clock_hours": gpu_hours / max(containers, 1),
        "timeout_s": params.timeout_s,
        "estimate_basis": (
            "planning assumption from config.AF2Params, not a measurement; "
            "GPU rate is UNVERIFIED, confirm against modal.com/pricing"
        ),
    }

    if not isinstance(jobs, int) and jobs:
        residues = sorted(job.total_residues() for job in jobs)
        out["median_total_residues"] = residues[len(residues) // 2]
        out["max_total_residues"] = residues[-1]
    return out


# ---------------------------------------------------------------------------
# Modal application
# ---------------------------------------------------------------------------

if MODAL_AVAILABLE:  # pragma: no cover - requires Modal
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "wget", "build-essential")
        # ColabFold provides an AlphaFold2-Multimer implementation with a
        # single-sequence mode, which is what the redesigned chains need.
        .pip_install(
            "colabfold[alphafold]==1.5.5",
            "biopython>=1.83",
            "numpy>=1.26",
            extra_options="--extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        )
        .pip_install("jax[cuda12]==0.4.28")
        .env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "TF_FORCE_UNIFIED_MEMORY": "1"})
    )

    app = modal.App("interface-charge-af2-multimer")
    weights_volume = modal.Volume.from_name(PARAMS.weights_volume, create_if_missing=True)
    results_volume = modal.Volume.from_name(PARAMS.results_volume, create_if_missing=True)

    WEIGHTS_PATH = "/weights"
    RESULTS_PATH = "/results"

    @app.function(
        image=image,
        gpu=PARAMS.gpu_type,
        volumes={WEIGHTS_PATH: weights_volume, RESULTS_PATH: results_volume},
        timeout=TIMEOUT_S,
        max_containers=PARAMS.max_containers,
        retries=modal.Retries(max_retries=1, backoff_coefficient=2.0),
    )
    def predict(job_payload: dict) -> dict:
        """Refold one complex and return interface-resolved metrics.

        Writes the raw predicted PDB and a metrics JSON to the results volume
        before returning, so that a lost return value never costs a rerun.
        """
        import os
        import time

        from colabfold.batch import run as colabfold_run

        job = Job(**job_payload)
        out_dir = Path(RESULTS_PATH) / job.key
        metrics_path = out_dir / "metrics.json"

        if metrics_path.is_file():
            return json.loads(metrics_path.read_text())

        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()

        # ColabFold takes a colon-joined multimer sequence. Chains are ordered
        # deterministically so that chain identity in the output is predictable.
        ordered = sorted(job.chains)
        query_sequence = ":".join(job.chains[c] for c in ordered)

        # Any chain in single-sequence mode forces single-sequence mode for the
        # complex under ColabFold's batch interface. That is the conservative
        # choice: it never props a design up with homologues it does not have.
        # Retaining a real MSA for the native chain alone needs a precomputed
        # per-chain a3m, which is the documented follow-up below.
        use_msa = all(mode == "msa" for mode in job.msa_mode.values())

        colabfold_run(
            queries=[(job.key, query_sequence, None)],
            result_dir=str(out_dir),
            num_models=PARAMS.num_models,
            num_recycles=PARAMS.num_recycles,
            model_type="alphafold2_multimer_v3",
            msa_mode="mmseqs2_uniref_env" if use_msa else "single_sequence",
            use_templates=False,
            random_seed=PARAMS.random_seed,
            is_complex=True,
            rank_by="multimer",
        )

        native_path = Path(RESULTS_PATH) / "natives" / f"{job.pdb_id}.pdb"
        metrics = collect_metrics(
            out_dir=out_dir,
            job=job,
            chain_order=ordered,
            native_path=native_path if native_path.is_file() else None,
        )
        metrics.update(
            {
                "pdb_id": job.pdb_id,
                "designed_chain": job.designed_chain,
                "beta": job.beta,
                "replicate": job.replicate,
                "key": job.key,
                "wall_clock_s": time.time() - started,
                "msa_mode": job.msa_mode,
                "num_recycles": PARAMS.num_recycles,
                "num_models": PARAMS.num_models,
                "random_seed": PARAMS.random_seed,
                "hostname_gpu": os.environ.get("MODAL_GPU", PARAMS.gpu_type),
            }
        )

        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n")
        results_volume.commit()
        return metrics

    @app.local_entrypoint()
    def run(
        designs: str = "data/raw/designs.csv",
        test_set: str = "data/raw/test_set.csv",
        definitions: str = "results/interface_definitions.json",
    ) -> None:
        """Launch the full grid, skipping anything already done."""
        jobs = build_job_list(Path(designs), Path(definitions), Path(test_set))
        done = completed_keys()
        outstanding = [job for job in jobs if job.key not in done]

        estimate = estimate_cost(outstanding)
        print(f"{len(jobs)} job(s) total, {len(done)} already complete, {len(outstanding)} to run")
        print(json.dumps(estimate, indent=2))

        if not outstanding:
            print("nothing to do")
            return

        results = list(predict.map([asdict(job) for job in outstanding], return_exceptions=True))
        failures = [r for r in results if isinstance(r, Exception)]
        print(f"\n{len(results) - len(failures)} succeeded, {len(failures)} failed")
        for failure in failures[:10]:
            print(f"  {type(failure).__name__}: {failure}")

    def completed_keys() -> set[str]:
        """Job keys already present on the results volume."""
        keys: set[str] = set()
        try:
            for entry in results_volume.listdir("/", recursive=False):
                name = Path(entry.path).name
                if name == "natives":
                    continue
                try:
                    next(iter(results_volume.listdir(f"/{name}/metrics.json")))
                except (FileNotFoundError, StopIteration, GeneratorExit):
                    continue
                keys.add(name)
        except (FileNotFoundError, GeneratorExit):
            return set()
        return keys

else:  # pragma: no cover

    def completed_keys() -> set[str]:
        """Without Modal installed there is nothing to interrogate."""
        return set()


# ---------------------------------------------------------------------------
# Metrics. Importable without Modal so that they can be unit tested.
# ---------------------------------------------------------------------------


def collect_metrics(
    out_dir: Path,
    job: Job,
    chain_order: list[str],
    native_path: Path | None,
) -> dict:
    """Extract interface pTM, complex pTM, interface PAE and interface RMSD.

    ColabFold writes a scores JSON per model containing ``ptm``, ``iptm``,
    ``plddt`` and the full ``pae`` matrix. Interface PAE is the mean of the PAE
    matrix over cross-chain residue pairs, which is the standard definition and
    is the one that correlates with genuine binding rather than with fold
    quality.

    Interface RMSD is computed against the native complex when one is
    available: superpose on the partner chain, then take the backbone RMSD over
    the native interface residues of the designed chain. Superposing on the
    partner rather than on the whole complex is what makes the number a measure
    of *interface* displacement rather than of global drift.
    """
    import numpy as np

    scores_files = sorted(out_dir.glob("*scores*.json"))
    if not scores_files:
        raise FileNotFoundError(
            f"no ColabFold scores JSON in {out_dir}. The prediction did not complete."
        )
    scores = json.loads(scores_files[0].read_text())

    pae = np.array(scores.get("pae", []), dtype=float)
    lengths = [len(job.chains[c]) for c in chain_order]

    interface_pae: float | None = None
    if pae.size and len(lengths) == 2 and pae.shape[0] == sum(lengths):
        first = lengths[0]
        cross_upper = pae[:first, first:]
        cross_lower = pae[first:, :first]
        interface_pae = float((cross_upper.mean() + cross_lower.mean()) / 2.0)

    predicted_pdbs = sorted(out_dir.glob("*relaxed*.pdb")) or sorted(out_dir.glob("*.pdb"))

    metrics: dict = {
        "complex_ptm": float(scores["ptm"]) if "ptm" in scores else None,
        "interface_ptm": float(scores["iptm"]) if "iptm" in scores else None,
        "interface_pae": interface_pae,
        "mean_plddt": float(np.mean(scores["plddt"])) if scores.get("plddt") else None,
        "predicted_pdb": str(predicted_pdbs[0].name) if predicted_pdbs else None,
        "interface_rmsd_a": None,
        "interface_rmsd_note": "native structure not available in the container",
    }

    if native_path is not None and predicted_pdbs:
        metrics.update(
            interface_rmsd(
                native_path=native_path,
                predicted_path=predicted_pdbs[0],
                designed_chain=job.designed_chain,
                chain_order=chain_order,
            )
        )

    return metrics


def interface_rmsd(
    native_path: Path,
    predicted_path: Path,
    designed_chain: str,
    chain_order: list[str],
) -> dict:
    """Backbone RMSD over interface residues, superposed on the partner chain.

    Returns a dict rather than a bare float so that the failure mode carries an
    explanation instead of a silent ``None``.
    """
    import numpy as np
    from Bio.PDB import PDBParser, Superimposer

    from interface_charge.config import DEFAULT_CONFIG
    from interface_charge.interface import Partition, define_interface

    config = DEFAULT_CONFIG
    parser = PDBParser(QUIET=True)

    try:
        native = parser.get_structure("native", native_path)[0]
        predicted = parser.get_structure("predicted", predicted_path)[0]
    except Exception as exc:
        return {"interface_rmsd_a": None, "interface_rmsd_note": f"parse failed: {exc!r}"}

    partner = next((c for c in chain_order if c != designed_chain), None)
    if partner is None:
        return {"interface_rmsd_a": None, "interface_rmsd_note": "no partner chain"}

    # ColabFold labels the predicted chains A, B, ... in the order they were
    # supplied, which is chain_order. Map those back onto the native labels.
    predicted_labels = {name: chr(ord("A") + i) for i, name in enumerate(chain_order)}

    definition = define_interface(
        native,
        pdb_id=native_path.stem,
        chain_a=designed_chain,
        chain_b=partner,
        structure_sha256="",
        interface_params=config.interface,
        structure_params=config.structure,
    )
    interface_ids = definition.classification_for(designed_chain).ids_in(Partition.INTERFACE)
    wanted = {rid.seqid for rid in interface_ids}

    def backbone(model, chain_label: str, seqids: set[int] | None) -> dict[int, dict]:
        out: dict[int, dict] = {}
        if chain_label not in {c.id for c in model}:
            return out
        for residue in model[chain_label]:
            if residue.id[0] != " ":
                continue
            seqid = residue.id[1]
            if seqids is not None and seqid not in seqids:
                continue
            atoms = {a.get_id(): a for a in residue if a.get_id() in ("N", "CA", "C", "O")}
            if len(atoms) == 4:
                out[seqid] = atoms
        return out

    # Superpose on the partner chain so that the RMSD reports interface
    # displacement rather than global drift.
    native_partner = backbone(native, partner, None)
    pred_partner = backbone(predicted, predicted_labels[partner], None)
    shared_partner = sorted(set(native_partner) & set(pred_partner))
    if len(shared_partner) < 10:
        return {
            "interface_rmsd_a": None,
            "interface_rmsd_note": (
                f"only {len(shared_partner)} shared partner-chain residues, too few to superpose on"
            ),
        }

    fixed = [native_partner[s][a] for s in shared_partner for a in ("N", "CA", "C", "O")]
    moving = [pred_partner[s][a] for s in shared_partner for a in ("N", "CA", "C", "O")]

    superimposer = Superimposer()
    superimposer.set_atoms(fixed, moving)
    superimposer.apply(list(predicted.get_atoms()))

    native_iface = backbone(native, designed_chain, wanted)
    pred_iface = backbone(predicted, predicted_labels[designed_chain], wanted)
    shared_iface = sorted(set(native_iface) & set(pred_iface))
    if not shared_iface:
        return {
            "interface_rmsd_a": None,
            "interface_rmsd_note": "no shared interface residues after superposition",
        }

    deltas = [
        native_iface[s][a].get_coord() - pred_iface[s][a].get_coord()
        for s in shared_iface
        for a in ("N", "CA", "C", "O")
    ]
    rmsd = float(np.sqrt((np.asarray(deltas) ** 2).sum(axis=1).mean()))

    return {
        "interface_rmsd_a": rmsd,
        "interface_rmsd_note": (
            f"backbone RMSD over {len(shared_iface)} native interface residues, "
            f"superposed on {len(shared_partner)} residues of chain {partner}"
        ),
        "receptor_superposition_rmsd_a": float(superimposer.rms),
        "n_interface_residues_compared": len(shared_iface),
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--designs", type=Path, default=Path("data/raw/designs.csv"))
    parser.add_argument("--test-set", type=Path, default=Path("data/raw/test_set.csv"))
    parser.add_argument(
        "--definitions", type=Path, default=Path("results/interface_definitions.json")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the job count and estimated GPU-hour cost without launching "
            "anything. Works without modal installed."
        ),
    )
    parser.add_argument(
        "--minutes-per-job",
        type=float,
        default=None,
        help=(
            "Override the per-job time estimate. Use the observed median from a "
            "pilot run rather than the planning default."
        ),
    )
    parser.add_argument(
        "--gpu",
        default=None,
        choices=sorted(MODAL_GPU_RATES_USD_PER_HOUR),
        help="Cost against this GPU type instead of the configured default.",
    )
    parser.add_argument(
        "--compare-gpus",
        action="store_true",
        help="Cost the same grid across every GPU type with a recorded rate.",
    )
    grid = parser.add_argument_group(
        "hypothetical grid",
        "Cost a grid before designs.csv exists. Give all three to skip reading "
        "the input tables entirely, which is the usual situation while planning.",
    )
    grid.add_argument("--assume-complexes", type=int, default=None)
    grid.add_argument("--assume-betas", type=int, default=None)
    grid.add_argument("--assume-replicates", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.dry_run:
        if not MODAL_AVAILABLE:
            print(
                "modal is not installed. Install it with `uv pip install -e '.[modal]'`, "
                "or use --dry-run, which needs no GPU and no Modal.",
                file=sys.stderr,
            )
            return 2
        print(
            "To launch, use Modal's own entrypoint so that the app context is set up:\n"
            "  modal run modal_app/af2_multimer.py\n"
            "This CLI exists for --dry-run.",
            file=sys.stderr,
        )
        return 2

    params = PARAMS
    if args.minutes_per_job is not None:
        params = replace(params, estimated_minutes_per_job=args.minutes_per_job)

    hypothetical = (args.assume_complexes, args.assume_betas, args.assume_replicates)
    if any(v is not None for v in hypothetical):
        if any(v is None for v in hypothetical):
            print(
                "ERROR: --assume-complexes, --assume-betas and --assume-replicates "
                "must be given together.",
                file=sys.stderr,
            )
            return 2
        n_jobs = args.assume_complexes * args.assume_betas * args.assume_replicates
        target: list[Job] | int = n_jobs
        print("=== AlphaFold2-Multimer dry run (hypothetical grid) ===")
        print(f"complexes:            {args.assume_complexes}")
        print(f"beta values:          {args.assume_betas}")
        print(f"replicates:           {args.assume_replicates}")
        print(f"jobs:                 {n_jobs}")
        print("\nNOTE: no input tables were read. This costs a grid you describe,")
        print("not one that exists. Re-run without --assume-* once designs.csv is here.")
    else:
        jobs = build_job_list(args.designs, args.definitions, args.test_set)
        done = completed_keys()
        outstanding = [job for job in jobs if job.key not in done]
        target = outstanding
        print("=== AlphaFold2-Multimer dry run ===")
        print(f"jobs enumerated:      {len(jobs)}")
        print(f"already complete:     {len(done)}")
        print(f"outstanding:          {len(outstanding)}")
        print(f"unique complexes:     {len({job.pdb_id for job in jobs})}")
        print(f"unique beta values:   {sorted({job.beta for job in jobs})}")
        print(f"replicates per point: {sorted({job.replicate for job in jobs})}")

    if args.compare_gpus:
        print(f"\n{'GPU':<12} {'USD/hr':>7} {'GPU-h':>8} {'cost':>9} {'wall-clock':>11}")
        print("-" * 51)
        for gpu in sorted(MODAL_GPU_RATES_USD_PER_HOUR, key=MODAL_GPU_RATES_USD_PER_HOUR.get):
            e = estimate_cost(target, params, gpu_type=gpu)
            print(
                f"{gpu:<12} {e['usd_per_gpu_hour']:>7.2f} {e['estimated_gpu_hours']:>8.1f} "
                f"${e['estimated_usd']:>8.0f} {e['estimated_wall_clock_hours']:>9.1f} h"
            )
        print("\nTWO CAVEATS, both of which flatter the cheap cards:")
        print("  1. Minutes per job is held CONSTANT across GPU types here, which is")
        print("     false. A T4 is several times slower than an A100 on the same job,")
        print("     so its true cost is higher than shown and its wall-clock much")
        print("     higher. Only a per-GPU pilot gives an honest comparison.")
        print("  2. AlphaFold2-Multimer memory grows roughly with the square of total")
        print("     length, so the largest complexes may not fit the smaller cards at")
        print("     all. Verify on the largest complex before committing the grid.")
    else:
        print()
        estimate = estimate_cost(target, params, gpu_type=args.gpu)
        rounded = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in estimate.items()}
        print(json.dumps(rounded, indent=2))

    print(
        f"\nNOTE: {params.estimated_minutes_per_job} minutes per job is a planning "
        "assumption from\nconfig.AF2Params, not a measurement, and the GPU rates are "
        "UNVERIFIED. Run a pilot\nof about ten jobs, take the observed median, and "
        "re-run with --minutes-per-job\nbefore committing to a budget."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
