#!/usr/bin/env python
"""Generate the Colab fallback notebook from the Modal source, so it cannot drift.

Why generate rather than write by hand
--------------------------------------
The Colab notebook has to run the same prediction and compute the same metrics
as the Modal app, otherwise its results are not comparable and cannot be pooled
or substituted. A hand-written copy of ``ipsae`` in a notebook would be correct
on the day it was written and silently wrong the first time the Modal version
changed.

So the shared functions are lifted verbatim from ``modal_app/af2_multimer.py``
at generation time, and ``tests/test_colab_notebook.py`` regenerates the
notebook and fails if the committed one differs. Drift becomes a failing test
rather than two subtly different numbers in the same paper.

What Colab can and cannot do here
---------------------------------
Colab's free tier gives a T4 with 16 GB, and AlphaFold2-Multimer memory grows
roughly with the square of total length, so the largest complexes in this set
will not fit. The notebook checks the available memory against each job and
skips what cannot run, recording the skip rather than crashing halfway through
a session. Sessions also disconnect, so every job writes its result to Drive as
it finishes and the loop skips anything already done.

Jobs run **smallest first**, which is the opposite of the Modal launcher. On
Modal the largest go first so that out-of-memory failures surface while
abandoning the run is still cheap. On Colab there is nothing to abandon and the
session may vanish at any moment, so the aim is to bank as many completed jobs
as possible before it does.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Functions and constants copied verbatim from the Modal app.
SHARED_NAMES = [
    "MODEL_TYPE",
    "MMSEQS_USER_AGENT",
    "require_parseable_complex_a3m",
    "_d0_scalar",
    "_d0_array",
    "ipsae",
]

SOURCE = Path("modal_app/af2_multimer.py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))
from af2_multimer import IMAGE_PACKAGES, JAX_PACKAGE


def extract(source_path: Path, names: list[str]) -> str:
    """Pull the named top-level definitions out of a module, in file order."""
    text = source_path.read_text()
    tree = ast.parse(text)
    lines = text.splitlines()
    wanted = set(names)
    chunks: list[tuple[int, str]] = []

    for node in tree.body:
        name = None
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif (
            isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
        if name not in wanted:
            continue
        start = node.lineno - 1
        # Include the sphinx-style ``#:`` comment block above a constant, which
        # is where the reason for its value is written.
        while start > 0 and lines[start - 1].lstrip().startswith("#:"):
            start -= 1
        chunks.append((node.lineno, "\n".join(lines[start : node.end_lineno])))
        wanted.discard(name)

    if wanted:
        raise SystemExit(f"could not find {sorted(wanted)} in {source_path}")
    return "\n\n\n".join(text for _, text in sorted(chunks))


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build(shared: str) -> dict:
    cells = [
        markdown(
            """# AlphaFold2-Multimer refolding, Colab fallback

Runs the same predictions as `modal_app/af2_multimer.py`, on Colab's GPU, for
when Modal is unavailable or out of budget.

**Generated, not hand written.** The metrics code below is lifted verbatim from
the Modal app by `scripts/make_colab_notebook.py`, and a test fails if the two
diverge. Do not edit those cells; edit the source and regenerate.

## What this can and cannot do

Colab free gives a T4 with 16 GB. AlphaFold2-Multimer memory grows roughly with
the square of total length, so the largest complexes in this set will not fit.
The notebook measures the available memory, skips what cannot run, and records
the skip. It does not pretend to be a complete substitute for the Modal run.

Sessions disconnect without warning, so every job writes its result to Drive as
it completes, and re-running skips anything already done. Jobs run smallest
first to bank as many as possible before a disconnect.

## What you need in Drive

Put these under `MyDrive/interface_charge/`:

- `designs.csv`
- `test_set.csv`
- `interface_definitions.json` (from `scripts/01_define_interfaces.py`)

Results are written to `MyDrive/interface_charge/af2_results/`."""
        ),
        markdown("## 1. Check the GPU\n\nRuntime, Change runtime type, T4 GPU."),
        code(
            """import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)"""
        ),
        markdown(
            "## 2. Install ColabFold\n\nTakes a few minutes. Pins generated from the "
            "Modal image, so the two backends install the same stack."
        ),
        code(
            "%%capture\n"
            "# Pins generated from modal_app/af2_multimer.py. Two of them are not\n"
            "# preferences and must not be raised:\n"
            "#   colabfold 1.5.5 requires biopython<1.83 and numpy<2\n"
            "#   dm-haiku 0.0.10 imports jax.linear_util, removed in jax 0.4.24, so\n"
            "#   jax must stay at or below 0.4.23 or every prediction dies at import\n"
            f"!pip install -q {' '.join(chr(34) + p + chr(34) for p in IMAGE_PACKAGES)}\n"
            f'!pip install -q "{JAX_PACKAGE}" '
            "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        ),
        markdown("## 3. Mount Drive and set paths"),
        code(
            """from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path

BASE = Path("/content/drive/MyDrive/interface_charge")
RESULTS = BASE / "af2_results"
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "msa_cache").mkdir(exist_ok=True)

for name in ("designs.csv", "test_set.csv", "interface_definitions.json"):
    path = BASE / name
    print(f"{'found  ' if path.is_file() else 'MISSING'} {path}")"""
        ),
        markdown(
            "## 4. Shared code\n\n**Generated from `modal_app/af2_multimer.py`. Do not edit.**"
        ),
        code(
            "from typing import Any\n\nimport numpy as np\n\n\n"
            + shared
            + "\n\n\nprint('shared metrics code loaded')"
        ),
        markdown("## 5. Build the job list\n\nSame pairing rules as the Modal launcher."),
        code(
            """import json

import pandas as pd

designs = pd.read_csv(BASE / "designs.csv")
test_set = pd.read_csv(BASE / "test_set.csv")
definitions = json.loads((BASE / "interface_definitions.json").read_text())

chains_by_id = dict(zip(test_set["pdb_id"], test_set["chains"]))

jobs = []
problems = []
for row in designs.itertuples():
    designed = str(row.designed_chain).split(",")[0].strip()
    pair = chains_by_id.get(row.pdb_id)
    entry = definitions.get(row.pdb_id)
    if pair is None or entry is None:
        problems.append(f"{row.pdb_id}: missing from test_set or definitions")
        continue
    partner = next((c.strip() for c in str(pair).split(",") if c.strip() != designed), None)
    partner_entry = entry["chains"].get(partner) if partner else None
    if partner_entry is None:
        problems.append(f"{row.pdb_id}: no partner sequence for chain {partner}")
        continue
    beta = f"{float(row.beta):+.4f}".replace("+", "p").replace("-", "m").replace(".", "_")
    jobs.append({
        "key": f"{row.pdb_id}_{designed}_beta{beta}_rep{int(row.replicate)}",
        "pdb_id": row.pdb_id,
        "beta": float(row.beta),
        "replicate": int(row.replicate),
        "designed_chain": designed,
        "partner_chain": partner,
        "chains": {designed: row.sequence, partner: partner_entry["sequence"]},
        "msa_mode": {designed: "single_sequence", partner: "msa"},
    })

# Smallest first. On Modal the largest go first so out-of-memory surfaces while
# abandoning is cheap; here the session may vanish at any moment, so the aim is
# to bank as many completed jobs as possible before it does.
jobs.sort(key=lambda j: sum(len(s) for s in j["chains"].values()))

print(f"{len(jobs)} job(s) over {len({j['pdb_id'] for j in jobs})} complex(es)")
if problems:
    print(f"{len(problems)} skipped while building the list:")
    for p in problems[:5]:
        print("  ", p)"""
        ),
        markdown(
            "## 6. Run\n\nResumable. Re-run this cell after a disconnect and it "
            "picks up where it stopped."
        ),
        code(
            '''import time
import traceback

import torch
from colabfold.batch import msa_to_str
from colabfold.batch import run as colabfold_run
from colabfold.colabfold import run_mmseqs2

# AlphaFold2-Multimer memory grows roughly with the square of total length. This
# cap is deliberately conservative: a job that dies takes the session with it,
# and a skipped job recorded honestly is worth more than a crashed notebook.
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
MAX_TOTAL_RESIDUES = 1000 if GPU_GB > 20 else 700
print(f"GPU has {GPU_GB:.0f} GB, capping jobs at {MAX_TOTAL_RESIDUES} total residues")


def build_mixed_a3m(job, ordered):
    """Native partner keeps its MSA, the design contributes depth one.

    Cached per complex and chain, not per job, because the partner sequence is
    identical at every beta and the MMseqs2 server is a free shared resource.
    """
    unpaired = []
    for chain_id in ordered:
        sequence = job["chains"][chain_id]
        if job["msa_mode"][chain_id] == "single_sequence":
            unpaired.append(f">{job['key']}_{chain_id}\\n{sequence}\\n")
            continue
        cached = RESULTS / "msa_cache" / f"{job['pdb_id']}_{chain_id}.a3m"
        if cached.is_file():
            unpaired.append(cached.read_text())
            continue
        result = run_mmseqs2(
            [sequence],
            str(RESULTS / "msa_cache" / f"mmseqs_{job['pdb_id']}_{chain_id}"),
            use_env=True, use_filter=True, use_templates=False, use_pairing=False,
            user_agent=MMSEQS_USER_AGENT,
        )
        lines = result[0] if isinstance(result, (list, tuple)) else result
        if not lines or ">" not in lines:
            raise RuntimeError(f"MMseqs2 returned no alignment for {job['pdb_id']} {chain_id}")
        cached.write_text(lines)
        unpaired.append(lines)
    # No paired block: pairing matches homologues by organism and a design has none.
    return msa_to_str(
        unpaired_msa=unpaired,
        paired_msa=None,
        query_seqs_unique=[job["chains"][c] for c in ordered],
        query_seqs_cardinality=[1] * len(ordered),
    )


done = skipped = failed = 0
for index, job in enumerate(jobs, start=1):
    out_dir = RESULTS / job["key"]
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        done += 1
        continue

    total = sum(len(s) for s in job["chains"].values())
    if total > MAX_TOTAL_RESIDUES:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({
            "key": job["key"], "pdb_id": job["pdb_id"], "beta": job["beta"],
            "replicate": job["replicate"], "designed_chain": job["designed_chain"],
            "skipped": True,
            "reason": f"{total} residues exceeds the {MAX_TOTAL_RESIDUES} cap for a {GPU_GB:.0f} GB GPU",
        }, indent=2) + "\\n")
        skipped += 1
        print(f"[{index}/{len(jobs)}] {job['key']}: SKIPPED, {total} residues")
        continue

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        ordered = sorted(job["chains"])
        query = ":".join(job["chains"][c] for c in ordered)
        lengths = [len(job["chains"][c]) for c in ordered]

        a3m = build_mixed_a3m(job, ordered)
        require_parseable_complex_a3m(a3m, lengths)

        colabfold_run(
            # A one-element LIST, not a string. ColabFold indexes a3m_lines[0],
            # so a bare string yields "#", fails the complex check, and silently
            # falls back to single sequence without raising.
            queries=[(job["key"], query, [a3m])],
            result_dir=str(out_dir),
            num_models=1,
            num_recycles=3,
            model_type=MODEL_TYPE,
            msa_mode="single_sequence",
            use_templates=False,
            random_seed=0,
            is_complex=True,
            rank_by="multimer",
            user_agent=MMSEQS_USER_AGENT,
        )

        scores_files = sorted(out_dir.glob("*scores*.json"))
        if not scores_files:
            raise FileNotFoundError("no ColabFold scores JSON; the prediction did not complete")
        scores = json.loads(scores_files[0].read_text())
        pae = np.array(scores.get("pae", []), dtype=float)

        metrics = {
            "key": job["key"], "pdb_id": job["pdb_id"], "beta": job["beta"],
            "replicate": job["replicate"], "designed_chain": job["designed_chain"],
            "complex_ptm": float(scores["ptm"]) if "ptm" in scores else None,
            "interface_ptm": float(scores["iptm"]) if "iptm" in scores else None,
            "mean_plddt": float(np.mean(scores["plddt"])) if scores.get("plddt") else None,
            "wall_clock_s": time.time() - started,
            "msa_paired": False,
            "source": "colab",
            "skipped": False,
        }
        if pae.size and pae.shape[0] == sum(lengths):
            first = lengths[0]
            metrics["interface_pae"] = float(
                (pae[:first, first:].mean() + pae[first:, :first].mean()) / 2.0
            )
            metrics.update(ipsae(pae, lengths, 10.0))

        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\\n")
        done += 1
        print(f"[{index}/{len(jobs)}] {job['key']}: {metrics['wall_clock_s']/60:.1f} min, "
              f"ipSAE {metrics.get('ipsae_d0res', float('nan')):.3f}")
    except Exception:
        failed += 1
        print(f"[{index}/{len(jobs)}] {job['key']}: FAILED")
        traceback.print_exc()

print(f"\\ndone {done}, skipped {skipped}, failed {failed}")'''
        ),
        markdown("## 7. Collect\n\nWrites one CSV you can download and hand to `scripts/04`."),
        code(
            """rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*/metrics.json"))]
table = pd.DataFrame(rows)
out = BASE / "af2_metrics_colab.csv"
table.to_csv(out, index=False)

ran = table[~table.get("skipped", False).astype(bool)] if "skipped" in table else table
print(f"{len(table)} record(s), {len(ran)} with predictions, written to {out}")
if len(ran):
    print(f"median wall clock: {ran['wall_clock_s'].median()/60:.1f} min")
    print(ran.groupby("beta")[["interface_ptm", "ipsae_d0res"]].median().round(3).to_string())
if "skipped" in table and table["skipped"].astype(bool).any():
    n = int(table["skipped"].astype(bool).sum())
    print(f"\\n{n} job(s) skipped as too large for this GPU. They are recorded, not lost;")
    print("run them on Modal, or report the shortfall explicitly.")"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("notebooks/af2_colab.ipynb"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed notebook differs from a fresh build.",
    )
    args = parser.parse_args(argv)

    notebook = build(extract(SOURCE, SHARED_NAMES))
    rendered = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"

    if args.check:
        if not args.out.is_file():
            print(f"ERROR: {args.out} does not exist", file=sys.stderr)
            return 1
        if args.out.read_text() != rendered:
            print(
                f"ERROR: {args.out} differs from a fresh build. The notebook shares "
                f"code with {SOURCE} and one of them has changed.\n"
                f"Regenerate with: python {Path(__file__).name} --out {args.out}",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out} is up to date with {SOURCE}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    print(f"wrote {args.out} ({len(notebook['cells'])} cells)")
    print(f"shared from {SOURCE}: {', '.join(SHARED_NAMES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
