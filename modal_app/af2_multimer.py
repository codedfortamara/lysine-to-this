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
sequence keeps its MSA, because there the evolutionary signal is real, and
dropping it would degrade every prediction at every beta for reasons unrelated
to charge, which is the kind of uniform degradation that hides a real effect.

ColabFold's ``msa_mode`` switch is all-or-nothing across a complex, so this is
done by building the alignment directly and passing it in: MMseqs2 is queried
for the native chain only, the designed chain contributes a depth-one block of
its own sequence, and ColabFold's own ``msa_to_str`` assembles the complex a3m
so that the format is never reimplemented here. There is no paired block, since
pairing matches homologues by organism and a design has no organism. That
limitation is recorded per job as ``msa_paired`` rather than left implicit.

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

#: How the grid is cut into waves, by beta. The usable band goes first because
#: that is where the science is: at beta = +/-3 the upstream data shows only 10
#: to 17 percent of designs folding at all, so refolding them complex-aware
#: mostly confirms that a broken monomer is still broken. Ordering this way means
#: that if the budget runs out after wave one, what remains is a complete grid
#: over the interesting range rather than a partial grid over everything.
WAVE_BETAS: list[list[float]] = [[0.0, -1.5, 1.5], [-3.0, 3.0]]

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
    minutes_per_job = params.estimated_minutes_per_job + params.msa_overhead_minutes
    compute_hours = n_jobs * minutes_per_job / 60.0
    cold_start_hours = containers * params.cold_start_minutes / 60.0
    gpu_hours = (compute_hours + cold_start_hours) * (1.0 + params.failure_overhead)

    out: dict[str, Any] = {
        "n_jobs": n_jobs,
        "gpu_type": gpu_type,
        "usd_per_gpu_hour": rate,
        "estimated_minutes_per_job": minutes_per_job,
        "estimated_minutes_per_job_compute": params.estimated_minutes_per_job,
        "estimated_minutes_per_job_msa_overhead": params.msa_overhead_minutes,
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
# Wave planning, canary ordering and the budget guard
# ---------------------------------------------------------------------------


def order_canary_first(jobs: list[Job], n_canary: int = 5) -> list[Job]:
    """Put the largest complexes first, then everything else deterministically.

    The failures that cost real money are out-of-memory and timeout, and both
    are driven by total complex length. Running the largest jobs first surfaces
    those failures within the first few minutes, when abandoning the run costs
    pennies, instead of at job 120 when it costs most of a wave.

    The remainder is ordered by identifier rather than by size, so that a resumed
    run processes the outstanding work in the same order every time and the
    progress output is comparable between runs.
    """
    if n_canary < 0:
        raise ValueError(f"n_canary must not be negative, got {n_canary}")
    by_size = sorted(jobs, key=lambda j: (-j.total_residues(), j.key))
    canary = by_size[:n_canary]
    rest = sorted(by_size[n_canary:], key=lambda j: j.key)
    return canary + rest


def split_into_waves(jobs: list[Job], wave_betas: list[list[float]]) -> list[list[Job]]:
    """Group jobs into waves by beta value.

    Waves are cut by beta rather than by complex so that every stopping point is
    a complete, analysable grid: stopping after wave one leaves every complex
    covered at the betas that wave contained, rather than some complexes covered
    at every beta and others not covered at all.

    Any job whose beta appears in no wave is returned as a final wave, so nothing
    is silently dropped by a mis-specified plan.
    """
    remaining = list(jobs)
    waves: list[list[Job]] = []
    for betas in wave_betas:
        wanted = {round(b, 6) for b in betas}
        wave = [j for j in remaining if round(j.beta, 6) in wanted]
        remaining = [j for j in remaining if round(j.beta, 6) not in wanted]
        waves.append(wave)
    if remaining:
        waves.append(remaining)
    return waves


class BudgetGuard:
    """Tracks spend and refuses to *start* work beyond a ceiling.

    Deliberately incapable of stopping a job that is already running. Killing
    work mid-flight wastes everything already paid for in that batch and leaves a
    partial grid, which is worse than a small overshoot. The guard therefore
    gates dispatch only: in-flight jobs always finish, and the overshoot is
    bounded by one chunk.

    Spend is estimated from observed wall-clock, so it becomes more accurate as
    the run proceeds and does not depend on the planning assumption after the
    first few jobs have returned.
    """

    def __init__(self, ceiling_usd: float, usd_per_gpu_hour: float) -> None:
        if ceiling_usd <= 0:
            raise ValueError(f"ceiling_usd must be positive, got {ceiling_usd}")
        self.ceiling_usd = ceiling_usd
        self.usd_per_gpu_hour = usd_per_gpu_hour
        self.gpu_seconds = 0.0
        self.n_recorded = 0

    def record(self, wall_clock_s: float) -> None:
        """Account for one completed job."""
        self.gpu_seconds += max(0.0, float(wall_clock_s))
        self.n_recorded += 1

    @property
    def spent_usd(self) -> float:
        return self.gpu_seconds / 3600.0 * self.usd_per_gpu_hour

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.ceiling_usd - self.spent_usd)

    def observed_minutes_per_job(self) -> float | None:
        """Measured median-free mean, or None before anything has returned."""
        if not self.n_recorded:
            return None
        return self.gpu_seconds / self.n_recorded / 60.0

    def may_start(self, n_jobs: int, minutes_per_job: float) -> bool:
        """Whether a chunk of ``n_jobs`` fits inside what is left."""
        observed = self.observed_minutes_per_job()
        minutes = observed if observed is not None else minutes_per_job
        projected = n_jobs * minutes / 60.0 * self.usd_per_gpu_hour
        return projected <= self.remaining_usd

    def status(self) -> str:
        observed = self.observed_minutes_per_job()
        measured = (
            f", {observed:.1f} min/job measured over {self.n_recorded}"
            if observed is not None
            else ", no jobs measured yet"
        )
        return (
            f"spent ${self.spent_usd:.2f} of ${self.ceiling_usd:.2f} "
            f"(${self.remaining_usd:.2f} left{measured})"
        )


@dataclass(frozen=True, slots=True)
class WaveReport:
    """What a wave produced, in the terms needed to decide whether to continue."""

    wave: int
    n_jobs: int
    n_succeeded: int
    n_failed: int
    median_interface_ptm: float | None
    median_wall_clock_minutes: float | None
    spent_usd: float
    failures: tuple[str, ...]

    @property
    def success_rate(self) -> float:
        return self.n_succeeded / self.n_jobs if self.n_jobs else 0.0

    def render(self) -> str:
        iptm = (
            f"{self.median_interface_ptm:.3f}"
            if self.median_interface_ptm is not None
            else "unavailable"
        )
        minutes = (
            f"{self.median_wall_clock_minutes:.1f}"
            if self.median_wall_clock_minutes is not None
            else "unavailable"
        )
        lines = [
            f"=== wave {self.wave} complete ===",
            f"  jobs:                 {self.n_jobs}",
            f"  succeeded:            {self.n_succeeded} ({self.success_rate:.1%})",
            f"  failed:               {self.n_failed}",
            f"  median interface pTM: {iptm}",
            f"  median minutes/job:   {minutes}",
            f"  spend this run:       ${self.spent_usd:.2f}",
        ]
        if self.failures:
            lines.append(f"  first failures:       {list(self.failures[:5])}")
        return "\n".join(lines)


def summarise_wave(wave: int, results: list, guard: BudgetGuard) -> WaveReport:
    """Build a report from a wave's returned metrics.

    Exceptions are counted as failures rather than raised. A wave that partly
    failed is still informative, and the decision about whether to continue
    belongs to a person looking at the numbers.
    """
    import statistics as _statistics

    succeeded = [r for r in results if isinstance(r, dict)]
    failed = [r for r in results if not isinstance(r, dict)]

    iptm = [float(r["interface_ptm"]) for r in succeeded if r.get("interface_ptm") is not None]
    minutes = [
        float(r["wall_clock_s"]) / 60.0 for r in succeeded if r.get("wall_clock_s") is not None
    ]

    return WaveReport(
        wave=wave,
        n_jobs=len(results),
        n_succeeded=len(succeeded),
        n_failed=len(failed),
        median_interface_ptm=_statistics.median(iptm) if iptm else None,
        median_wall_clock_minutes=_statistics.median(minutes) if minutes else None,
        spent_usd=guard.spent_usd,
        failures=tuple(f"{type(f).__name__}: {f}"[:200] for f in failed),
    )


def chunked(items: list, size: int) -> list[list]:
    """Split a list into fixed-size chunks, the unit the budget guard gates on."""
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Pilot selection and extrapolation
# ---------------------------------------------------------------------------


def select_pilot(jobs: list[Job], n_complexes: int = 4, betas_each: int = 2) -> list[Job]:
    """Choose a small job set that measures the two costs the grid is made of.

    A pilot exists to replace a planning assumption with a measurement, so it
    has to measure the same thing the grid will spend. Two traps here, and the
    obvious pilot falls into both.

    **Do not pilot only the largest complexes.** Canary ordering runs the
    largest first, which is right for surfacing out-of-memory and timeout
    quickly, but a median taken from the largest complexes extrapolates to a cost
    far above what the grid will actually pay. The pilot therefore takes a
    spread across the size distribution, while still including the largest so
    the risk check is not lost.

    **Do not pilot one beta per complex.** The MMseqs2 search is paid once per
    complex and cached for the other four charge settings, so a pilot of
    distinct complexes measures only cold jobs and overstates the grid, where
    four jobs in five are warm. Running each pilot complex at more than one beta
    measures both, and ``extrapolate_from_pilot`` uses them separately.
    """
    if n_complexes < 1 or betas_each < 1:
        raise ValueError(
            f"pilot needs at least one complex and one beta, got {n_complexes} and {betas_each}"
        )

    by_complex: dict[str, list[Job]] = {}
    for job in jobs:
        by_complex.setdefault(job.pdb_id, []).append(job)

    # Order complexes by size, then take evenly spaced ranks so the sample spans
    # the distribution rather than clustering at one end.
    ranked = sorted(by_complex, key=lambda pdb: max(j.total_residues() for j in by_complex[pdb]))
    if not ranked:
        return []
    n_complexes = min(n_complexes, len(ranked))
    if n_complexes == 1:
        picked = [ranked[-1]]
    else:
        step = (len(ranked) - 1) / (n_complexes - 1)
        picked = [ranked[round(i * step)] for i in range(n_complexes)]
        picked[-1] = ranked[-1]  # always keep the largest, for the risk check

    selected: list[Job] = []
    for pdb_id in dict.fromkeys(picked):
        # Sort by |beta| so the reference setting is the cold job and the
        # extreme setting is warm, which is the order the grid will run in.
        candidates = sorted(by_complex[pdb_id], key=lambda j: (abs(j.beta), j.beta, j.replicate))
        selected.extend(candidates[:betas_each])
    return selected


def extrapolate_from_pilot(
    results: list[dict],
    n_grid_jobs: int,
    n_grid_complexes: int,
    params: AF2Params = PARAMS,
    gpu_type: str | None = None,
) -> dict[str, Any]:
    """Re-cost the full grid from observed pilot timings.

    The first job of a complex pays the MMseqs2 search and the rest reuse it, so
    the grid costs one cold job per complex plus warm jobs for everything else.
    Costing every job at the cold rate would overstate the grid by roughly the
    MSA overhead times four fifths of it.

    Falls back to treating every observed job as cold when the pilot did not
    contain a warm one, which is the conservative direction: it can only
    overstate, never understate, and an overstated estimate stops a run rather
    than blowing a budget.
    """
    import statistics as _statistics

    seen: set[str] = set()
    cold: list[float] = []
    warm: list[float] = []
    for result in results:
        if not isinstance(result, dict) or result.get("wall_clock_s") is None:
            continue
        pdb_id = str(result.get("pdb_id", ""))
        minutes = float(result["wall_clock_s"]) / 60.0
        if pdb_id in seen:
            warm.append(minutes)
        else:
            seen.add(pdb_id)
            cold.append(minutes)

    if not cold and not warm:
        raise ValueError(
            "the pilot returned no timed job. Nothing can be extrapolated from "
            "it, and guessing here is exactly what the pilot was meant to stop."
        )

    cold_median = _statistics.median(cold) if cold else _statistics.median(warm)
    warm_median = _statistics.median(warm) if warm else cold_median

    n_cold = min(n_grid_complexes, n_grid_jobs)
    n_warm = max(0, n_grid_jobs - n_cold)
    compute_hours = (n_cold * cold_median + n_warm * warm_median) / 60.0

    gpu_type = gpu_type or params.gpu_type
    rate = MODAL_GPU_RATES_USD_PER_HOUR[gpu_type]
    containers = min(params.max_containers, n_grid_jobs)
    cold_start_hours = containers * params.cold_start_minutes / 60.0
    gpu_hours = (compute_hours + cold_start_hours) * (1.0 + params.failure_overhead)

    return {
        "n_pilot_jobs_timed": len(cold) + len(warm),
        "observed_cold_minutes_median": cold_median,
        "observed_warm_minutes_median": warm_median,
        "observed_max_minutes": max(cold + warm),
        "msa_overhead_observed_minutes": cold_median - warm_median,
        "warm_measured": bool(warm),
        "n_grid_jobs": n_grid_jobs,
        "n_grid_cold_jobs": n_cold,
        "n_grid_warm_jobs": n_warm,
        "estimated_gpu_hours": gpu_hours,
        "estimated_usd": gpu_hours * rate,
        "estimated_wall_clock_hours": gpu_hours / max(containers, 1),
        "basis": (
            "measured from the pilot, one cold job per complex plus warm jobs for the rest"
            if warm
            else "measured from the pilot, but no warm job was observed, so every "
            "grid job is costed at the cold rate. This overstates the grid."
        ),
    }


# ---------------------------------------------------------------------------
# Mixed-mode MSAs
# ---------------------------------------------------------------------------


def msa_plan(job: Job) -> dict[str, str]:
    """Decide, per chain, whether to search for homologues.

    Returns the chain-to-mode mapping actually used, after checking it against
    the one invariant that matters: a redesigned chain must never be given an
    MSA. Searching with a design's sequence returns homologues of the *native*
    it was derived from, because the design retains enough of the native's
    profile to be found by them. AlphaFold would then be predicting the fold of
    a family the design does not belong to, and a high-charge design that ought
    to fail could be propped up into looking fine.

    The partner chain is different. It is held at its native sequence, its
    evolutionary signal is real, and dropping it would degrade the prediction
    for reasons that have nothing to do with the charge dial. Discarding it
    would make every complex look worse at every beta, which is exactly the kind
    of uniform degradation that hides a real effect.
    """
    modes = dict(job.msa_mode)
    missing = set(job.chains) - set(modes)
    if missing:
        raise ValueError(f"{job.key}: no MSA mode recorded for chain(s) {sorted(missing)}")
    unknown = {mode for mode in modes.values()} - {"msa", "single_sequence"}
    if unknown:
        raise ValueError(f"{job.key}: unrecognised MSA mode(s) {sorted(unknown)}")
    if modes.get(job.designed_chain) != "single_sequence":
        raise ValueError(
            f"{job.key}: designed chain {job.designed_chain} is set to "
            f"{modes.get(job.designed_chain)!r}. A design has no evolutionary "
            "history, so an MSA for it would return homologues of the native it "
            "was derived from and prop the prediction up with information the "
            "design does not carry."
        )
    return modes


def pairing_is_meaningful(modes: dict[str, str]) -> bool:
    """Can the chains' alignments be paired by organism?

    Only when every chain has a real MSA. Pairing matches homologues that
    co-occur in the same species, which is where AlphaFold-Multimer gets much of
    its interface signal. A designed chain has no species, so nothing can be
    paired to it and the unpaired blocks are all there is.

    This is a real cost of the mixed-mode treatment and it should be stated in
    the paper rather than buried: interface confidence for a design is derived
    from the partner's evolutionary signal plus the design's own sequence, with
    no co-evolutionary pairing across the interface.
    """
    return all(mode == "msa" for mode in modes.values())


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

        modes = msa_plan(job)
        a3m_lines = build_mixed_a3m(job, ordered, modes, cache_dir=Path(RESULTS_PATH) / "msa_cache")

        colabfold_run(
            # Passing the alignment directly is what makes the mixed treatment
            # possible. ColabFold's own ``msa_mode`` switch is all-or-nothing
            # across the complex, so going through it would force either an MSA
            # for the design or no MSA for the partner, and both are wrong.
            queries=[(job.key, query_sequence, a3m_lines)],
            result_dir=str(out_dir),
            num_models=PARAMS.num_models,
            num_recycles=PARAMS.num_recycles,
            model_type="alphafold2_multimer_v3",
            msa_mode="single_sequence",
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
                "msa_paired": pairing_is_meaningful(modes),
                "num_recycles": PARAMS.num_recycles,
                "num_models": PARAMS.num_models,
                "random_seed": PARAMS.random_seed,
                "hostname_gpu": os.environ.get("MODAL_GPU", PARAMS.gpu_type),
            }
        )

        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n")
        results_volume.commit()
        return metrics

    def build_mixed_a3m(
        job: Job, chain_order: list[str], modes: dict[str, str], cache_dir: Path
    ) -> str:
        """Assemble one alignment where some chains have homologues and some do not.

        The format is generated by ColabFold's own ``msa_to_str`` rather than
        written out here. The complex a3m has a header encoding per-chain
        lengths and cardinalities and pads every hit with gaps across the other
        chains' columns, and a version skew in any of that would not raise, it
        would silently misalign the MSA against the sequence and produce
        confident nonsense. Letting the consumer build its own input removes
        that whole class of failure.

        The search result is cached on the results volume under the complex and
        chain, **not** under the job. The partner chain is held at its native
        sequence at every beta, so all five charge settings of a complex want
        the identical alignment. Caching per job would issue one query per job
        rather than one per complex, which on this grid is 275 queries instead
        of 55. That matters twice over: the MMseqs2 server is a free shared
        resource, and the wait happens inside the GPU container, so a repeated
        search is billed at the A100 rate for doing nothing.
        """
        from colabfold.batch import msa_to_str
        from colabfold.colabfold import run_mmseqs2

        cache_dir.mkdir(parents=True, exist_ok=True)
        unpaired: list[str] = []

        for chain_id in chain_order:
            sequence = job.chains[chain_id]
            if modes[chain_id] == "single_sequence":
                # The design is its own alignment, depth one.
                unpaired.append(f">{job.key}_{chain_id}\n{sequence}\n")
                continue

            cached = cache_dir / f"{job.pdb_id}_{chain_id}.a3m"
            # Another container may have written this since this one started.
            results_volume.reload()
            if cached.is_file():
                unpaired.append(cached.read_text())
                continue

            result = run_mmseqs2(
                [sequence],
                str(cache_dir / f"mmseqs_{job.pdb_id}_{chain_id}"),
                use_env=True,
                use_filter=True,
                use_templates=False,
                use_pairing=False,
            )
            lines = result[0] if isinstance(result, list | tuple) else result
            if not lines or ">" not in lines:
                raise RuntimeError(
                    f"{job.key}: MMseqs2 returned no alignment for native chain "
                    f"{chain_id}. Refusing to fall back to single-sequence "
                    "silently, because that would change what this job measures "
                    "without changing what it is labelled."
                )
            cached.write_text(lines)
            results_volume.commit()
            unpaired.append(lines)

        # No paired block, ever, on this job set. Pairing matches homologues by
        # organism across chains, and a design belongs to no organism, so there
        # is nothing to pair it to. Every job here has exactly one designed
        # chain, so pairing_is_meaningful is uniformly false; it is recorded in
        # the metrics rather than branched on, because the limitation belongs in
        # the results table where it can be reported.
        return msa_to_str(
            unpaired_msa=unpaired,
            paired_msa=None,
            query_seqs_unique=[job.chains[c] for c in chain_order],
            query_seqs_cardinality=[1] * len(chain_order),
        )

    @app.local_entrypoint()
    def run(
        designs: str = "data/raw/designs.csv",
        test_set: str = "data/raw/test_set.csv",
        definitions: str = "results/interface_definitions.json",
        budget_usd: float = 122.0,
        wave: int = 0,
        chunk_size: int = 20,
        n_canary: int = 5,
        yes: bool = False,
        pilot: int = 0,
        pilot_betas: int = 2,
    ) -> None:
        """Launch the grid in waves, largest complexes first, gated between waves.

        ``wave=0`` runs every wave in sequence, stopping for approval after each.
        ``wave=N`` runs only wave N, which is how you resume after stopping.
        ``yes=True`` skips the approval prompts, for an unattended run.

        ``pilot=N`` runs N complexes at ``pilot_betas`` charge settings each,
        re-costs the full grid from what it observed, and stops without touching
        the rest. Its results stay on the volume and count towards the grid, so
        a pilot is never wasted work.
        """
        jobs = build_job_list(Path(designs), Path(definitions), Path(test_set))
        done = completed_keys()
        outstanding = [job for job in jobs if job.key not in done]

        print(f"{len(jobs)} job(s) total, {len(done)} already complete, {len(outstanding)} to run")
        if not outstanding:
            print("nothing to do")
            return

        if pilot:
            selected = select_pilot(outstanding, n_complexes=pilot, betas_each=pilot_betas)
            if not selected:
                print("no jobs left to pilot")
                return
            sizes = sorted(job.total_residues() for job in selected)
            print(
                f"\n{'=' * 60}\nPILOT: {len(selected)} job(s) across "
                f"{len({j.pdb_id for j in selected})} complex(es), "
                f"{sizes[0]} to {sizes[-1]} residues.\n"
                "Sized across the distribution, not just the largest, so the median "
                "extrapolates.\nEach complex runs more than one beta so the MSA search "
                "is measured cold and warm.\n"
                "These results count towards the grid; nothing here is thrown away."
            )
            pilot_results = list(
                predict.map([asdict(job) for job in selected], return_exceptions=True)
            )
            failures = [r for r in pilot_results if not isinstance(r, dict)]
            print(f"\n{len(pilot_results) - len(failures)} of {len(pilot_results)} job(s) returned")
            for result in failures[:5]:
                print(f"  FAILED: {result!r}"[:300])

            try:
                projection = extrapolate_from_pilot(
                    [r for r in pilot_results if isinstance(r, dict)],
                    n_grid_jobs=len(jobs),
                    n_grid_complexes=len({j.pdb_id for j in jobs}),
                )
            except ValueError as exc:
                print(f"\nCannot re-cost: {exc}")
                return

            print(
                "\n"
                + json.dumps(
                    {
                        k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in projection.items()
                    },
                    indent=2,
                )
            )
            headroom = budget_usd - projection["estimated_usd"]
            planned = estimate_cost(len(jobs))["estimated_usd"]
            print(
                f"\nplanning assumption said ${planned:.2f}; the pilot says "
                f"${projection['estimated_usd']:.2f}."
            )
            if headroom < 0:
                wave_one = [j for j in jobs if j.beta in WAVE_BETAS[0]]
                wave_one_cost = projection["estimated_usd"] * len(wave_one) / max(len(jobs), 1)
                print(
                    f"OVER the ${budget_usd:.2f} ceiling by ${-headroom:.2f}. Wave 1 alone "
                    f"(beta {WAVE_BETAS[0]}, {len(wave_one)} jobs) is about "
                    f"${wave_one_cost:.2f}.\nRun it with --wave 1. Wave 2 is the part to "
                    "drop: only 10 to 18 percent of those designs fold at all."
                )
            else:
                print(
                    f"WITHIN the ${budget_usd:.2f} ceiling, ${headroom:.2f} spare. "
                    "Launch the grid with --pilot 0."
                )
            if not projection["warm_measured"]:
                print(
                    "\nNote: no warm job was observed, so this costs every grid job as "
                    "though it paid\nits own MSA search. The real figure is lower."
                )
            return

        waves = split_into_waves(outstanding, WAVE_BETAS)
        guard = BudgetGuard(budget_usd, PARAMS.usd_per_gpu_hour)

        print(f"\nplan: {len(waves)} wave(s), ceiling ${budget_usd:.2f}")
        for index, wave_jobs in enumerate(waves, start=1):
            if not wave_jobs:
                continue
            betas = sorted({j.beta for j in wave_jobs})
            cost = estimate_cost(len(wave_jobs))["estimated_usd"]
            print(f"  wave {index}: {len(wave_jobs):>4} jobs, beta {betas}, about ${cost:.0f}")

        for index, wave_jobs in enumerate(waves, start=1):
            if not wave_jobs or (wave and index != wave):
                continue

            ordered = order_canary_first(wave_jobs, n_canary)
            print(f"\n{'=' * 60}\nwave {index}: {len(ordered)} job(s)")
            print(
                f"canary: the {n_canary} largest complexes go first, up to "
                f"{ordered[0].total_residues()} residues. If this wave is going to hit "
                "an out-of-memory or a timeout, it will do it in the next few minutes."
            )

            results: list = []
            for batch_number, batch in enumerate(chunked(ordered, chunk_size), start=1):
                if not guard.may_start(
                    len(batch), PARAMS.estimated_minutes_per_job + PARAMS.msa_overhead_minutes
                ):
                    print(
                        f"\nBUDGET STOP before batch {batch_number}: {guard.status()}.\n"
                        f"Starting {len(batch)} more job(s) would exceed the ceiling. "
                        "Nothing in flight was cancelled.\n"
                        "Raise --budget-usd to continue, or collect what has finished."
                    )
                    break

                print(f"\n  batch {batch_number}: {len(batch)} job(s) | {guard.status()}")
                batch_results = list(
                    predict.map([asdict(job) for job in batch], return_exceptions=True)
                )
                for result in batch_results:
                    if isinstance(result, dict) and result.get("wall_clock_s") is not None:
                        guard.record(result["wall_clock_s"])
                results.extend(batch_results)

            report = summarise_wave(index, results, guard)
            print("\n" + report.render())

            if report.success_rate < 0.5 and report.n_jobs:
                print(
                    "\nWARNING: more than half of this wave failed. Investigate before "
                    "spending anything further; a systematic failure will repeat."
                )

            remaining_waves = [w for i, w in enumerate(waves, start=1) if i > index and w]
            if wave or not remaining_waves:
                break
            if not yes:
                print(
                    f"\nNext wave is {len(remaining_waves[0])} job(s), about "
                    f"${estimate_cost(len(remaining_waves[0]))['estimated_usd']:.0f}. "
                    f"{guard.status()}"
                )
                answer = input("Continue to the next wave? [y/N] ").strip().lower()
                if answer != "y":
                    print(
                        f"Stopped after wave {index}. Everything completed is on the volume "
                        "and the run is resumable: relaunch and finished jobs are skipped."
                    )
                    break

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
    ipsae_scores: dict = {"ipsae_note": "PAE matrix absent or not shaped like the complex"}
    if pae.size and len(lengths) == 2 and pae.shape[0] == sum(lengths):
        first = lengths[0]
        cross_upper = pae[:first, first:]
        cross_lower = pae[first:, :first]
        interface_pae = float((cross_upper.mean() + cross_lower.mean()) / 2.0)
        ipsae_scores = ipsae(pae, lengths, PARAMS.ipsae_pae_cutoff_a)

    predicted_pdbs = sorted(out_dir.glob("*relaxed*.pdb")) or sorted(out_dir.glob("*.pdb"))

    metrics: dict = {
        "complex_ptm": float(scores["ptm"]) if "ptm" in scores else None,
        "interface_ptm": float(scores["iptm"]) if "iptm" in scores else None,
        "interface_pae": interface_pae,
        "mean_plddt": float(np.mean(scores["plddt"])) if scores.get("plddt") else None,
        "predicted_pdb": str(predicted_pdbs[0].name) if predicted_pdbs else None,
        "interface_rmsd_a": None,
        "interface_rmsd_note": "native structure not available in the container",
        **ipsae_scores,
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


def _d0_scalar(length: float) -> float:
    """Yang and Skolnick (2004) length normalisation, reference scalar form.

    Reproduces ``calc_d0`` in the ipSAE reference, which returns exactly 1.0 at
    or below 27 residues rather than evaluating the cube root there. This
    differs from the array form below 28 residues, and the two are kept separate
    rather than unified because the reference applies each in a specific place
    and matching it is the point.
    """
    if length > 27.0:
        return max(1.0, 1.24 * (float(length) - 15.0) ** (1.0 / 3.0) - 1.8)
    return 1.0


def _d0_array(lengths: Any) -> Any:
    """Reference ``calc_d0_array``: floors the length at 26, then the result at 1.0."""
    import numpy as np

    clamped = np.maximum(26.0, np.asarray(lengths, dtype=float))
    return np.maximum(1.0, 1.24 * (clamped - 15.0) ** (1.0 / 3.0) - 1.8)


def ipsae(
    pae: Any,
    lengths: list[int],
    pae_cutoff_a: float,
) -> dict:
    """Interface pTM with the length normalisation taken from the interface.

    Why this is here at all: ipTM applies a d0 derived from the *total* number
    of residues in the complex. d0 grows with the cube root of that total, so
    the same physical interface scores differently depending on how large the
    chains around it happen to be. Across a set of complexes spanning a wide
    range of chain lengths, ipTM is therefore not comparable between complexes,
    and a charge effect estimated across them is confounded by size.

    ipSAE (Dunbrack, 2025) computes d0 from the number of residues actually
    involved in the interface instead, which removes the dependence on the parts
    of the chains that have nothing to do with binding.

    Implemented from the reference at github.com/DunbrackLab/IPSAE. Three
    normalisations are returned, all of them restricted to cross-chain pairs
    scoring below ``pae_cutoff_a``:

    ``ipsae_d0res``
        The headline score. d0 is recomputed for every aligned residue from the
        number of partner residues it confidently places.
    ``ipsae_d0dom``
        d0 from the count of distinct residues on both sides that participate in
        any confident pair, that is, from the size of the interface as a whole.
    ``ipsae_d0chn``
        d0 from the two chain lengths, so it differs from ipTM only by the
        cutoff. Reported as the control: if a charge trend appears in this one
        as strongly as in the others, the length normalisation was not what
        mattered.

    Each is asymmetric, so both directions are returned along with the maximum,
    which is the value the reference reports.
    """
    import numpy as np

    pae = np.asarray(pae, dtype=float)
    if len(lengths) != 2:
        raise ValueError(f"ipSAE is defined for a two-chain interface, got {len(lengths)} chains")
    total = int(sum(lengths))
    if pae.ndim != 2 or pae.shape != (total, total):
        raise ValueError(
            f"PAE matrix is {pae.shape}, expected ({total}, {total}) for chains of "
            f"lengths {lengths}. Refusing to score a matrix that does not match "
            "the sequences, because slicing it wrongly would silently score the "
            "wrong residue pairs."
        )

    first = int(lengths[0])
    chain_of = np.concatenate([np.zeros(first, dtype=int), np.ones(total - first, dtype=int)])

    result: dict = {"ipsae_pae_cutoff_a": float(pae_cutoff_a)}

    for direction, (aligned, scored) in enumerate([(0, 1), (1, 0)]):
        rows = chain_of == aligned
        # valid[i, j]: pair is cross-chain in this direction and confident.
        valid = np.outer(rows, chain_of == scored) & (pae < pae_cutoff_a)
        n_per_residue = valid.sum(axis=1)

        # d0res: one d0 per aligned residue, from its own partner count.
        d0_res = _d0_array(n_per_residue)
        with np.errstate(invalid="ignore", divide="ignore"):
            ptm_res = 1.0 / (1.0 + (pae / d0_res[:, None]) ** 2)

        # d0dom: one d0 for the pair, from the number of distinct residues on
        # either side that take part in any confident pair.
        n_interface = int((valid.any(axis=1) & rows).sum() + valid.any(axis=0).sum())
        d0_dom = _d0_scalar(n_interface)
        ptm_dom = 1.0 / (1.0 + (pae / d0_dom) ** 2)

        d0_chn = _d0_scalar(total)
        ptm_chn = 1.0 / (1.0 + (pae / d0_chn) ** 2)

        counts = np.where(n_per_residue > 0, n_per_residue, 1)
        by_residue = {
            "d0res": (ptm_res * valid).sum(axis=1) / counts,
            "d0dom": (ptm_dom * valid).sum(axis=1) / counts,
            "d0chn": (ptm_chn * valid).sum(axis=1) / counts,
        }
        label = f"{aligned}to{scored}"
        for name, values in by_residue.items():
            # Residues with no confident partner score zero, matching the
            # reference, rather than being dropped.
            scores = np.where(n_per_residue > 0, values, 0.0)[rows]
            result[f"ipsae_{name}_{label}"] = float(scores.max()) if scores.size else 0.0
        result[f"ipsae_n_interface_residues_{label}"] = n_interface
        result[f"ipsae_d0dom_value_{label}"] = d0_dom
        if direction == 0:
            result["ipsae_d0chn_value"] = d0_chn

    for name in ("d0res", "d0dom", "d0chn"):
        result[f"ipsae_{name}"] = max(result[f"ipsae_{name}_0to1"], result[f"ipsae_{name}_1to0"])
    return result


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
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=122.0,
        help=(
            "Ceiling to check the estimate against. Only reports how much "
            "headroom is left; the dry run never launches anything."
        ),
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
        f"\nNOTE: {params.estimated_minutes_per_job} minutes of compute plus "
        f"{params.msa_overhead_minutes} for the MSA is a\nplanning assumption from "
        "config.AF2Params, not a measurement, and the GPU rates\nare UNVERIFIED. Run a "
        "pilot of about ten jobs, take the observed median, and\nre-run with "
        "--minutes-per-job before committing to a budget."
    )

    if not args.compare_gpus:
        estimate = estimate_cost(target, params, gpu_type=args.gpu)
        headroom = args.budget_usd - estimate["estimated_usd"]
        if headroom < 0:
            print(
                f"\nOVER BUDGET: estimated ${estimate['estimated_usd']:.2f} against a "
                f"${args.budget_usd:.2f} ceiling.\nRun wave 1 only (beta 0 and +/-1.5), "
                "or lower the MSA overhead with a pilot\nmeasurement before launching."
            )
        elif headroom < 0.15 * args.budget_usd:
            print(
                f"\nTIGHT: estimated ${estimate['estimated_usd']:.2f} against a "
                f"${args.budget_usd:.2f} ceiling leaves only\n${headroom:.2f} of "
                "headroom. Running with MSAs costs roughly "
                f"{params.msa_overhead_minutes / params.estimated_minutes_per_job:.0%} "
                "more than\nsingle-sequence would. If the pilot comes in above "
                f"{params.estimated_minutes_per_job + params.msa_overhead_minutes:.0f} "
                "minutes per job, wave 2\n(beta +/-3) is the part to drop: the upstream "
                "data shows only 10 to 18 percent\nof those designs fold at all."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
