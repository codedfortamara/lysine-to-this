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

The volume mounts at ``/weights`` and is passed to ColabFold as ``data_dir``,
which resolves parameters at ``data_dir/params``. ColabFold's ``run()`` never
downloads them; only its command line entry point does, so if they are absent
the job fails rather than quietly fetching several gigabytes per container.

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

#: Container dependencies, and the constraints they have to satisfy.
#:
#: These bounds belong to colabfold 1.5.5, which declares ``biopython<1.83``,
#: ``numpy>=1.22.0,<2.0.0`` and ``jax>=0.4.20,<0.5.0``. Asking for anything
#: outside them makes the image unresolvable, and the failure arrives several
#: minutes into a build rather than at the call site.
#:
#: Kept at module level, outside the Modal guard, so a test can check the pins
#: without Modal installed. The first version of this file asked for
#: ``biopython>=1.83`` and ``numpy>=1.26``, both of which contradict the above,
#: and nothing caught it until a real build failed.
COLABFOLD_VERSION: str = "1.5.5"
IMAGE_PACKAGES: list[str] = [
    f"colabfold[alphafold]=={COLABFOLD_VERSION}",
    "biopython<1.83",
    "numpy>=1.22,<2.0",
]
#: JAX is pinned by dm-haiku, not by colabfold, and this is the one that bites.
#:
#: colabfold 1.5.5 declares ``jax>=0.4.20,<0.5.0`` and ``dm-haiku==0.0.10``.
#: Haiku 0.0.10 imports ``jax.linear_util``, which JAX removed in 0.4.24, so
#: anything from 0.4.24 upwards satisfies colabfold's own bound and still dies
#: at import with::
#:
#:     AttributeError: module 'jax' has no attribute 'linear_util'
#:
#: Checked against the published wheels: ``jax/linear_util.py`` is present in
#: 0.4.20 through 0.4.23 and absent from 0.4.24 onwards. So 0.4.23 is the
#: highest usable version, not a cautious choice.
#:
#: The extra matters as much as the version. At 0.4.23 the ``cuda12`` extra
#: routes through the then-new plugin packages, while ``cuda12_pip`` pulls the
#: monolithic ``jaxlib==0.4.23+cuda12.cudnn89`` wheel, which is the well-trodden
#: path for this release and the one the jax-releases index actually carries a
#: cp311 build of.
JAX_PACKAGE: str = "jax[cuda12_pip]==0.4.23"

#: Present in JAX up to and including this version, removed in the next. Haiku
#: 0.0.10 needs it. Recorded so the constraint is checkable rather than folklore.
JAX_MAX_WITH_LINEAR_UTIL: str = "0.4.23"

#: Where the CUDA-tagged jaxlib actually lives.
#:
#: ``jaxlib==0.4.23+cuda12.cudnn89`` is not on PyPI. Local version identifiers
#: like ``+cuda12.cudnn89`` are not accepted there, so the CUDA builds are
#: published on this page instead, and pip only sees them when pointed at it.
#:
#: It has to be ``-f`` (find-links) rather than ``--extra-index-url``. The page
#: is a flat HTML list of wheel links, not a PEP 503 index, so an index-style
#: flag reaches it and finds nothing. The first attempt passed this URL as
#: ``--extra-index-url`` on the *colabfold* install, where it was neither needed
#: nor effective, and omitted it from the jax install, where it was essential.
JAX_FIND_LINKS: str = "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

#: cuDNN must be the 8.9 series, and nothing in the dependency graph says so.
#:
#: jaxlib 0.4.23+cuda12.cudnn89 is built against cuDNN 8.9, as the local version
#: tag states. Its declared requirement is ``nvidia-cudnn-cu12>=8.9`` with no
#: upper bound, so pip installs the 9.x series and the loader cannot find the
#: symbols it wants. The result is not an error at install time and not a crash
#: at run time. It is this, once, in the container log::
#:
#:     CUDA backend failed to initialize: Unable to load cuDNN. Is it installed?
#:
#: and then JAX quietly runs the whole of AlphaFold on CPU.
#:
#: That is the most expensive failure mode available here. Everything still
#: works and every number is correct; it is simply tens of times slower, on a
#: machine billed at 2.10 USD an hour. It is what turned three minute jobs into
#: thirty minute ones, what made the largest complexes hit the two hour timeout
#: having produced nothing, and what made every cost estimate built on those
#: timings wrong.
#:
#: A silent fall back to a working-but-wrong configuration is exactly the class
#: of failure this project refuses everywhere else. It should have been refused
#: here too, and now is: the pin below forces the matching series, and predict
#: checks at run time that a GPU is actually in use before folding anything.
CUDNN_PACKAGE: str = "nvidia-cudnn-cu12>=8.9,<9.0"

#: colabfold 1.5.5 declares ``requires_python >=3.9,<3.12``.
IMAGE_PYTHON_VERSION: str = "3.11"

#: The AlphaFold parameter release. One constant: the weights downloaded into
#: the volume and the model requested at prediction time must be the same set,
#: and a mismatch is a wasted download followed by a failed run.
MODEL_TYPE: str = "alphafold2_multimer_v3"

#: Identifies this project to the free MMseqs2 server. ColabFold warns when it
#: is unset and says the warning will become an error, and it is basic courtesy
#: on a shared public resource.
MMSEQS_USER_AGENT: str = "interface-charge-rcsb/1.0 (AF2-Multimer charge study)"


def require_parseable_complex_a3m(a3m: str, chain_lengths: list[int]) -> None:
    """Refuse an alignment ColabFold would quietly reinterpret.

    ColabFold's ``unserialize_msa`` checks that the first line begins with ``#``
    and splits into exactly two tab-separated fields. If it does not, it does
    **not** raise: it falls through to a single-sequence branch and returns an
    alignment of depth one. Every downstream number then looks normal while the
    MSA has been discarded, which on this grid would mean paying for the whole
    run and reporting results the run did not actually produce.

    Silent reinterpretation is the failure mode worth spending code on, so the
    header is checked here against the same conditions, and against the chain
    lengths the alignment claims to describe.
    """
    lines = a3m.replace("\x00", "").splitlines()
    if len(lines) < 3:
        raise ValueError(
            f"the assembled a3m has {len(lines)} line(s); ColabFold requires at "
            "least three (header, query name, query sequence)."
        )
    header = lines[0]
    if not header.startswith("#"):
        raise ValueError(
            f"the a3m header is {header[:40]!r}. ColabFold expects it to start "
            "with '#', and silently treats anything else as a single sequence."
        )
    fields = header[1:].split("\t")
    if len(fields) != 2:
        raise ValueError(
            f"the a3m header splits into {len(fields)} tab-separated field(s), "
            "expected exactly two (lengths, cardinalities). ColabFold would "
            "discard the alignment rather than reject it."
        )
    declared = [int(v) for v in fields[0].split(",")]
    if declared != list(chain_lengths):
        raise ValueError(
            f"the a3m header declares chain lengths {declared} but the job's "
            f"chains are {list(chain_lengths)}. The alignment would be sliced "
            "against the wrong residues."
        )
    cardinality = [int(v) for v in fields[1].split(",")]
    if len(cardinality) != len(declared):
        raise ValueError(
            f"the a3m header declares {len(declared)} chain length(s) but "
            f"{len(cardinality)} cardinality value(s)."
        )


#: Repository packages the container needs, over and above this file.
#:
#: Modal mounts the entrypoint module automatically and nothing else, so a
#: package imported from ``src/`` is present locally and absent remotely. The
#: ``sys.path`` insert at the top of this file hides that completely during a
#: dry run and every local test, and the failure only appears once containers
#: start: each one crash-loops on ``ModuleNotFoundError`` after the image has
#: already built and the GPUs have already been allocated.
#:
#: Declared here so a test can check it against this module's own imports.
LOCAL_PYTHON_SOURCES: list[str] = ["interface_charge"]


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

    # Scale the measured per-job time by complex size when real jobs are given.
    #
    # AlphaFold attention is quadratic in sequence length, and this set spans
    # 165 to 1257 residues, so one flat per-job figure cannot describe it. The
    # measurement available is always the pessimistic one, because canary
    # ordering runs the largest complexes first on purpose: 32.2 minutes came
    # from jobs with a root-mean-square length of 905 against a grid median of
    # 395. Applied flat that overstates this grid roughly fourfold.
    size_factor = 1.0
    if not isinstance(jobs, int) and jobs:
        mean_square = sum(job.total_residues() ** 2 for job in jobs) / len(jobs)
        size_factor = mean_square / (params.timing_reference_residues**2)
    rate = MODAL_GPU_RATES_USD_PER_HOUR.get(gpu_type)
    if rate is None:
        raise ValueError(
            f"no published rate recorded for GPU {gpu_type!r}; "
            f"known: {sorted(MODAL_GPU_RATES_USD_PER_HOUR)}"
        )

    containers = min(params.max_containers, n_jobs)
    minutes_per_job = (params.estimated_minutes_per_job * size_factor) + params.msa_overhead_minutes
    compute_hours = n_jobs * minutes_per_job / 60.0
    cold_start_hours = containers * params.cold_start_minutes / 60.0
    # Container time that belongs to no job and is not a cold start: containers
    # held open waiting for the slowest job in a chunk. Invisible in every job's
    # own wall clock, and about half the bill on the first real batch.
    idle_hours = max(0.0, compute_hours * (params.billing_overhead - 1.0) - cold_start_hours)
    gpu_hours = (compute_hours + cold_start_hours + idle_hours) * (1.0 + params.failure_overhead)

    out: dict[str, Any] = {
        "idle_gpu_hours": idle_hours,
        "billing_overhead": params.billing_overhead,
        "size_factor": size_factor,
        "size_factor_basis": (
            "mean squared complex length against the size the per-job time was "
            "measured at; 1.0 when only a job count was supplied, which is "
            "pessimistic for this set"
        ),
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


def split_plan(jobs: list[Job], colab_max_residues: int = 700) -> dict[str, Any]:
    """Cost the division of labour between a free T4 and paid A100 time.

    AlphaFold-Multimer compute grows roughly with the square of total length,
    so this set is heavily top weighted: a handful of large complexes carry most
    of the bill. They are also exactly the ones a 16 GB card cannot hold.

    The two facts point the same way. Give the free GPU the many small jobs,
    where it is slow but costs nothing, and spend money only on the few large
    ones that have nowhere else to go. This reports what that costs, and what
    fraction of the science each side carries, since the split is only
    acceptable if the expensive half is small.

    Cost shares use length squared. That is a rule of thumb, not a measurement,
    and it is used here only to divide a total, never to predict one.
    """
    per_complex: dict[str, int] = {}
    for job in jobs:
        per_complex.setdefault(job.pdb_id, job.total_residues())
    if not per_complex:
        return {"n_complexes": 0}

    weight = {pdb: float(length) ** 2 for pdb, length in per_complex.items()}
    total_weight = sum(weight.values())
    small = {p for p, n in per_complex.items() if n <= colab_max_residues}
    large = set(per_complex) - small

    colab_jobs = [j for j in jobs if j.pdb_id in small]
    modal_jobs = [j for j in jobs if j.pdb_id in large]

    return {
        "colab_max_residues": colab_max_residues,
        "colab": {
            "n_complexes": len(small),
            "n_jobs": len(colab_jobs),
            "cost_share": sum(weight[p] for p in small) / total_weight,
        },
        "modal": {
            "n_complexes": len(large),
            "n_jobs": len(modal_jobs),
            "cost_share": sum(weight[p] for p in large) / total_weight,
            "complexes": sorted(large, key=lambda p: -per_complex[p]),
            "estimated_usd": estimate_cost(len(modal_jobs))["estimated_usd"] if modal_jobs else 0.0,
        },
    }


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

    def __init__(
        self,
        ceiling_usd: float,
        usd_per_gpu_hour: float,
        billing_overhead: float = PARAMS.billing_overhead,
    ) -> None:
        if ceiling_usd <= 0:
            raise ValueError(f"ceiling_usd must be positive, got {ceiling_usd}")
        if billing_overhead < 1.0:
            raise ValueError(
                f"billing_overhead must be at least 1.0, got {billing_overhead}. "
                "Below one it would claim Modal bills less than the jobs consume."
            )
        self.ceiling_usd = ceiling_usd
        self.usd_per_gpu_hour = usd_per_gpu_hour
        self.billing_overhead = billing_overhead
        self.gpu_seconds = 0.0
        self.n_recorded = 0

    def record(self, wall_clock_s: float) -> None:
        """Account for one completed job."""
        self.gpu_seconds += max(0.0, float(wall_clock_s))
        self.n_recorded += 1

    @property
    def spent_usd(self) -> float:
        """Estimated *billed* spend, not the sum of the jobs' own wall clocks.

        The two differ by about a factor of two, and the difference is all
        container time no job accounts for: image pulls, loading the model
        parameters, and containers held idle waiting for the slowest job in a
        chunk. Reporting the raw job total made this guard read $12.41 against
        roughly $26.00 actually billed, so a ceiling set at $54 would not have
        stopped anything until well past $100.
        """
        return self.gpu_seconds / 3600.0 * self.usd_per_gpu_hour * self.billing_overhead

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
        projected = n_jobs * minutes / 60.0 * self.usd_per_gpu_hour * self.billing_overhead
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


def require_gpu_backend() -> None:
    """Refuse to fold on CPU inside a container that is billed as a GPU.

    JAX does not fail when its CUDA backend cannot initialise. It logs one line
    and silently uses CPU. AlphaFold then runs correctly and tens of times more
    slowly, on hardware costing 2.10 USD an hour, and the only symptom is the
    invoice. That happened here: a cuDNN version mismatch sent an entire grid
    to CPU, jobs that should have taken three minutes took thirty, the largest
    complexes hit the two hour timeout without producing anything, and every
    cost estimate derived from those timings was wrong.

    A wrong-but-working configuration is the failure this project refuses
    everywhere else, so it is refused here. Ninety seconds of a dead container
    is cheap; two hours of one is not.
    """
    import jax

    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(
            "JAX has no GPU device; it would run AlphaFold on CPU at GPU prices.\n"
            f"Devices visible: {devices}\n"
            "Look in the container log for 'CUDA backend failed to initialize'. "
            "The usual cause is a cuDNN series mismatch: jaxlib "
            "0.4.23+cuda12.cudnn89 needs the 8.9 series, and its own dependency "
            "declaration does not pin the upper bound, so pip installs 9.x."
        )


def searches_required(jobs: list[Job]) -> list[tuple[str, str, str]]:
    """Every distinct homology search the grid needs, as (pdb_id, chain, sequence).

    One entry per native chain per complex, not per job. The partner is held at
    its native sequence at every charge setting, so all five betas of a complex
    want the identical alignment: on this grid that is 55 searches rather than
    275.

    Exists as a free function so that the prefetch can be planned, counted and
    tested without Modal installed and without a network.

    Raises if the same (complex, chain) appears with two different sequences.
    Silently caching one of them would give some jobs an alignment built from
    the wrong sequence, which AlphaFold would happily consume.
    """
    seen: dict[tuple[str, str], str] = {}
    for job in jobs:
        for chain_id, mode in msa_plan(job).items():
            if mode != "msa":
                continue
            key = (job.pdb_id, chain_id)
            sequence = job.chains[chain_id]
            previous = seen.setdefault(key, sequence)
            if previous != sequence:
                raise ValueError(
                    f"{job.pdb_id} chain {chain_id} appears with two different "
                    f"native sequences ({len(previous)} and {len(sequence)} "
                    "residues). One cache entry cannot serve both, and reusing "
                    "either would align some jobs against the wrong sequence."
                )
    return [(pdb_id, chain_id, seq) for (pdb_id, chain_id), seq in sorted(seen.items())]


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
        modal.Image.debian_slim(python_version=IMAGE_PYTHON_VERSION)
        .apt_install("git", "wget", "build-essential")
        # ColabFold provides an AlphaFold2-Multimer implementation with a
        # single-sequence mode, which is what the redesigned chains need. All
        # of these are on PyPI, so no extra index is involved.
        .pip_install(*IMAGE_PACKAGES)
        # JAX is separate because the CUDA-tagged jaxlib is not on PyPI and has
        # to be found on the jax-releases page. This is the install that needs
        # the find-links, and the one that was missing it.
        .pip_install(JAX_PACKAGE, CUDNN_PACKAGE, extra_options=JAX_FIND_LINKS)
        .env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "TF_FORCE_UNIFIED_MEMORY": "1"})
        # Ship the project's own package. Modal mounts the entrypoint module and
        # nothing else, so without this the container has af2_multimer.py and no
        # interface_charge to import from it. Local source is attached after the
        # build layers, so editing it does not rebuild the colabfold image.
        .add_local_python_source(*LOCAL_PYTHON_SOURCES)
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
    def predict(job_payload: dict, allow_msa_search: bool = False) -> dict:
        """Refold one complex and return interface-resolved metrics.

        Writes the raw predicted PDB and a metrics JSON to the results volume
        before returning, so that a lost return value never costs a rerun.
        """
        import os
        import time

        from colabfold.batch import run as colabfold_run

        require_gpu_backend()

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
        msa_started = time.time()
        a3m = build_mixed_a3m(
            job,
            ordered,
            modes,
            cache_dir=Path(RESULTS_PATH) / "msa_cache",
            # A cache miss must not turn into an MMseqs2 search here. The search
            # is an HTTP poll against a free shared server that queues under
            # load, and this container is an A100. That is what the two-hour
            # timeouts in the first pilot were: jobs sitting in a queue, billed
            # at GPU rates, until the timeout killed them with nothing produced.
            # Run ``prefetch`` first; it does the identical searches on CPU.
            allow_search=allow_msa_search,
        )
        msa_seconds = time.time() - msa_started
        require_parseable_complex_a3m(a3m, [len(job.chains[c]) for c in ordered])

        colabfold_run(
            # Passing the alignment directly is what makes the mixed treatment
            # possible. ColabFold's own ``msa_mode`` switch is all-or-nothing
            # across the complex, so going through it would force either an MSA
            # for the design or no MSA for the partner, and both are wrong.
            #
            # The alignment goes in as a one-element LIST, not a bare string.
            # ColabFold indexes it as ``a3m_lines[0]``, so a string yields its
            # first character, "#", which fails the complex-a3m check and falls
            # through to a single-sequence fallback. That path does not raise:
            # it would have run the whole grid against an empty alignment and
            # returned confident-looking numbers with the MSA silently
            # discarded.
            queries=[(job.key, query_sequence, [a3m])],
            result_dir=str(out_dir),
            num_models=PARAMS.num_models,
            num_recycles=PARAMS.num_recycles,
            model_type=MODEL_TYPE,
            msa_mode="single_sequence",
            use_templates=False,
            random_seed=PARAMS.random_seed,
            is_complex=True,
            rank_by="multimer",
            # ColabFold's run() never downloads parameters; only its command
            # line entry point does. Without this it looks in its default data
            # directory, finds nothing, and fails. The weights volume is the
            # whole reason cold starts are affordable.
            data_dir=WEIGHTS_PATH,
            user_agent=MMSEQS_USER_AGENT,
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
                # Kept apart from wall clock deliberately. In the first pilot
                # these were the same number and there was no way to see it: the
                # alignment wait and the folding were both "the job took two
                # hours". Recorded separately, a prefetched run shows
                # msa_seconds near zero and the rest is real compute.
                "msa_seconds": round(msa_seconds, 1),
                "fold_seconds": round(time.time() - started - msa_seconds, 1),
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
        job: Job,
        chain_order: list[str],
        modes: dict[str, str],
        cache_dir: Path,
        allow_search: bool = True,
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

            if not allow_search:
                raise RuntimeError(
                    f"{job.key}: no cached alignment for {job.pdb_id} chain "
                    f"{chain_id}, and searching is disabled on the GPU path.\n"
                    "MMseqs2 here is an HTTP poll against a free shared server "
                    "that queues for tens of minutes under load, and this "
                    "container is billed at GPU rates for the whole wait. Fill "
                    "the cache on CPU first:\n\n"
                    "  modal run modal_app/af2_multimer.py::prefetch\n\n"
                    "Pass --allow-msa-search to override, knowing the cost."
                )

            result = run_mmseqs2(
                [sequence],
                str(cache_dir / f"mmseqs_{job.pdb_id}_{chain_id}"),
                use_env=True,
                use_filter=True,
                use_templates=False,
                use_pairing=False,
                user_agent=MMSEQS_USER_AGENT,
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

        # The paired block contains the query and nothing else.
        #
        # No homologue can be paired here: pairing matches homologues by
        # organism across chains, and a design belongs to no organism. But
        # passing paired_msa=None produces an alignment with no row spanning
        # both chains at all, and AlphaFold-Multimer's feature builder rejects
        # that outright:
        #
        #     ValueError: MSA 0 must contain at least one sequence
        #
        # ColabFold does the same thing for homooligomers, pairing the query
        # with itself. So the paired block is depth one, carrying the query row
        # the multimer path requires and no co-evolutionary information, which
        # is exactly the limitation already documented. The partner's homologues
        # are still in the unpaired block where they belong.
        #
        # pair_sequences joins these line by line and rewrites ">" as a tab for
        # every chain after the first, which is what produces the ">101\t102"
        # header that marks a row as paired.
        paired = [
            f">{101 + index}\n{job.chains[chain_id]}\n"
            for index, chain_id in enumerate(chain_order)
        ]

        return msa_to_str(
            unpaired_msa=unpaired,
            paired_msa=paired,
            query_seqs_unique=[job.chains[c] for c in chain_order],
            query_seqs_cardinality=[1] * len(chain_order),
        )

    @app.function(
        image=image,
        volumes={RESULTS_PATH: results_volume},
        timeout=TIMEOUT_S,
        # Deliberately low. The other side of this is api.colabfold.com, which
        # is free, shared and run by people who did not agree to absorb this
        # grid. Twenty containers submitting at once is also self-defeating:
        # the server throttles and every one of them waits longer.
        max_containers=4,
        retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    )
    def fetch_msa(pdb_id: str, chain_id: str, sequence: str) -> dict:
        """Search MMseqs2 for one native chain and cache the alignment. CPU only.

        This is the same search ``predict`` used to do inline, moved off the
        GPU. Nothing about the alignment changes: same server, same flags, same
        cache path, so a run with a warm cache is identical to one without,
        minus the bill.

        The bill is the point. A CPU container costs a few cents an hour and an
        A100 costs two dollars, and the wait is the same wait either way.
        """
        import time

        from colabfold.colabfold import run_mmseqs2

        cache_dir = Path(RESULTS_PATH) / "msa_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{pdb_id}_{chain_id}.a3m"

        results_volume.reload()
        if cached.is_file():
            return {"pdb_id": pdb_id, "chain_id": chain_id, "cached": True, "seconds": 0.0}

        started = time.time()
        result = run_mmseqs2(
            [sequence],
            str(cache_dir / f"mmseqs_{pdb_id}_{chain_id}"),
            use_env=True,
            use_filter=True,
            use_templates=False,
            use_pairing=False,
            user_agent=MMSEQS_USER_AGENT,
        )
        lines = result[0] if isinstance(result, list | tuple) else result
        if not lines or ">" not in lines:
            raise RuntimeError(
                f"MMseqs2 returned no alignment for {pdb_id} chain {chain_id}. "
                "Refusing to write an empty cache entry, which would send every "
                "beta of this complex to the GPU with a silently degraded MSA."
            )
        cached.write_text(lines)
        results_volume.commit()
        return {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "cached": False,
            "seconds": round(time.time() - started, 1),
            "depth": lines.count(">"),
        }

    @app.function(
        image=image,
        volumes={RESULTS_PATH: results_volume},
        # Long enough for the whole grid with room to spare. This container
        # holds no GPU and costs pennies an hour, so a generous ceiling here is
        # not a generous bill.
        timeout=43200,
    )
    def drive(job_payloads: list[dict], budget_usd: float, chunk_size: int) -> dict:
        """Run the whole grid from inside Modal, so no laptop is in the loop.

        The local launcher submits work batch by batch and holds the budget
        guard in memory. That makes a laptop part of the apparatus: a dropped
        wifi connection, a closed lid or a reboot stops the grid, and on this
        run it did so three times. ``--detach`` keeps the *containers* alive but
        not the thing that decides what to start next, so the grid still halts
        at the end of the batch in flight.

        Moving the loop here removes the dependency entirely. The launcher
        spawns this and exits; the run continues whether or not anything is
        listening. Progress is written to the results volume after every batch,
        so ``status`` can report on it from any machine at any time, including
        one that was switched off while the work happened.

        The budget guard and canary ordering are unchanged. They now run beside
        the work rather than across an internet connection from it.
        """
        import time

        jobs = [Job(**payload) for payload in job_payloads]
        progress_path = Path(RESULTS_PATH) / "progress.json"
        guard = BudgetGuard(budget_usd, PARAMS.usd_per_gpu_hour)
        started = time.time()
        completed = 0
        failures: list[str] = []
        stopped_early = ""

        def publish(state: str) -> None:
            progress_path.write_text(
                json.dumps(
                    {
                        "state": state,
                        "n_jobs": len(jobs),
                        "n_completed_this_run": completed,
                        "n_failed_this_run": len(failures),
                        "spent_usd_estimated": round(guard.spent_usd, 2),
                        "ceiling_usd": budget_usd,
                        "observed_minutes_per_job": guard.observed_minutes_per_job(),
                        "elapsed_minutes": round((time.time() - started) / 60.0, 1),
                        "stopped_early": stopped_early,
                        "recent_failures": failures[-5:],
                    },
                    indent=2,
                    default=str,
                )
                + "\n"
            )
            results_volume.commit()

        publish("running")

        for wave_jobs in split_into_waves(jobs, WAVE_BETAS):
            if not wave_jobs or stopped_early:
                continue
            ordered = order_canary_first(wave_jobs, n_canary=5)
            for batch in chunked(ordered, chunk_size):
                if not guard.may_start(len(batch), PARAMS.estimated_minutes_per_job):
                    stopped_early = (
                        f"budget ceiling ${budget_usd:.2f} reached; "
                        f"{guard.status()}. Nothing in flight was cancelled."
                    )
                    break
                for result in predict.map(
                    [asdict(job) for job in batch],
                    kwargs={"allow_msa_search": False},
                    return_exceptions=True,
                ):
                    if isinstance(result, dict) and result.get("wall_clock_s") is not None:
                        guard.record(result["wall_clock_s"])
                        completed += 1
                    else:
                        failures.append(repr(result)[:200])
                publish("running")

        publish("finished")
        return {
            "n_completed_this_run": completed,
            "n_failed_this_run": len(failures),
            "spent_usd_estimated": round(guard.spent_usd, 2),
            "observed_minutes_per_job": guard.observed_minutes_per_job(),
            "elapsed_minutes": round((time.time() - started) / 60.0, 1),
            "stopped_early": stopped_early,
        }

    @app.function(
        image=image,
        volumes={WEIGHTS_PATH: weights_volume},
        timeout=3600,
    )
    def populate_weights() -> dict:
        """Download the AlphaFold parameters into the weights volume. Run once.

        ColabFold's ``run()`` never downloads parameters; only its command line
        entry point does. So the volume has to be filled before any prediction,
        and an empty volume means every GPU job fails after allocation.

        Done here rather than with ``modal volume put`` because the download
        happens on Modal's network straight into the volume, instead of several
        gigabytes coming down to a laptop and going back up again.

        No GPU is requested: this is a download, and paying A100 rates to wait on
        a file transfer would be careless.
        """
        import time

        from colabfold.download import download_alphafold_params

        target = Path(WEIGHTS_PATH)
        marker = target / "params" / "download_complexes_multimer_v3_finished.txt"
        if marker.is_file():
            existing = sorted(p.name for p in (target / "params").glob("*.npz"))
            return {"already_present": True, "n_param_files": len(existing)}

        started = time.time()
        download_alphafold_params(MODEL_TYPE, target)
        weights_volume.commit()

        files = sorted((target / "params").glob("*.npz"))
        if not files:
            raise RuntimeError(
                f"no parameter files landed in {target / 'params'} after the "
                "download reported success. Refusing to report a populated "
                "volume that would fail on the first prediction."
            )
        return {
            "already_present": False,
            "n_param_files": len(files),
            "total_gb": round(sum(f.stat().st_size for f in files) / 1e9, 2),
            "seconds": round(time.time() - started, 1),
        }

    @app.local_entrypoint()
    def setup() -> None:
        """One-time: fill the weights volume. Safe to re-run, it checks first."""
        print(f"populating {PARAMS.weights_volume} with {MODEL_TYPE} parameters")
        print("this runs on Modal's network, on CPU, and is a one-off")
        result = populate_weights.remote()
        if result.get("already_present"):
            print(f"already populated: {result['n_param_files']} parameter file(s), nothing to do")
        else:
            print(
                f"downloaded {result['n_param_files']} parameter file(s), "
                f"{result['total_gb']} GB, in {result['seconds']}s"
            )
        print("\nnow fill the alignment cache, also on CPU:")
        print("  modal run modal_app/af2_multimer.py::prefetch")

    @app.local_entrypoint()
    def prefetch(
        designs: str = "data/raw/designs.csv",
        test_set: str = "data/raw/test_set.csv",
        definitions: str = "results/interface_definitions.json",
    ) -> None:
        """Fill the alignment cache on CPU, before any GPU is allocated.

        This is a prerequisite of ``run``, not an optimisation. The first pilot
        did these searches inside the A100 containers and three of eight jobs
        hit the two-hour timeout without folding anything, because
        api.colabfold.com queues under load and ``run_mmseqs2`` polls until it
        is served. The searches are identical here; only the machine underneath
        them is cheap.

        One search per native chain per complex, not per job, so the grid needs
        55 of them rather than 275.
        """
        jobs = build_job_list(Path(designs), Path(definitions), Path(test_set))
        wanted = searches_required(jobs)
        print(f"{len(jobs)} job(s) over {len({j.pdb_id for j in jobs})} complex(es)")
        print(f"{len(wanted)} distinct alignment(s) needed, one per native chain per complex")
        print("running on CPU, four at a time, to stay polite to a free shared server\n")

        results = list(fetch_msa.starmap(wanted, order_outputs=False))

        already = [r for r in results if r["cached"]]
        fetched = [r for r in results if not r["cached"]]
        print(f"\n{len(already)} already cached, {len(fetched)} newly searched")
        if fetched:
            waits = sorted(r["seconds"] for r in fetched)
            depths = sorted(r["depth"] for r in fetched)
            print(
                f"search wait: median {waits[len(waits) // 2] / 60:.1f} min, "
                f"max {waits[-1] / 60:.1f} min"
            )
            print(f"alignment depth: median {depths[len(depths) // 2]}, min {depths[0]}")
            print(
                "\nAll of that wait would have been billed at the GPU rate if it "
                "had happened inside predict."
            )
        print("\nnow the grid can run with no network in the GPU containers:")
        print("  modal run modal_app/af2_multimer.py::run --pilot 4")

    @app.local_entrypoint()
    def launch(
        designs: str = "data/raw/designs.csv",
        test_set: str = "data/raw/test_set.csv",
        definitions: str = "results/interface_definitions.json",
        budget_usd: float = 250.0,
        chunk_size: int = 20,
        min_residues: int = 0,
        max_residues: int = 0,
        only_betas: str = "",
    ) -> None:
        """Start the grid on Modal and exit. Nothing needs to stay connected.

        Use this rather than ``run`` when the machine launching it cannot be
        relied on to stay awake and online for several hours, which on this
        project turned out to be every time.
        """
        if not weights_present():
            print(
                f"The weights volume {PARAMS.weights_volume!r} is empty. Run "
                "setup first:\n  modal run modal_app/af2_multimer.py::setup"
            )
            return

        jobs = build_job_list(Path(designs), Path(definitions), Path(test_set))

        cached = cached_alignments()
        missing = [
            f"{pdb} {chain}"
            for pdb, chain, _ in searches_required(jobs)
            if f"{pdb}_{chain}.a3m" not in cached
        ]
        if missing:
            print(
                f"{len(missing)} alignment(s) are not cached, starting with "
                f"{missing[:5]}.\nFetch them on CPU first, it is the same search on a "
                "machine that costs cents:\n  modal run modal_app/af2_multimer.py::prefetch"
            )
            return

        done = completed_keys()
        outstanding = [job for job in jobs if job.key not in done]
        if min_residues:
            outstanding = [j for j in outstanding if j.total_residues() >= min_residues]
        if max_residues:
            outstanding = [j for j in outstanding if j.total_residues() <= max_residues]
        if only_betas:
            wanted = {float(b) for b in only_betas.replace(" ", "").split(",")}
            outstanding = [j for j in outstanding if j.beta in wanted]

        print(f"{len(jobs)} job(s) in the grid, {len(done)} already on the volume")
        if not outstanding:
            print("nothing outstanding. Collect what is there:")
            print("  modal volume get interface-charge-af2-results / ./modal_output")
            return

        estimate = estimate_cost(outstanding)
        print(
            f"{len(outstanding)} to run, about ${estimate['estimated_usd']:.0f} and "
            f"{estimate['estimated_wall_clock_hours']:.1f} h wall clock, "
            f"ceiling ${budget_usd:.2f}"
        )

        call = drive.spawn([asdict(job) for job in outstanding], budget_usd, chunk_size)
        print(f"\nstarted on Modal as {call.object_id}")
        print("This machine is no longer involved. Close the laptop, lose wifi, reboot.")
        print("\ncheck on it any time, from anywhere:")
        print("  modal run modal_app/af2_multimer.py::progress")

    @app.local_entrypoint()
    def progress() -> None:
        """What the grid is doing, read from the volume rather than a live client."""
        done = completed_keys()
        print(f"{len(done)} job(s) complete on {PARAMS.results_volume}")

        try:
            blob = b"".join(results_volume.read_file("progress.json"))
        except VOLUME_MISSING:
            print("no progress.json yet; either nothing has been launched or the first")
            print("batch has not finished. A launched run writes it after every batch.")
            return

        report = json.loads(blob)
        print(json.dumps(report, indent=2))
        if report.get("state") == "finished":
            print("\nfinished. Collect it:")
            print("  modal volume get interface-charge-af2-results / ./modal_output")
            print("  python modal_app/collect.py --from-dir modal_output")
        if report.get("stopped_early"):
            print(f"\nstopped early: {report['stopped_early']}")

    @app.local_entrypoint()
    def timings() -> None:
        """Report what the completed jobs actually cost, and re-cost the grid.

        The planning figure in ``config.AF2Params`` is an assumption and has
        always said so. This replaces it with the measurement, taken from the
        jobs already banked on the results volume, and separates the alignment
        wait from the folding so that a pre-prefetch job is not mistaken for a
        slow one.
        """
        records = completed_metrics()
        if not records:
            print("no completed jobs on the results volume yet, so nothing to measure")
            return

        def stat(values: list[float]) -> str:
            values = sorted(values)
            return (
                f"median {values[len(values) // 2] / 60:.1f} min, "
                f"min {values[0] / 60:.1f}, max {values[-1] / 60:.1f}"
            )

        wall = [float(r["wall_clock_s"]) for r in records if "wall_clock_s" in r]
        # Older records predate the split and carry no msa_seconds. They are
        # reported as unattributed rather than folded into the fold time, which
        # would understate the saving the prefetch makes.
        split = [r for r in records if "msa_seconds" in r]

        print(f"{len(records)} completed job(s) on {PARAMS.results_volume}\n")
        print(f"wall clock:  {stat(wall)}")
        if split:
            print(f"  alignment: {stat([float(r['msa_seconds']) for r in split])}")
            print(f"  folding:   {stat([float(r['fold_seconds']) for r in split])}")
            basis = sorted(float(r["fold_seconds"]) for r in split)
        else:
            print(
                f"  ({len(records)} record(s) predate the alignment/folding split, "
                "so their wall clock still includes the MMseqs2 wait)"
            )
            basis = sorted(wall)

        minutes = basis[len(basis) // 2] / 60.0
        print(f"\nre-costing the grid at the observed median of {minutes:.1f} min per job")
        print("(folding only: with the cache warm, the alignment wait is off the GPU)\n")

        jobs = build_job_list(
            Path("data/raw/designs.csv"),
            Path("results/interface_definitions.json"),
            Path("data/raw/test_set.csv"),
        )
        outstanding = [j for j in jobs if j.key not in completed_keys()]
        measured = replace(PARAMS, estimated_minutes_per_job=minutes, msa_overhead_minutes=0.0)
        estimate = estimate_cost(len(outstanding), params=measured)
        print(f"{len(outstanding)} job(s) outstanding")
        print(f"estimated ${estimate['estimated_usd']:.2f} on {estimate['gpu_type']}")
        print(f"estimated {estimate['estimated_wall_clock_hours']:.1f} h wall clock")

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
        min_residues: int = 0,
        max_residues: int = 0,
        only_betas: str = "",
        allow_msa_search: bool = False,
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
        # Checked before anything is dispatched. An empty weights volume fails
        # every job, and it fails them after the containers have started and the
        # GPUs have been allocated, which is the expensive place to find out.
        if not weights_present():
            print(
                f"\nThe weights volume {PARAMS.weights_volume!r} has no AlphaFold "
                "parameters in it.\nColabFold does not download them at prediction "
                "time, so every job would fail\nafter its GPU was allocated. Populate "
                "it first, once:\n\n"
                "  modal run modal_app/af2_multimer.py::setup\n\n"
                "That runs on CPU, on Modal's network, and takes a few minutes."
            )
            return

        jobs = build_job_list(Path(designs), Path(definitions), Path(test_set))

        # Checked here for the same reason the weights are: a cold cache does
        # not fail a job, it makes every job sit in an MMseqs2 queue with an
        # A100 attached to it. That is what the first pilot did, and it is the
        # single most expensive way this run can go wrong.
        if not allow_msa_search:
            missing = [
                (pdb_id, chain_id)
                for pdb_id, chain_id, _ in searches_required(jobs)
                if f"{pdb_id}_{chain_id}.a3m" not in cached_alignments()
            ]
            if missing:
                shown = ", ".join(f"{p} {c}" for p, c in missing[:6])
                more = f" and {len(missing) - 6} more" if len(missing) > 6 else ""
                print(
                    f"\n{len(missing)} of {len(searches_required(jobs))} alignment(s) "
                    f"are not cached: {shown}{more}.\n\n"
                    "Fetch them on CPU first. It is the same search, on a machine "
                    "that costs cents\nrather than dollars an hour:\n\n"
                    "  modal run modal_app/af2_multimer.py::prefetch\n\n"
                    "Pass --allow-msa-search to run anyway and pay GPU rates for "
                    "the wait."
                )
                return

        done = completed_keys()
        outstanding = [job for job in jobs if job.key not in done]

        print(f"{len(jobs)} job(s) total, {len(done)} already complete, {len(outstanding)} to run")

        # Size and charge filters, for splitting the grid across backends.
        #
        # AlphaFold-Multimer compute grows roughly with the square of total
        # length, so this set is very top heavy: the five largest complexes are
        # about 39 percent of the whole bill and the ten largest about 55. Those
        # same complexes are the ones a 16 GB card cannot fit at all.
        #
        # That makes the division of labour obvious rather than arbitrary. A
        # free T4 runs the many small jobs, where it is slow but costs nothing,
        # and paid A100 time is spent only on the few large ones that have no
        # other home. --min-residues is what makes Modal pick up exactly what
        # Colab recorded as too large.
        before = len(outstanding)
        if min_residues:
            outstanding = [j for j in outstanding if j.total_residues() >= min_residues]
        if max_residues:
            outstanding = [j for j in outstanding if j.total_residues() <= max_residues]
        if only_betas:
            wanted = {float(b) for b in only_betas.replace(" ", "").split(",")}
            outstanding = [j for j in outstanding if j.beta in wanted]

        if len(outstanding) != before:
            bounds = []
            if min_residues:
                bounds.append(f"at least {min_residues} residues")
            if max_residues:
                bounds.append(f"at most {max_residues} residues")
            if only_betas:
                bounds.append(f"beta in {sorted(wanted)}")
            print(
                f"filtered to {len(outstanding)} job(s) over "
                f"{len({j.pdb_id for j in outstanding})} complex(es): {', '.join(bounds)}.\n"
                f"{before - len(outstanding)} job(s) excluded here; run them elsewhere or "
                "record the shortfall."
            )
        if not outstanding:
            print("nothing left to run under those filters")
            return
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
                predict.map(
                    [asdict(job) for job in selected],
                    kwargs={"allow_msa_search": allow_msa_search},
                    return_exceptions=True,
                )
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
                    predict.map(
                        [asdict(job) for job in batch],
                        kwargs={"allow_msa_search": allow_msa_search},
                        return_exceptions=True,
                    )
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

    #: What a Modal volume raises when a path is not there.
    #:
    #: Not ``FileNotFoundError``. Modal raises its own ``NotFoundError`` from
    #: the gRPC layer, and catching only the built-in means "this path is
    #: absent", the ordinary case on a fresh or partly populated volume, comes
    #: out as a crash. That is what happened here: leftover directories from a
    #: failed run had no ``metrics.json`` in them, and probing for it took the
    #: launcher down before a single job was dispatched.
    #:
    #: Imported from ``modal.exception`` directly. ``modal/__init__.py``
    #: re-exports only the ``Error`` base class, so ``modal.exception.NotFoundError``
    #: is not reachable as an attribute of ``modal`` without this import.
    from modal.exception import NotFoundError as _ModalNotFound

    VOLUME_MISSING = (FileNotFoundError, StopIteration, GeneratorExit, _ModalNotFound)

    def weights_present() -> bool:
        """Are the AlphaFold parameters actually in the weights volume?

        Checked from the launcher, before any container starts, because the
        alternative is discovering it once every GPU in the wave has been
        allocated and every job has failed identically.
        """
        try:
            names = {Path(entry.path).name for entry in weights_volume.listdir("/params")}
        except VOLUME_MISSING:
            return False
        return any(name.endswith(".npz") for name in names)

    def completed_keys() -> set[str]:
        """Job keys already present on the results volume.

        A directory alone does not count. A job that started and died leaves its
        output directory behind, and treating that as complete would skip the
        work permanently and quietly shrink the grid. Only a written
        ``metrics.json`` marks a job as done, which is also why it is the last
        thing ``predict`` writes.
        """
        keys: set[str] = set()
        try:
            entries = list(results_volume.listdir("/", recursive=False))
        except VOLUME_MISSING:
            return set()

        for entry in entries:
            name = Path(entry.path).name
            if name in {"natives", "msa_cache"}:
                continue
            try:
                next(iter(results_volume.listdir(f"/{name}/metrics.json")))
            except VOLUME_MISSING:
                continue
            keys.add(name)
        return keys

    def cached_alignments() -> set[str]:
        """Alignment file names already sitting in the cache on the volume."""
        try:
            entries = results_volume.listdir("/msa_cache")
        except VOLUME_MISSING:
            return set()
        return {Path(entry.path).name for entry in entries}

    def completed_metrics() -> list[dict]:
        """Read back the metrics of every completed job, for measurement only.

        Kept separate from ``completed_keys`` because it downloads a file per
        job rather than listing directories, which is fine for reporting and
        wasteful in the launcher's hot path.
        """
        records: list[dict] = []
        for key in sorted(completed_keys()):
            try:
                blob = b"".join(results_volume.read_file(f"{key}/metrics.json"))
            except VOLUME_MISSING:
                continue
            records.append(json.loads(blob))
        return records

else:  # pragma: no cover

    def completed_keys() -> set[str]:
        """Without Modal installed there is nothing to interrogate."""
        return set()

    def completed_metrics() -> list[dict]:
        """Likewise."""
        return []


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

    # Split per chain, because the complex mean cannot answer the question the
    # analysis needs it for.
    #
    # The confound for the whole AF2 arm is that a highly charged design may
    # simply not fold, and an interface score falls when the monomer collapses
    # for reasons that have nothing to do with the interface. Separating the two
    # needs the *designed* chain's confidence on its own. The complex mean is
    # roughly half native partner, held identical at every beta, which damps
    # exactly the signal being looked for: a design that has fallen apart can
    # still show a respectable complex mean.
    #
    # plddt comes back as one value per residue in chain_order concatenation
    # order, the same order the PAE matrix uses.
    plddt = np.array(scores.get("plddt", []), dtype=float)
    per_chain_plddt: dict[str, float] = {}
    if plddt.size == sum(lengths):
        start = 0
        for chain_id, length in zip(chain_order, lengths, strict=True):
            per_chain_plddt[chain_id] = float(plddt[start : start + length].mean())
            start += length

    metrics: dict = {
        "complex_ptm": float(scores["ptm"]) if "ptm" in scores else None,
        "interface_ptm": float(scores["iptm"]) if "iptm" in scores else None,
        "interface_pae": interface_pae,
        "mean_plddt": float(np.mean(scores["plddt"])) if scores.get("plddt") else None,
        "designed_chain_plddt": per_chain_plddt.get(job.designed_chain),
        "partner_chain_plddt": next(
            (v for c, v in per_chain_plddt.items() if c != job.designed_chain), None
        ),
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
        "--split-plan",
        type=int,
        nargs="?",
        const=700,
        default=None,
        metavar="MAX_RESIDUES",
        help=(
            "Cost splitting the grid between a free GPU and paid A100 time, with "
            "complexes at or below MAX_RESIDUES going to the free one. Default 700, "
            "which is about what a 16 GB card holds."
        ),
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
            "  modal run modal_app/af2_multimer.py::run\n"
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

    if args.split_plan is not None and target and not isinstance(target, int):
        plan = split_plan(target, args.split_plan)
        colab, modal_side = plan["colab"], plan["modal"]
        print(f"\n=== split at {plan['colab_max_residues']} residues ===")
        print(
            f"  free GPU:  {colab['n_jobs']:>4} job(s) over {colab['n_complexes']:>2} complex(es), "
            f"{colab['cost_share']:.0%} of the compute, no cost"
        )
        print(
            f"  paid A100: {modal_side['n_jobs']:>4} job(s) over "
            f"{modal_side['n_complexes']:>2} complex(es), "
            f"{modal_side['cost_share']:.0%} of the compute, "
            f"about ${modal_side['estimated_usd']:.2f}"
        )
        if modal_side["complexes"]:
            print(f"  paid complexes: {', '.join(modal_side['complexes'])}")
        print(
            "\n  Run the free side first with notebooks/af2_colab.ipynb, which caps\n"
            "  itself at what the card holds and records anything larger as skipped.\n"
            f"  Then pick up the remainder here with --min-residues {plan['colab_max_residues'] + 1}.\n"
            "  Both write the same metrics and tag their source, so the two halves pool."
        )

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
