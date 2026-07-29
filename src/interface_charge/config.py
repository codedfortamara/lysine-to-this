"""Single source of truth for every tunable parameter in this project.

Nothing in ``scripts/`` or ``modal_app/`` may hard-code a numeric threshold.
If a number influences a result, it lives here, it is written into the run
manifest by :mod:`interface_charge.provenance`, and it can therefore be traced
from any figure back to the run that produced it.

Values may be overridden at run time from a TOML file via
:func:`load_overrides`, so that a sensitivity analysis does not require a
source edit.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
NATIVE_DIR: Final[Path] = DATA_DIR / "native"
INTERIM_DIR: Final[Path] = DATA_DIR / "interim"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Amino acid reference data
# ---------------------------------------------------------------------------

#: The twenty standard amino acids. Any other single-letter code in an input
#: sequence is an error rather than something to be silently skipped, because
#: an unexpected ``X`` would quietly change a net charge.
STANDARD_AA: Final[str] = "ACDEFGHIKLMNPQRSTVWY"

#: Maximum solvent accessible surface area per residue, in square angstroms,
#: from the theoretical Gly-X-Gly tripeptide values of Tien et al. (2013),
#: PLoS ONE 8:e80635. Used to convert an absolute SASA into a relative one so
#: that the buried core can be defined on a per-residue basis.
MAX_ASA_TIEN_2013: Final[dict[str, float]] = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "E": 223.0,
    "Q": 225.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
}

# ---------------------------------------------------------------------------
# pKa sets for the Henderson-Hasselbalch charge definition
# ---------------------------------------------------------------------------
#
# Which set the collaborator used is an OPEN QUESTION (see data/README.md).
# The two sets disagree by roughly 0.5 to 1.5 charge units on a typical
# hundred-residue chain at pH 7.4, which is small next to the charge shifts the
# controller produces but large enough to matter when reconciling our numbers
# against the ``net_charge_reported`` column. Script 02 reports which
# combination of definition and pKa set best reproduces that column rather than
# assuming one.

#: EMBOSS ``iep`` pKa values, the most commonly cited set.
PKA_EMBOSS: Final[dict[str, float]] = {
    "Nterm": 8.6,
    "Cterm": 3.6,
    "K": 10.8,
    "R": 12.5,
    "H": 6.5,
    "D": 3.9,
    "E": 4.1,
    "C": 8.5,
    "Y": 10.1,
}

#: Bjellqvist values, as used by Biopython's ``ProtParam.IsoelectricPoint``.
#: If the collaborator scored charge with Biopython, this is the set to use.
PKA_BJELLQVIST: Final[dict[str, float]] = {
    "Nterm": 7.5,
    "Cterm": 3.55,
    "K": 10.0,
    "R": 12.0,
    "H": 5.98,
    "D": 4.05,
    "E": 4.45,
    "C": 9.0,
    "Y": 10.0,
}

#: Published Modal on-demand rates in US dollars per GPU-hour, for the dry-run
#: estimate only. UNVERIFIED: recorded from memory because modal.com was not
#: reachable from the environment this was written in. Confirm before quoting.
MODAL_GPU_RATES_USD_PER_HOUR: Final[dict[str, float]] = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
}

PKA_SETS: Final[dict[str, dict[str, float]]] = {
    "emboss": PKA_EMBOSS,
    "bjellqvist": PKA_BJELLQVIST,
}

#: Residues carrying a positive charge in the Henderson-Hasselbalch model.
CATIONIC_RESIDUES: Final[tuple[str, ...]] = ("K", "R", "H")

#: Residues carrying a negative charge in the Henderson-Hasselbalch model.
ANIONIC_RESIDUES: Final[tuple[str, ...]] = ("D", "E", "C", "Y")

#: Residues counted by the simple integer definition that the controller targets.
SIMPLE_CATIONIC: Final[tuple[str, ...]] = ("K", "R")
SIMPLE_ANIONIC: Final[tuple[str, ...]] = ("D", "E")


# ---------------------------------------------------------------------------
# Parameter blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InterfaceParams:
    """Parameters controlling both interface definitions and the core cutoff."""

    #: A residue is at the interface if it loses more than this many square
    #: angstroms of solvent accessible surface area on complexation.
    delta_sasa_threshold_a2: float = 1.0

    #: Any heavy atom of a residue within this distance, in angstroms, of a
    #: heavy atom of the partner chain puts that residue at the interface under
    #: the secondary contact definition.
    contact_cutoff_a: float = 5.0

    #: Relative SASA of a residue in the isolated chain below which it is
    #: called buried core. Relative SASA is the absolute value divided by the
    #: Tien et al. maximum for that residue type.
    core_relative_sasa_max: float = 0.25

    #: Solvent probe radius in angstroms.
    probe_radius_a: float = 1.4

    #: Number of test points per atom for the Shrake-Rupley sphere. Higher is
    #: more accurate and slower. 100 is the Biopython default.
    n_sphere_points: int = 100

    #: Backend for the SASA calculation. ``shrake_rupley`` uses Biopython and
    #: is the default: it is deterministic, needs no compiled dependency, and
    #: takes about a second on a six thousand atom complex. ``freesasa`` runs
    #: the Lee-Richards algorithm instead, over the identical atom selection
    #: and identical radii, so switching backends isolates the effect of the
    #: algorithm alone. Reporting both is a cheap robustness check.
    sasa_backend: str = "shrake_rupley"


@dataclass(frozen=True, slots=True)
class ChargeParams:
    """Parameters for the two charge definitions."""

    #: pH at which the Henderson-Hasselbalch charge is evaluated. The paper
    #: reports charge at pH 7.4.
    ph: float = 7.4

    #: Key into :data:`PKA_SETS`.
    pka_set: str = "emboss"

    #: Whether cysteine and tyrosine contribute to the Henderson-Hasselbalch
    #: charge. Standard implementations include them. At pH 7.4 they contribute
    #: a small negative amount. Turning this off makes the pH-based definition
    #: closer in spirit to the simple one, which is occasionally what a
    #: reviewer will ask for.
    include_cys_tyr: bool = True

    #: Whether the free alpha-amino and alpha-carboxyl termini contribute.
    include_termini: bool = True

    #: Absolute tolerance, in charge units, when checking our computed charge
    #: against the ``net_charge_reported`` column of designs.csv.
    reported_charge_tolerance: float = 0.51


@dataclass(frozen=True, slots=True)
class ComplementarityParams:
    """Parameters for the electrostatic complementarity statistic.

    This is deliberately a contact-level charge-product statistic and not a
    Poisson-Boltzmann calculation. See :mod:`interface_charge.complementarity`
    for the full argument and the limitations.
    """

    #: Distance cutoff, in angstroms, between heavy atoms of the two chains for
    #: a residue pair to count as contacting.
    contact_cutoff_a: float = 5.0

    #: Charge definition used for the per-residue charges in the product.
    #: One of ``simple`` or ``ph``.
    charge_definition: str = "ph"

    #: Distance weighting of each pair term. ``none`` gives every contacting
    #: pair equal weight. ``inverse_distance`` weights by the reciprocal of the
    #: minimum heavy-atom distance, which is a crude nod to Coulomb's law
    #: without pretending to be a solved electrostatic potential.
    weight: str = "none"

    #: Whether to skip pairs where either residue is formally neutral, which
    #: only affects the reported pair count, never the sum.
    count_neutral_pairs: bool = False


@dataclass(frozen=True, slots=True)
class SaltBridgeParams:
    """Parameters for salt-bridge counting.

    Two definitions are provided because they disagree substantially and the
    size of that disagreement is itself worth reporting.
    """

    #: Maximum distance, in angstroms, between a cationic nitrogen and an
    #: anionic oxygen for a salt bridge under the geometric definition. Four
    #: angstroms between charged-group heavy atoms is the conventional
    #: criterion (Barlow and Thornton, 1983).
    cutoff_a: float = 4.0

    #: Maximum distance, in angstroms, between the Cbeta atoms of oppositely
    #: charged residues under the coarse proxy definition. Cbeta sits at the
    #: base of the side chain, so this criterion is indifferent to which way
    #: the charged group actually points and will count pairs whose side
    #: chains face apart. Retained so the two definitions can be compared on
    #: the same structures rather than argued about.
    proxy_cbeta_cutoff_a: float = 6.0

    #: Whether histidine counts as cationic. At pH 7.4 it is roughly a tenth
    #: protonated, so counting it as a full positive charge overstates its
    #: contribution. Off by default.
    include_histidine: bool = False


@dataclass(frozen=True, slots=True)
class StructureParams:
    """Parameters for fetching and parsing native structures."""

    #: Template for the RCSB download URL. ``{pdb_id}`` is substituted with the
    #: upper-case four-character identifier.
    rcsb_url_template: str = "https://files.rcsb.org/download/{pdb_id}.cif"

    #: File suffix matching the template above.
    native_format: str = "cif"

    #: Number of download attempts before giving up on a single identifier.
    download_attempts: int = 4

    #: Initial backoff in seconds. Doubles on each retry.
    download_backoff_s: float = 2.0

    #: Socket timeout for a single download attempt, in seconds.
    download_timeout_s: float = 60.0

    #: Which model to take from a multi-model file such as an NMR ensemble.
    model_index: int = 0

    #: Waters and other heteroatoms are excluded from every calculation. Ions
    #: at an interface can matter physically, but including them would make the
    #: charge partition depend on crystallisation conditions.
    keep_heteroatoms: bool = False

    #: Hydrogens are stripped before any SASA or contact calculation so that
    #: results do not depend on whether a structure was deposited with them.
    #: This lives here rather than under the interface block because it is a
    #: property of how a structure is read, and both the interface and the
    #: complementarity calculations have to make the identical choice.
    ignore_hydrogens: bool = True

    #: Alternate location indicators that are kept. Everything else is dropped.
    #: The count of dropped atoms is reported so the choice is never invisible.
    accepted_altlocs: tuple[str, ...] = ("", " ", "A")


@dataclass(frozen=True, slots=True)
class AF2Params:
    """Parameters for the AlphaFold2-Multimer refolding run on Modal.

    Consumed by ``modal_app/af2_multimer.py``. Kept here so that the dry-run
    cost estimate and the real run cannot drift apart.
    """

    #: GPU type requested from Modal.
    #:
    #: L4, not A100. Three reasons, all of them money.
    #:
    #: It costs 0.80 an hour against 2.10. It is slower per job, perhaps two to
    #: three times, so the cost per job comes out lower rather than merely
    #: proportional. And it is far less contended: the A100 pool is the most
    #: fought over Modal has, which on 29 July put containers into a preemption
    #: loop that ran 141 minutes, completed nothing, and was billed in full.
    #:
    #: 24 GB is ample for what remains. The complexes under 450 residues have a
    #: median total length of 270, and AlphaFold-Multimer memory grows with the
    #: square of length. The nine complexes that need more than this are exactly
    #: the ones already excluded on cost.
    gpu_type: str = "L4"

    #: Per-call timeout in seconds, sized for the work that remains.
    #:
    #: Was 7200, on the reasoning that a killed job costs more than an idle
    #: slot. That reasoning was wrong, and expensively so: it meant every job
    #: that went wrong ran for two hours before anything noticed. 2ZXE did that
    #: three separate times and produced nothing.
    #:
    #: A 270-residue complex on an L4 should fold in well under fifteen minutes.
    #: Anything still running at twenty is not slow, it is stuck, and the cheap
    #: response is to kill it and move on. This caps the cost of any single
    #: failure at one eighth of what it was.
    timeout_s: int = 1200

    #: Number of AlphaFold model parameter sets to run per job.
    num_models: int = 1

    #: Which multimer parameter release to use.
    model_preset: str = "multimer"

    #: Number of recycles.
    num_recycles: int = 3

    #: Random seed for the prediction. Recorded in the manifest.
    random_seed: int = 0

    #: Maximum number of containers Modal may run at once.
    #:
    #: Low on purpose. This was 40, and that was costing money in two ways.
    #:
    #: Every container loads several gigabytes of AlphaFold parameters before it
    #: folds anything, and that load is billed. Forty containers running 99 jobs
    #: means roughly one job each, so the model is loaded 40 times rather than
    #: 6, and the whole of that overhead is paid rather than amortised.
    #:
    #: Wide fan-out also multiplies exposure to preemption. Every container is
    #: independently reclaimable, and a container reclaimed mid-job loses its
    #: work and restarts from the beginning, having been billed for both.
    #:
    #: Six containers over 99 jobs is about sixteen jobs each. Slower in wall
    #: clock, considerably cheaper, and far less likely to spend an afternoon
    #: restarting.
    max_containers: int = 6

    #: What Modal actually bills, divided by what the completed jobs account for.
    #:
    #: The budget guard sums each job's own wall clock. Modal bills *container*
    #: time, which is a larger number: a container is paid for while it pulls
    #: the image, while it loads the model parameters, and while it sits idle
    #: waiting for the slowest job in a chunk to finish so the next chunk can
    #: start. None of that appears in any job's wall clock.
    #:
    #: Measured, not assumed. The first real batch reported $12.41 of job time
    #: against roughly $26.00 billed, giving 2.1. Without this factor the guard
    #: under-reports by half, which is the worst possible direction for a
    #: safeguard to be wrong in: it was set to stop at $54 and would have run
    #: past $100.
    #:
    #: It is a floor rather than a typical case. The measurement came from a
    #: chunk containing the five largest jobs in the grid, where idle time is
    #: worst, so later chunks should pack better and come in below it.
    billing_overhead: float = 2.1

    #: Extra GPU minutes per job attributable to running with an MSA rather than
    #: single-sequence. Two effects, both billed at the GPU rate because both
    #: happen inside the GPU container: the one-off MMseqs2 search for each
    #: complex's native partner chain, amortised over the five beta settings that
    #: share it, and the larger MSA representation AlphaFold then has to embed
    #: and recycle on every job.
    #:
    #: Now zero, because the search no longer happens on the GPU.
    #:
    #: It used to: ``predict`` called run_mmseqs2 inline, and the wait against a
    #: free shared server was billed at the A100 rate. Three jobs in the first
    #: pilot hit the two-hour timeout sitting in that queue and produced
    #: nothing. The searches moved to a CPU-only ``prefetch`` entrypoint, where
    #: all 55 cost $0.86 including one that waited 15.5 minutes.
    #:
    #: What remains is the cost of embedding and recycling a larger MSA
    #: representation, which is real but is part of the folding time now
    #: measured directly rather than a separate addend.
    msa_overhead_minutes: float = 0.0

    #: PAE below which a cross-chain residue pair counts towards ipSAE, in
    #: angstroms. Ten is the value used throughout Dunbrack (2025). The score is
    #: not scale-free in this parameter, so it is pinned here rather than passed
    #: at the call site, and it is recorded in every manifest.
    ipsae_pae_cutoff_a: float = 10.0

    #: Price per GPU-hour in US dollars. Left at 0.0 to derive it from
    #: :attr:`gpu_type`, which is almost always what is wanted.
    #:
    #: It used to be a hardcoded 2.10, independent of which card was requested.
    #: Switching to an L4 therefore left the budget guard pricing every second
    #: at A100 rates, nearly three times the truth. That errs towards stopping
    #: rather than overspending, so it was not dangerous, but a guard that does
    #: not know what the hardware costs cannot be reasoned about.
    #:
    #: Rates are UNVERIFIED, recorded from memory. Check modal.com/pricing
    #: before quoting a number to anyone.
    usd_per_gpu_hour: float = 0.0

    def rate_usd_per_hour(self) -> float:
        """The configured rate, or the published one for the configured GPU."""
        if self.usd_per_gpu_hour > 0.0:
            return self.usd_per_gpu_hour
        rate = MODAL_GPU_RATES_USD_PER_HOUR.get(self.gpu_type)
        if rate is None:
            raise ValueError(
                f"no published rate recorded for GPU {self.gpu_type!r}. Add one to "
                f"MODAL_GPU_RATES_USD_PER_HOUR or set usd_per_gpu_hour explicitly; "
                "a budget guard with no price is not a guard."
            )
        return rate

    #: Cold start per container: image pull plus loading the model parameters
    #: from the weights volume. Paid once per container, not once per job, so
    #: it matters at small job counts and washes out at large ones.
    cold_start_minutes: float = 3.0

    #: Fraction added to cover failed jobs and their one retry. Refolding
    #: failures are not random; large and heavily charged complexes fail more
    #: often, so this is a floor rather than a typical case.
    failure_overhead: float = 0.10

    #: Consecutive failures, with nothing succeeding in between, after which the
    #: run stops itself.
    #:
    #: On 29 July a driver failed 20 jobs, completed 0, and kept dispatching for
    #: 141 minutes with ten GPUs attached. Nothing in the code was watching. A
    #: systematic failure repeats by definition, so the only question is how many
    #: times it is paid for before someone notices, and the answer should not
    #: depend on someone noticing.
    #:
    #: Four is deliberately tight. Scattered failures are normal and reset the
    #: counter; four in a row with no success between them is a broken run.
    max_consecutive_failures: int = 4

    #: Minimum jobs attempted before the overall failure rate is allowed to stop
    #: a run, so that a single unlucky first job does not.
    failure_rate_min_attempts: int = 8

    #: Overall failure rate above which the run stops, once enough jobs have been
    #: attempted to measure it. Catches the case the consecutive counter misses:
    #: alternating success and failure, which is still half the money wasted.
    max_failure_rate: float = 0.5

    #: Wall-clock estimate per job in minutes.
    #:
    #: No longer a planning figure. 32.2 minutes is the mean over the first
    #: eleven completed jobs, reported by the budget guard on 28 July 2026.
    #:
    #: The previous value here was 8.0, and it was wrong by a factor of four.
    #: That error went straight into a quoted cost of $55 for a grid that is
    #: closer to $194, and the quote was acted on. The measurement replaces it.
    #:
    #: It is still an overestimate for the grid as a whole, because those eleven
    #: jobs were the largest complexes in the set: canary ordering runs them
    #: first on purpose, so the first measurement available is always the
    #: pessimistic one. Compute grows roughly with the square of total length
    #: and the median complex is 395 residues against 1257 for the largest.
    estimated_minutes_per_job: float = 32.2

    #: Total residues of the complex the 32.2 minutes above was measured on.
    #:
    #: Not a detail. AlphaFold attention is quadratic in sequence length, so a
    #: per-job time is meaningless without the size it was measured at. Those
    #: eleven jobs averaged 819,000 residues squared, a root-mean-square length
    #: of 905, against a grid median of 395. Applying 32.2 minutes flat to every
    #: job therefore overstates the grid by roughly a factor of four, which is
    #: exactly the mistake made in the other direction an hour earlier.
    #:
    #: When ``estimate_cost`` is handed real jobs it rescales by the square of
    #: the length ratio against this figure. Given only a job count it cannot,
    #: and falls back to the flat number, which is pessimistic and says so.
    timing_reference_residues: float = 905.0

    #: Name of the Modal Volume holding model weights.
    weights_volume: str = "af2-weights"

    #: Name of the Modal Volume receiving per-job outputs.
    results_volume: str = "interface-charge-af2-results"


@dataclass(frozen=True, slots=True)
class Config:
    """Top-level configuration object passed around explicitly."""

    interface: InterfaceParams = field(default_factory=InterfaceParams)
    charge: ChargeParams = field(default_factory=ChargeParams)
    salt_bridge: SaltBridgeParams = field(default_factory=SaltBridgeParams)
    complementarity: ComplementarityParams = field(default_factory=ComplementarityParams)
    structure: StructureParams = field(default_factory=StructureParams)
    af2: AF2Params = field(default_factory=AF2Params)

    #: Global random seed. Recorded in every manifest even where unused, so
    #: that a later stochastic step cannot be added without leaving a trace.
    random_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict suitable for JSON serialisation."""
        out: dict[str, Any] = {"random_seed": self.random_seed}
        for name in (
            "interface",
            "charge",
            "salt_bridge",
            "complementarity",
            "structure",
            "af2",
        ):
            block = getattr(self, name)
            out[name] = {f: getattr(block, f) for f in block.__slots__}
        return out


DEFAULT_CONFIG: Final[Config] = Config()

_BLOCKS: Final[tuple[str, ...]] = (
    "interface",
    "charge",
    "salt_bridge",
    "complementarity",
    "structure",
    "af2",
)


def load_overrides(path: Path, base: Config | None = None) -> Config:
    """Return ``base`` with values replaced by those in a TOML file.

    The TOML is expected to mirror the block structure, for example::

        random_seed = 7

        [interface]
        delta_sasa_threshold_a2 = 2.0

        [charge]
        pka_set = "bjellqvist"

    An unknown block or key raises :class:`KeyError` rather than being ignored,
    because a silently misspelled override is a silently wrong result.
    """
    base = base if base is not None else DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"config override file not found: {path}")

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    updated = base
    for key, value in raw.items():
        if key == "random_seed":
            updated = replace(updated, random_seed=int(value))
            continue
        if key not in _BLOCKS:
            raise KeyError(
                f"unknown configuration block {key!r} in {path}. Known blocks: {', '.join(_BLOCKS)}"
            )
        if not isinstance(value, dict):
            raise TypeError(f"configuration block {key!r} in {path} must be a table")

        block = getattr(updated, key)
        known = set(block.__slots__)
        unknown = set(value) - known
        if unknown:
            raise KeyError(
                f"unknown key(s) {sorted(unknown)} in block [{key}] of {path}. "
                f"Known keys: {sorted(known)}"
            )
        updated = replace(updated, **{key: replace(block, **value)})

    validate(updated)
    return updated


def validate(config: Config) -> None:
    """Raise if a configuration is internally inconsistent.

    Called on every override load. Cheap, and it turns a typo into an
    immediate error rather than a plausible-looking wrong number.
    """
    if config.charge.pka_set not in PKA_SETS:
        raise ValueError(
            f"charge.pka_set must be one of {sorted(PKA_SETS)}, got {config.charge.pka_set!r}"
        )
    if not 0.0 <= config.charge.ph <= 14.0:
        raise ValueError(f"charge.ph must lie in [0, 14], got {config.charge.ph}")
    if config.interface.sasa_backend not in {"freesasa", "shrake_rupley"}:
        raise ValueError(
            "interface.sasa_backend must be 'freesasa' or 'shrake_rupley', "
            f"got {config.interface.sasa_backend!r}"
        )
    if config.interface.delta_sasa_threshold_a2 <= 0:
        raise ValueError("interface.delta_sasa_threshold_a2 must be positive")
    if config.interface.contact_cutoff_a <= 0:
        raise ValueError("interface.contact_cutoff_a must be positive")
    if not 0.0 < config.interface.core_relative_sasa_max < 1.0:
        raise ValueError("interface.core_relative_sasa_max must lie in (0, 1)")
    if config.complementarity.charge_definition not in {"simple", "ph"}:
        raise ValueError(
            "complementarity.charge_definition must be 'simple' or 'ph', "
            f"got {config.complementarity.charge_definition!r}"
        )
    if config.complementarity.weight not in {"none", "inverse_distance"}:
        raise ValueError(
            "complementarity.weight must be 'none' or 'inverse_distance', "
            f"got {config.complementarity.weight!r}"
        )
    if config.structure.native_format not in {"cif", "pdb"}:
        raise ValueError("structure.native_format must be 'cif' or 'pdb'")
    if config.af2.timeout_s <= 0:
        raise ValueError("af2.timeout_s must be positive")
    if config.salt_bridge.cutoff_a <= 0:
        raise ValueError("salt_bridge.cutoff_a must be positive")
    if config.salt_bridge.proxy_cbeta_cutoff_a <= 0:
        raise ValueError("salt_bridge.proxy_cbeta_cutoff_a must be positive")
