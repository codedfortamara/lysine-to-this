"""Net charge, computed two ways, kept rigorously apart.

The source paper uses two different quantities and calls both of them "net
charge":

``net_charge_simple``
    ``(#K + #R) - (#D + #E)``. An integer. This is what the inference-time
    controller actually targets, because the bias is added to the K and R
    logits and subtracted from the D and E logits. When the paper says the
    controller hits a target to within 6.9 charge units, this is the scale.

``net_charge_ph``
    A Henderson-Hasselbalch sum at a stated pH, including histidine, the free
    alpha-amino and alpha-carboxyl termini, and by convention cysteine and
    tyrosine. A float. This is what the paper's developability tables report,
    and it is the physically meaningful one, because it is the charge the
    molecule actually carries in buffer.

The two differ systematically. At pH 7.4 a lysine is not fully protonated, an
arginine effectively is, a histidine is roughly a tenth protonated, and the
termini contribute close to plus one and minus one. On a hundred-residue chain
the gap between the definitions is commonly two to four charge units and it is
not a constant offset, so the two must never be averaged, compared or written
into the same column. Every function here is explicit about which definition it
implements, and :class:`ChargePartition` carries both side by side under
separate names.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import (
    ANIONIC_RESIDUES,
    CATIONIC_RESIDUES,
    PKA_SETS,
    SIMPLE_ANIONIC,
    SIMPLE_CATIONIC,
    STANDARD_AA,
    ChargeParams,
)

__all__ = [
    "ChargePartition",
    "SequenceError",
    "net_charge_ph",
    "net_charge_simple",
    "partition_charge",
    "per_residue_charge_ph",
    "per_residue_charge_simple",
    "validate_sequence",
]

_STANDARD_SET = frozenset(STANDARD_AA)


class SequenceError(ValueError):
    """Raised when a sequence contains anything other than the standard twenty."""


def validate_sequence(seq: str) -> str:
    """Return ``seq`` unchanged, or raise if it is not a clean protein sequence.

    Non-standard codes are rejected rather than skipped. An ``X`` silently
    treated as neutral would shift a reported net charge without leaving any
    trace, which is exactly the class of error this project exists to rule out.
    """
    if not isinstance(seq, str):
        raise SequenceError(f"sequence must be a string, got {type(seq).__name__}")
    if not seq:
        raise SequenceError("sequence is empty")
    bad = sorted({ch for ch in seq if ch not in _STANDARD_SET})
    if bad:
        positions = {ch: [i for i, c in enumerate(seq) if c == ch][:5] for ch in bad}
        raise SequenceError(
            f"sequence contains non-standard residue code(s) {bad} "
            f"at (up to five) zero-based positions {positions}. "
            "Expected only the twenty standard amino acids "
            f"({STANDARD_AA}). Fix the input rather than masking it."
        )
    return seq


# ---------------------------------------------------------------------------
# Definition one: the integer count the controller targets
# ---------------------------------------------------------------------------


def per_residue_charge_simple(seq: str) -> np.ndarray:
    """Per-residue integer charge under the simple definition.

    Returns an array of length ``len(seq)`` holding ``+1`` at K and R, ``-1``
    at D and E, and ``0`` everywhere else.
    """
    validate_sequence(seq)
    q = np.zeros(len(seq), dtype=np.int64)
    for i, aa in enumerate(seq):
        if aa in SIMPLE_CATIONIC:
            q[i] = 1
        elif aa in SIMPLE_ANIONIC:
            q[i] = -1
    return q


def net_charge_simple(seq: str) -> int:
    """``(#K + #R) - (#D + #E)``.

    This is the quantity the inference-time controller moves directly, so it is
    the right x-axis for anything describing what the dial does. It ignores
    histidine, the termini and pH entirely.
    """
    return int(per_residue_charge_simple(seq).sum())


# ---------------------------------------------------------------------------
# Definition two: Henderson-Hasselbalch at a stated pH
# ---------------------------------------------------------------------------


def _residue_charge_ph(aa: str, ph: float, pka: dict[str, float], include_cys_tyr: bool) -> float:
    """Charge contribution of a single side chain at ``ph``."""
    if aa in CATIONIC_RESIDUES:
        return 1.0 / (1.0 + 10.0 ** (ph - pka[aa]))
    if aa in ANIONIC_RESIDUES:
        if aa in ("C", "Y") and not include_cys_tyr:
            return 0.0
        return -1.0 / (1.0 + 10.0 ** (pka[aa] - ph))
    return 0.0


def per_residue_charge_ph(
    seq: str,
    ph: float = 7.4,
    pka_set: str = "emboss",
    include_cys_tyr: bool = True,
    include_termini: bool = True,
) -> np.ndarray:
    """Per-residue Henderson-Hasselbalch charge at ``ph``.

    The two terminal contributions are folded into the first and last elements
    of the returned array rather than being held separately. That is what makes
    :func:`partition_charge` exactly additive: each terminus is attributed to
    the partition of the residue that carries it, so the partition sums equal
    the whole-chain total with no remainder to explain away.
    """
    validate_sequence(seq)
    if pka_set not in PKA_SETS:
        raise ValueError(f"unknown pka_set {pka_set!r}, expected one of {sorted(PKA_SETS)}")
    pka = PKA_SETS[pka_set]

    q = np.array(
        [_residue_charge_ph(aa, ph, pka, include_cys_tyr) for aa in seq],
        dtype=np.float64,
    )

    if include_termini:
        q[0] += 1.0 / (1.0 + 10.0 ** (ph - pka["Nterm"]))
        q[-1] += -1.0 / (1.0 + 10.0 ** (pka["Cterm"] - ph))

    return q


def net_charge_ph(
    seq: str,
    ph: float = 7.4,
    pka_set: str = "emboss",
    include_cys_tyr: bool = True,
    include_termini: bool = True,
) -> float:
    """Net charge at ``ph`` by Henderson-Hasselbalch.

    Includes histidine and, unless switched off, the free chain termini and the
    weakly acidic cysteine and tyrosine side chains. This is the definition
    behind the paper's developability tables.

    The default pKa set is EMBOSS. Biopython's ``ProtParam`` uses the
    Bjellqvist set instead, and the two disagree by roughly half a charge unit
    to one and a half on a typical chain. If the collaborator's
    ``net_charge_reported`` column was produced with Biopython, pass
    ``pka_set="bjellqvist"``. Script 02 reports which combination reproduces
    that column rather than guessing.
    """
    return float(
        per_residue_charge_ph(
            seq,
            ph=ph,
            pka_set=pka_set,
            include_cys_tyr=include_cys_tyr,
            include_termini=include_termini,
        ).sum()
    )


def net_charge_from_params(seq: str, params: ChargeParams) -> float:
    """Convenience wrapper applying a :class:`ChargeParams` block."""
    return net_charge_ph(
        seq,
        ph=params.ph,
        pka_set=params.pka_set,
        include_cys_tyr=params.include_cys_tyr,
        include_termini=params.include_termini,
    )


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChargePartition:
    """Charge of a chain split into interface, non-interface surface and core.

    Both definitions are carried, under separate names, for every partition.
    There is deliberately no field that merges them and no default definition:
    a caller has to say which one it wants, every time.

    The three partitions are mutually exclusive and jointly exhaustive over the
    residues of the chain, so ``interface_* + surface_* + core_*`` equals
    ``total_*`` for each definition. That identity is asserted in the tests.
    """

    n_interface: int
    n_surface: int
    n_core: int
    n_total: int

    interface_simple: int
    surface_simple: int
    core_simple: int
    total_simple: int

    interface_ph: float
    surface_ph: float
    core_ph: float
    total_ph: float

    ph: float
    pka_set: str

    def to_row(self) -> dict[str, Any]:
        """Flat mapping for a results table.

        Every charge column carries an explicit ``_simple`` or ``_ph`` suffix,
        and the pH-based columns additionally carry the pH and pKa set, so that
        two runs at different settings cannot be concatenated into one column
        without the difference being visible.
        """
        ph_tag = f"ph{self.ph:g}_{self.pka_set}"
        return {
            "n_residues_interface": self.n_interface,
            "n_residues_surface": self.n_surface,
            "n_residues_core": self.n_core,
            "n_residues_total": self.n_total,
            "charge_interface_simple": self.interface_simple,
            "charge_surface_simple": self.surface_simple,
            "charge_core_simple": self.core_simple,
            "charge_total_simple": self.total_simple,
            f"charge_interface_{ph_tag}": self.interface_ph,
            f"charge_surface_{ph_tag}": self.surface_ph,
            f"charge_core_{ph_tag}": self.core_ph,
            f"charge_total_{ph_tag}": self.total_ph,
        }

    def charge_density_simple(self) -> dict[str, float]:
        """Charge per residue in each partition, simple definition.

        Partitions differ greatly in size, so the raw sums answer "where is the
        charge" while these densities answer "how charged is this region",
        which is the form the buffering hypothesis is stated in.
        """
        return {
            "interface": self.interface_simple / self.n_interface if self.n_interface else 0.0,
            "surface": self.surface_simple / self.n_surface if self.n_surface else 0.0,
            "core": self.core_simple / self.n_core if self.n_core else 0.0,
        }


def _check_indices(name: str, idx: Iterable[int], n: int) -> frozenset[int]:
    out = frozenset(int(i) for i in idx)
    bad = sorted(i for i in out if i < 0 or i >= n)
    if bad:
        raise IndexError(
            f"{name} contains zero-based index(es) {bad[:10]} outside the sequence "
            f"of length {n}. This usually means a residue numbering offset was "
            "not applied when mapping structure residues onto the sequence."
        )
    return out


def partition_charge(
    seq: str,
    interface_idx: Iterable[int],
    core_idx: Iterable[int],
    ph: float = 7.4,
    pka_set: str = "emboss",
    include_cys_tyr: bool = True,
    include_termini: bool = True,
) -> ChargePartition:
    """Split the charge of ``seq`` into interface, surface and core.

    Parameters
    ----------
    seq
        The chain sequence, one letter per residue.
    interface_idx
        Zero-based positions into ``seq`` of the interface residues. These come
        from :mod:`interface_charge.interface` and are defined on the *native*
        structure, so that the partition boundaries do not move as beta changes
        and the comparison across beta stays like for like.
    core_idx
        Zero-based positions of the buried core residues, likewise from the
        native structure.

    Notes
    -----
    Non-interface surface is defined by subtraction: every residue that is
    neither interface nor core. The two input sets must be disjoint, which is
    guaranteed by :func:`interface_charge.interface.classify_residues` and
    checked again here because this function is also callable directly.
    """
    validate_sequence(seq)
    n = len(seq)
    interface = _check_indices("interface_idx", interface_idx, n)
    core = _check_indices("core_idx", core_idx, n)

    overlap = sorted(interface & core)
    if overlap:
        raise ValueError(
            f"interface and core index sets overlap at positions {overlap[:10]}. "
            "A residue cannot be both buried in the monomer and buried by "
            "complexation. Check the classification thresholds."
        )

    surface = frozenset(range(n)) - interface - core

    q_simple = per_residue_charge_simple(seq)
    q_ph = per_residue_charge_ph(
        seq,
        ph=ph,
        pka_set=pka_set,
        include_cys_tyr=include_cys_tyr,
        include_termini=include_termini,
    )

    def take(mask: frozenset[int], arr: np.ndarray) -> np.ndarray:
        if not mask:
            return arr[:0]
        return arr[np.fromiter(sorted(mask), dtype=np.int64, count=len(mask))]

    return ChargePartition(
        n_interface=len(interface),
        n_surface=len(surface),
        n_core=len(core),
        n_total=n,
        interface_simple=int(take(interface, q_simple).sum()),
        surface_simple=int(take(surface, q_simple).sum()),
        core_simple=int(take(core, q_simple).sum()),
        total_simple=int(q_simple.sum()),
        interface_ph=float(take(interface, q_ph).sum()),
        surface_ph=float(take(surface, q_ph).sum()),
        core_ph=float(take(core, q_ph).sum()),
        total_ph=float(q_ph.sum()),
        ph=ph,
        pka_set=pka_set,
    )
