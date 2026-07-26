"""Interface residue definitions, and the partition of a chain into three regions.

Two independent definitions are implemented because they disagree, and the
size of that disagreement is itself worth reporting:

Primary, delta-SASA
    A residue is at the interface if its solvent accessible surface area falls
    by more than a threshold, one square angstrom by default, when the partner
    chain is added. This is the definition that matches the physical question,
    which is which residues are actually desolvated by binding.

Secondary, contact
    A residue is at the interface if any of its heavy atoms lies within five
    angstroms of a heavy atom of the partner chain. Cheaper, more common in the
    literature, and systematically more generous: it includes residues that sit
    near the partner without losing accessible surface.

Both are computed on the **native** structure by default and returned as a
fixed set of residue identities. This matters more than it looks. If the
interface were re-derived from each design's predicted structure, the partition
boundary would move as beta changed, and any trend in interface charge would be
confounded with a trend in what counts as interface. Fixing the definition on
the native makes the comparison across beta like for like.

:func:`recompute_on_predicted` exists for the separate and clearly labelled
question of whether the interface itself survives the charge shift. Its output
carries ``source="predicted"`` and must never be substituted for the native
definition when partitioning charge.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

import numpy as np
from Bio.PDB import NeighborSearch
from Bio.PDB.Model import Model
from Bio.PDB.SASA import ATOMIC_RADII, ShrakeRupley

from .config import MAX_ASA_TIEN_2013, InterfaceParams, StructureParams
from .structures import (
    ChainExtract,
    ResidueId,
    StructureError,
    extract_chain,
    heavy_atoms,
    residue_id,
    submodel,
)

__all__ = [
    "InterfaceAgreement",
    "InterfaceDefinition",
    "Partition",
    "ResidueClassification",
    "classify_residues",
    "compare_definitions",
    "define_interface",
    "interface_by_contact",
    "interface_by_delta_sasa",
    "recompute_on_predicted",
    "residue_sasa",
]


class Partition(StrEnum):
    """The three mutually exclusive regions a residue can occupy."""

    INTERFACE = "interface"
    SURFACE = "surface"
    CORE = "core"


#: Atom radii used by both SASA backends, so that the two differ only in
#: algorithm and not in parameterisation. These are Biopython's van der Waals
#: radii, the standard Bondi-derived set.
_ATOM_RADII: Final[dict[str, float]] = dict(ATOMIC_RADII)
_DEFAULT_RADIUS: Final[float] = 1.80


# ---------------------------------------------------------------------------
# Solvent accessible surface area
# ---------------------------------------------------------------------------


def residue_sasa(
    model: Model,
    chain_ids: Iterable[str],
    interface_params: InterfaceParams,
    structure_params: StructureParams,
) -> dict[ResidueId, float]:
    """Per-residue solvent accessible surface area, in square angstroms.

    The calculation is run over exactly the chains in ``chain_ids``, so calling
    it once with both chains and once with each chain alone yields the two
    numbers whose difference defines the delta-SASA interface.
    """
    clone = submodel(model, chain_ids, structure_params)

    if interface_params.sasa_backend == "shrake_rupley":
        sr = ShrakeRupley(
            probe_radius=interface_params.probe_radius_a,
            n_points=interface_params.n_sphere_points,
        )
        sr.compute(clone, level="R")
        out: dict[ResidueId, float] = {}
        for chain in clone:
            for res in chain:
                out[residue_id(res, chain.id)] = float(res.sasa)
        return out

    if interface_params.sasa_backend == "freesasa":
        return _residue_sasa_freesasa(clone, interface_params, structure_params)

    raise ValueError(f"unknown sasa_backend {interface_params.sasa_backend!r}")


def _residue_sasa_freesasa(
    clone: Model,
    interface_params: InterfaceParams,
    structure_params: StructureParams,
) -> dict[ResidueId, float]:
    """FreeSASA (Lee-Richards) backend.

    Coordinates and radii are handed over explicitly rather than letting
    FreeSASA re-read the file, so that both backends see the identical atom
    selection and the identical radii and any difference between them is
    attributable to the algorithm alone.
    """
    try:
        import freesasa
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "sasa_backend='freesasa' requires the freesasa package. "
            "Install it, or set interface.sasa_backend='shrake_rupley'."
        ) from exc

    coords: list[float] = []
    radii: list[float] = []
    owners: list[ResidueId] = []

    for chain in clone:
        for res in chain:
            rid = residue_id(res, chain.id)
            for atom in heavy_atoms(res, structure_params):
                x, y, z = (float(v) for v in atom.get_coord())
                coords.extend((x, y, z))
                radii.append(_ATOM_RADII.get(atom.element, _DEFAULT_RADIUS))
                owners.append(rid)

    if not owners:
        return {}

    params = freesasa.Parameters(
        {"probe-radius": interface_params.probe_radius_a, "algorithm": freesasa.LeeRichards}
    )
    result = freesasa.calcCoord(coords, radii, params)

    out: dict[ResidueId, float] = {rid: 0.0 for rid in owners}
    for index, rid in enumerate(owners):
        out[rid] += float(result.atomArea(index))
    return out


# ---------------------------------------------------------------------------
# Interface definitions
# ---------------------------------------------------------------------------


def interface_by_delta_sasa(
    model: Model,
    chain_a: str,
    chain_b: str,
    interface_params: InterfaceParams,
    structure_params: StructureParams,
) -> tuple[frozenset[ResidueId], dict[ResidueId, float]]:
    """Primary definition. Returns the interface set and the raw delta-SASA map.

    The delta map is returned alongside the set so that a caller can report the
    buried surface area, which is the standard descriptor of interface size,
    without recomputing anything.
    """
    if chain_a == chain_b:
        raise ValueError(f"chain_a and chain_b must differ, both were {chain_a!r}")

    sasa_complex = residue_sasa(model, (chain_a, chain_b), interface_params, structure_params)
    sasa_a = residue_sasa(model, (chain_a,), interface_params, structure_params)
    sasa_b = residue_sasa(model, (chain_b,), interface_params, structure_params)

    delta: dict[ResidueId, float] = {}
    for rid, isolated in list(sasa_a.items()) + list(sasa_b.items()):
        bound = sasa_complex.get(rid)
        if bound is None:  # pragma: no cover - defensive
            raise StructureError(
                f"residue {rid} present in the isolated chain but absent from the "
                "complex calculation. The two calculations saw different atoms."
            )
        # Numerical noise can make this very slightly negative. Clamping to zero
        # here is safe because the threshold is well above the noise floor.
        delta[rid] = max(0.0, float(isolated) - float(bound))

    threshold = interface_params.delta_sasa_threshold_a2
    selected = frozenset(rid for rid, value in delta.items() if value > threshold)
    return selected, delta


def interface_by_contact(
    model: Model,
    chain_a: str,
    chain_b: str,
    interface_params: InterfaceParams,
    structure_params: StructureParams,
) -> frozenset[ResidueId]:
    """Secondary definition: any heavy atom within the cutoff of the partner chain."""
    if chain_a == chain_b:
        raise ValueError(f"chain_a and chain_b must differ, both were {chain_a!r}")

    clone = submodel(model, (chain_a, chain_b), structure_params)

    atoms_a = []
    atoms_b = []
    owner: dict[int, ResidueId] = {}
    for chain in clone:
        for res in chain:
            rid = residue_id(res, chain.id)
            for atom in heavy_atoms(res, structure_params):
                owner[id(atom)] = rid
                (atoms_a if chain.id == chain_a else atoms_b).append(atom)

    if not atoms_a or not atoms_b:
        raise StructureError(
            f"one of the chains has no heavy atoms after filtering "
            f"({chain_a}: {len(atoms_a)}, {chain_b}: {len(atoms_b)})"
        )

    search = NeighborSearch(atoms_b)
    cutoff = interface_params.contact_cutoff_a
    selected: set[ResidueId] = set()

    for atom in atoms_a:
        neighbours = search.search(atom.get_coord(), cutoff, level="A")
        if neighbours:
            selected.add(owner[id(atom)])
            for other in neighbours:
                selected.add(owner[id(other)])

    return frozenset(selected)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InterfaceAgreement:
    """How far the two interface definitions agree on one complex."""

    n_delta_sasa: int
    n_contact: int
    n_both: int
    n_delta_sasa_only: int
    n_contact_only: int
    jaccard: float

    def to_row(self) -> dict[str, float | int]:
        return {
            "n_interface_delta_sasa": self.n_delta_sasa,
            "n_interface_contact": self.n_contact,
            "n_interface_both": self.n_both,
            "n_interface_delta_sasa_only": self.n_delta_sasa_only,
            "n_interface_contact_only": self.n_contact_only,
            "interface_definition_jaccard": self.jaccard,
        }


def compare_definitions(
    delta_sasa_set: frozenset[ResidueId],
    contact_set: frozenset[ResidueId],
) -> InterfaceAgreement:
    """Quantify the overlap between the two interface definitions.

    Reported per complex so that a reader can see whether a conclusion depends
    on the choice. If the Jaccard index is high the choice is immaterial; if it
    is low the paper has to say which definition it used and show the other.
    """
    both = delta_sasa_set & contact_set
    union = delta_sasa_set | contact_set
    return InterfaceAgreement(
        n_delta_sasa=len(delta_sasa_set),
        n_contact=len(contact_set),
        n_both=len(both),
        n_delta_sasa_only=len(delta_sasa_set - contact_set),
        n_contact_only=len(contact_set - delta_sasa_set),
        jaccard=(len(both) / len(union)) if union else 0.0,
    )


@dataclass(frozen=True, slots=True)
class ResidueClassification:
    """Per-residue partition assignment for one chain, plus the SASA it rests on."""

    chain_id: str
    partition: dict[ResidueId, Partition]
    delta_sasa: dict[ResidueId, float]
    monomer_sasa: dict[ResidueId, float]
    relative_monomer_sasa: dict[ResidueId, float]

    def ids_in(self, partition: Partition) -> frozenset[ResidueId]:
        return frozenset(rid for rid, p in self.partition.items() if p is partition)

    def counts(self) -> dict[str, int]:
        return {
            p.value: sum(1 for value in self.partition.values() if value is p) for p in Partition
        }

    def buried_surface_area_a2(self) -> float:
        """Total surface area buried by complexation, in square angstroms.

        The conventional interface size descriptor. Reported per chain, so the
        commonly quoted buried surface area of the whole interface is the sum
        over the two chains, or half of it if the convention being followed
        halves it. Say which in the methods.
        """
        return float(sum(self.delta_sasa.values()))


@dataclass(frozen=True, slots=True)
class InterfaceDefinition:
    """The frozen interface of one complex, with everything needed to reproduce it.

    ``source`` is either ``"native"`` or ``"predicted"``. Anything that
    partitions charge must assert that it received a native definition, because
    a predicted one drifts with beta and would confound the very trend being
    measured.
    """

    pdb_id: str
    chain_a: str
    chain_b: str
    source: Literal["native", "predicted"]
    structure_sha256: str
    delta_sasa_set: frozenset[ResidueId]
    contact_set: frozenset[ResidueId]
    agreement: InterfaceAgreement
    classification_a: ResidueClassification
    classification_b: ResidueClassification
    params: InterfaceParams

    def require_native(self) -> None:
        """Raise unless this definition came from the native structure."""
        if self.source != "native":
            raise ValueError(
                "charge partitioning requires an interface defined on the native "
                f"structure, but this definition has source={self.source!r}. "
                "A predicted interface moves with beta, which would confound the "
                "charge trend with a change in what counts as interface."
            )

    def classification_for(self, chain_id: str) -> ResidueClassification:
        if chain_id == self.chain_a:
            return self.classification_a
        if chain_id == self.chain_b:
            return self.classification_b
        raise KeyError(
            f"chain {chain_id!r} is not part of this interface definition "
            f"({self.chain_a}, {self.chain_b})"
        )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_residues(
    extract: ChainExtract,
    delta_sasa: dict[ResidueId, float],
    monomer_sasa: dict[ResidueId, float],
    interface_set: frozenset[ResidueId],
    interface_params: InterfaceParams,
) -> ResidueClassification:
    """Assign every residue of a chain to exactly one of the three partitions.

    The order of the tests matters and is deliberate:

    1. A residue whose relative SASA **in the isolated monomer** is below the
       core cutoff is core. Burial in the monomer is a property of the fold and
       has nothing to do with the partner.
    2. Otherwise, a residue in ``interface_set`` is interface.
    3. Everything else is non-interface surface.

    Testing for core first is what makes the partitions disjoint. A residue
    cannot be simultaneously buried inside its own fold and desolvated by
    binding, and if the thresholds ever produced such a residue, resolving it in
    favour of core is the conservative choice: it keeps the interface set to
    genuinely solvent-exposed residues.
    """
    partition: dict[ResidueId, Partition] = {}
    relative: dict[ResidueId, float] = {}

    for position, rid in enumerate(extract.residue_ids()):
        aa = extract.sequence[position]
        max_asa = MAX_ASA_TIEN_2013[aa]
        monomer = float(monomer_sasa.get(rid, 0.0))
        rel = monomer / max_asa
        relative[rid] = rel

        if rel < interface_params.core_relative_sasa_max:
            partition[rid] = Partition.CORE
        elif rid in interface_set:
            partition[rid] = Partition.INTERFACE
        else:
            partition[rid] = Partition.SURFACE

    return ResidueClassification(
        chain_id=extract.chain_id,
        partition=partition,
        delta_sasa={rid: float(delta_sasa.get(rid, 0.0)) for rid in partition},
        monomer_sasa={rid: float(monomer_sasa.get(rid, 0.0)) for rid in partition},
        relative_monomer_sasa=relative,
    )


def define_interface(
    model: Model,
    pdb_id: str,
    chain_a: str,
    chain_b: str,
    structure_sha256: str,
    interface_params: InterfaceParams,
    structure_params: StructureParams,
    source: Literal["native", "predicted"] = "native",
) -> InterfaceDefinition:
    """Compute both interface definitions and the three-way partition of each chain.

    This is the single entry point the scripts use. It runs the SASA
    calculation three times (complex, chain A alone, chain B alone) and reuses
    those numbers for the interface set, the buried surface area and the core
    assignment, rather than recomputing.
    """
    delta_set, delta_map = interface_by_delta_sasa(
        model, chain_a, chain_b, interface_params, structure_params
    )
    contact_set = interface_by_contact(model, chain_a, chain_b, interface_params, structure_params)

    monomer_a = residue_sasa(model, (chain_a,), interface_params, structure_params)
    monomer_b = residue_sasa(model, (chain_b,), interface_params, structure_params)

    extract_a = extract_chain(model, chain_a, structure_params)
    extract_b = extract_chain(model, chain_b, structure_params)

    return InterfaceDefinition(
        pdb_id=pdb_id,
        chain_a=chain_a,
        chain_b=chain_b,
        source=source,
        structure_sha256=structure_sha256,
        delta_sasa_set=delta_set,
        contact_set=contact_set,
        agreement=compare_definitions(delta_set, contact_set),
        classification_a=classify_residues(
            extract_a, delta_map, monomer_a, delta_set, interface_params
        ),
        classification_b=classify_residues(
            extract_b, delta_map, monomer_b, delta_set, interface_params
        ),
        params=interface_params,
    )


def recompute_on_predicted(
    model: Model,
    pdb_id: str,
    chain_a: str,
    chain_b: str,
    structure_sha256: str,
    interface_params: InterfaceParams,
    structure_params: StructureParams,
) -> InterfaceDefinition:
    """Recompute the interface on a predicted structure. Clearly labelled as such.

    This answers a different question from :func:`define_interface`: not "where
    is the interface" but "does the interface still exist after the charge
    shift". Use it to compare interface size and composition between the native
    and the AlphaFold2-Multimer prediction at each beta. Never use its output to
    partition charge, which :meth:`InterfaceDefinition.require_native` enforces.
    """
    return define_interface(
        model,
        pdb_id=pdb_id,
        chain_a=chain_a,
        chain_b=chain_b,
        structure_sha256=structure_sha256,
        interface_params=interface_params,
        structure_params=structure_params,
        source="predicted",
    )


def interface_positions(
    extract: ChainExtract,
    classification: ResidueClassification,
) -> tuple[frozenset[int], frozenset[int]]:
    """Zero-based sequence positions of the interface and core residues of a chain.

    This is the bridge from structure space to sequence space, and it is the
    step where an off-by-one becomes a wrong charge partition, so it goes
    through :meth:`ChainExtract.positions_of`, which raises on any residue it
    cannot place rather than dropping it.
    """
    interface_ids = classification.ids_in(Partition.INTERFACE)
    core_ids = classification.ids_in(Partition.CORE)
    return extract.positions_of(interface_ids), extract.positions_of(core_ids)


def summarise_delta_sasa(delta: dict[ResidueId, float]) -> dict[str, float]:
    """Descriptive statistics of a delta-SASA map, for the manifest."""
    values = np.array(list(delta.values()), dtype=np.float64)
    if values.size == 0:
        return {"total": 0.0, "max": 0.0, "mean_nonzero": 0.0, "n_nonzero": 0}
    nonzero = values[values > 0]
    return {
        "total": float(values.sum()),
        "max": float(values.max()),
        "mean_nonzero": float(nonzero.mean()) if nonzero.size else 0.0,
        "n_nonzero": int(nonzero.size),
    }
