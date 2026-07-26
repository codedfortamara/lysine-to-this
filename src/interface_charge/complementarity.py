"""Electrostatic complementarity across a protein-protein interface.

The statistic is the summed product of formal residue charges over contacting
cross-chain residue pairs:

.. math::

    S = \\sum_{(i, j) \\in \\mathrm{contacts}} q_i \\, q_j

A **negative** sum means opposite charges face each other, which is a
complementary interface. A positive sum means like charges are apposed, which
is the electrostatic signature of a design that has been pushed past the point
where the interface can absorb the shift.

Why not Poisson-Boltzmann
-------------------------
APBS or any other Poisson-Boltzmann solver would give a continuum
electrostatic energy, which is a strictly richer quantity. It is deliberately
not used here, for two reasons. First, it is a heavy dependency with its own
parameterisation choices (dielectric boundary, ionic strength, protonation
state assignment) each of which would need justifying and none of which the
claim requires. Second, and decisively, this project does not cite a tool it
has not run end to end. The claim being made is a comparative one, that
complementarity degrades in a particular direction as beta moves, and a
contact-level charge product supports that comparison without asserting an
energy.

Limitations, to be stated wherever the number appears
----------------------------------------------------
This is a formal-charge contact statistic, not an energy. It has no dielectric
screening, no distance dependence unless ``weight="inverse_distance"`` is set,
no explicit ions, and no desolvation term. It is meaningful when compared
across designs of the same complex, which is exactly how it is used here, and
it is close to meaningless as an absolute number compared across different
complexes. Normalise by pair count before comparing complexes, and say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from Bio.PDB import NeighborSearch
from Bio.PDB.Model import Model

from .charge import per_residue_charge_ph, per_residue_charge_simple
from .config import ChargeParams, ComplementarityParams, StructureParams
from .structures import ChainExtract, ResidueId, StructureError, heavy_atoms, residue_id, submodel

__all__ = [
    "ComplementarityResult",
    "ContactPair",
    "cross_chain_contacts",
    "electrostatic_complementarity",
]


@dataclass(frozen=True, slots=True)
class ContactPair:
    """One contacting cross-chain residue pair and its minimum heavy-atom distance."""

    residue_a: ResidueId
    residue_b: ResidueId
    min_distance_a: float


@dataclass(frozen=True, slots=True)
class ComplementarityResult:
    """Outcome of one electrostatic complementarity calculation.

    ``sum_product`` is the headline number and is negative for a complementary
    interface. ``mean_product`` is ``sum_product`` divided by the number of
    contacting pairs and is the form to use when comparing across complexes of
    different interface size.
    """

    sum_product: float
    mean_product: float
    n_pairs: int
    n_attractive: int
    n_repulsive: int
    n_neutral: int
    charge_definition: str
    contact_cutoff_a: float
    weight: str
    ph: float | None
    pka_set: str | None

    def to_row(self) -> dict[str, Any]:
        """Flat mapping for a results table, with the definition in the column names.

        As in :mod:`interface_charge.charge`, the charge definition is carried
        in the column name rather than in a separate column, so two runs under
        different definitions cannot be concatenated into one column by
        accident.
        """
        tag = self.charge_definition
        if self.charge_definition == "ph":
            tag = f"ph{self.ph:g}_{self.pka_set}"
        return {
            f"ec_sum_product_{tag}": self.sum_product,
            f"ec_mean_product_{tag}": self.mean_product,
            "ec_n_pairs": self.n_pairs,
            "ec_n_attractive": self.n_attractive,
            "ec_n_repulsive": self.n_repulsive,
            "ec_n_neutral": self.n_neutral,
            "ec_contact_cutoff_a": self.contact_cutoff_a,
            "ec_weight": self.weight,
        }


def cross_chain_contacts(
    model: Model,
    chain_a: str,
    chain_b: str,
    cutoff_a: float,
    structure_params: StructureParams,
) -> list[ContactPair]:
    """Every residue pair across the two chains with a heavy-atom contact.

    The minimum heavy-atom distance is retained per pair so that a distance
    weighting can be applied without a second pass over the coordinates.
    """
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
    best: dict[tuple[ResidueId, ResidueId], float] = {}

    for atom in atoms_a:
        coord = atom.get_coord()
        for other in search.search(coord, cutoff_a, level="A"):
            distance = float(np.linalg.norm(coord - other.get_coord()))
            key = (owner[id(atom)], owner[id(other)])
            if distance < best.get(key, np.inf):
                best[key] = distance

    return [
        ContactPair(residue_a=a, residue_b=b, min_distance_a=d)
        for (a, b), d in sorted(best.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))
    ]


def _charges_for(extract: ChainExtract, params: ComplementarityParams, charge: ChargeParams):
    if params.charge_definition == "simple":
        return per_residue_charge_simple(extract.sequence).astype(np.float64)
    if params.charge_definition == "ph":
        return per_residue_charge_ph(
            extract.sequence,
            ph=charge.ph,
            pka_set=charge.pka_set,
            include_cys_tyr=charge.include_cys_tyr,
            include_termini=charge.include_termini,
        )
    raise ValueError(
        f"charge_definition must be 'simple' or 'ph', got {params.charge_definition!r}"
    )


def electrostatic_complementarity(
    model: Model,
    extract_a: ChainExtract,
    extract_b: ChainExtract,
    params: ComplementarityParams,
    charge_params: ChargeParams,
    structure_params: StructureParams,
) -> ComplementarityResult:
    """Summed charge product over contacting cross-chain residue pairs.

    Parameters
    ----------
    model
        The structure holding both chains. For the primary analysis this is the
        native complex, so that the contact set is fixed while the sequence
        charges vary with beta. Passing a predicted complex answers the
        different question of whether the predicted geometry is itself
        complementary, and the two must be reported separately.
    extract_a, extract_b
        The two chains, whose ``sequence`` supplies the per-residue charges and
        whose ``index_of`` maps structure residues onto sequence positions.

    Returns
    -------
    ComplementarityResult
        ``sum_product`` negative means complementary.
    """
    q_a = _charges_for(extract_a, params, charge_params)
    q_b = _charges_for(extract_b, params, charge_params)

    pairs = cross_chain_contacts(
        model,
        extract_a.chain_id,
        extract_b.chain_id,
        params.contact_cutoff_a,
        structure_params,
    )

    total = 0.0
    n_attractive = 0
    n_repulsive = 0
    n_neutral = 0
    n_counted = 0

    for pair in pairs:
        pos_a = extract_a.index_of.get(pair.residue_a)
        pos_b = extract_b.index_of.get(pair.residue_b)
        if pos_a is None or pos_b is None:
            # A contact involving a residue that was excluded from the sequence,
            # for example an unmapped modified residue. extract_chain already
            # raises on those, so reaching here means the extract and the model
            # disagree, which must not be silently ignored.
            raise StructureError(
                f"contact pair {pair.residue_a} to {pair.residue_b} references a "
                "residue absent from the chain extract. The extract and the model "
                "were built from different structures."
            )

        product = float(q_a[pos_a] * q_b[pos_b])

        if params.weight == "inverse_distance":
            distance = max(pair.min_distance_a, 1e-6)
            product /= distance
        elif params.weight != "none":
            raise ValueError(f"unknown weight {params.weight!r}")

        if product < 0:
            n_attractive += 1
        elif product > 0:
            n_repulsive += 1
        else:
            n_neutral += 1
            if not params.count_neutral_pairs:
                total += product  # exactly zero, kept for clarity
                continue

        total += product
        n_counted += 1

    denominator = n_counted if n_counted else 1
    return ComplementarityResult(
        sum_product=total,
        mean_product=total / denominator,
        n_pairs=len(pairs),
        n_attractive=n_attractive,
        n_repulsive=n_repulsive,
        n_neutral=n_neutral,
        charge_definition=params.charge_definition,
        contact_cutoff_a=params.contact_cutoff_a,
        weight=params.weight,
        ph=charge_params.ph if params.charge_definition == "ph" else None,
        pka_set=charge_params.pka_set if params.charge_definition == "ph" else None,
    )
