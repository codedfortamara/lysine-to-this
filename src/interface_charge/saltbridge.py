"""Salt-bridge counting under two definitions, with the normalisations that make
the counts comparable.

Why two definitions
-------------------
A salt bridge is conventionally a charged-group heavy-atom contact: a nitrogen
of a lysine or arginine within about four angstroms of a carboxylate oxygen of
an aspartate or glutamate (Barlow and Thornton, 1983). A much cheaper proxy is
to measure between Cbeta atoms with a looser cutoff, typically six angstroms.

The proxy is not a conservative approximation of the geometric definition, it is
a different quantity. Cbeta sits at the base of the side chain, so the criterion
is indifferent to which way the charged group points: two residues whose side
chains splay apart score identically to two forming a genuine ion pair. On a
folded protein the proxy typically returns several times more pairs than the
geometric definition, and the ratio is not constant between structures.

Both are implemented here so a claim resting on one can be checked against the
other on the same coordinates, rather than argued about in the abstract.

Why normalisation matters more than the raw count
-------------------------------------------------
Inverse-folding models are known to emit more charged surface residues than
occur natively. If a design carries more lysine, arginine, aspartate and
glutamate than its native counterpart, the raw number of salt bridges rises
whether or not the model has placed any of them well. A raw count therefore
cannot distinguish *more charge* from *better arranged charge*, and only the
second is a statement about the model's spatial reasoning.

Three normalisations are returned alongside the raw count:

``per_charged_residue``
    Bridges divided by the number of charged residues present. Answers: given
    the charge this sequence carries, how much of it is paired?
``per_opportunity``
    Bridges divided by the number of cation-anion residue pairs that exist at
    all. This is the natural denominator, since that product is the number of
    bridges the sequence could in principle form.
``fraction_charged_residues_engaged``
    Fraction of charged residues participating in at least one bridge. Robust
    to a handful of residues each forming many bridges, which the other two are
    not.

Report the raw count if you like, but a comparison between sequences of
different charge composition needs at least one of these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
from Bio.PDB import NeighborSearch
from Bio.PDB.Model import Model

from .config import SaltBridgeParams, StructureParams
from .structures import ChainExtract, ResidueId, StructureError, residue_id, submodel

__all__ = [
    "SaltBridgeResult",
    "compare_salt_bridge_definitions",
    "count_salt_bridges",
    "count_salt_bridges_threaded",
]

#: Side-chain nitrogen atoms carrying the positive charge.
_CATIONIC_ATOMS: Final[dict[str, tuple[str, ...]]] = {
    "LYS": ("NZ",),
    "ARG": ("NE", "NH1", "NH2"),
    "HIS": ("ND1", "NE2"),
}

#: Side-chain carboxylate oxygens carrying the negative charge.
_ANIONIC_ATOMS: Final[dict[str, tuple[str, ...]]] = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
}

_CATIONIC_ONE_LETTER: Final[dict[str, str]] = {"LYS": "K", "ARG": "R", "HIS": "H"}
_ANIONIC_ONE_LETTER: Final[dict[str, str]] = {"ASP": "D", "GLU": "E"}


@dataclass(frozen=True, slots=True)
class SaltBridgeResult:
    """Salt-bridge count for one structure under one definition.

    ``n_bridges`` is the raw count. Prefer one of the normalised fields for any
    comparison between sequences that differ in charge composition, which is
    every design-versus-native comparison.
    """

    definition: Literal["geometric", "cbeta_proxy"]
    cutoff_a: float
    n_bridges: int
    n_cationic_residues: int
    n_anionic_residues: int
    n_charged_residues: int
    n_opportunities: int
    n_engaged_residues: int
    include_histidine: bool
    scope: str

    @property
    def per_charged_residue(self) -> float:
        """Bridges per charged residue. Zero if the sequence carries no charge."""
        return self.n_bridges / self.n_charged_residues if self.n_charged_residues else 0.0

    @property
    def per_opportunity(self) -> float:
        """Bridges divided by the number of cation-anion residue pairs available."""
        return self.n_bridges / self.n_opportunities if self.n_opportunities else 0.0

    @property
    def fraction_charged_residues_engaged(self) -> float:
        """Fraction of charged residues in at least one bridge."""
        return self.n_engaged_residues / self.n_charged_residues if self.n_charged_residues else 0.0

    def per_1000_a2(self, buried_surface_area_a2: float) -> float:
        """Bridges per 1000 square angstroms of buried interface.

        A third normalisation, and the one least like the other two. Both
        ``per_charged_residue`` and ``per_opportunity`` divide by something that
        grows with the charge being added, and ``per_opportunity`` divides by a
        Cartesian product that grows quadratically, so a flat ratio there is
        weaker evidence than it looks: the denominator can outrun a real gain.

        Buried area is fixed by the native backbone and does not move with the
        charge dial at all, so this asks a different question. How many bridges
        does this interface carry, for its size? If the raw count rises while
        this rises too, the interface really is carrying more bridges per unit of
        contact, whatever the opportunity ratio says.
        """
        if buried_surface_area_a2 <= 0:
            raise ValueError(
                f"buried surface area must be positive, got {buried_surface_area_a2}. "
                "A zero or negative area means the interface was not computed, and "
                "dividing by it would produce a density from nothing."
            )
        return 1000.0 * self.n_bridges / buried_surface_area_a2

    def to_row(self) -> dict[str, Any]:
        """Flat mapping for a results table, with the definition in the names."""
        tag = self.definition
        return {
            f"sb_n_bridges_{tag}": self.n_bridges,
            f"sb_per_charged_residue_{tag}": self.per_charged_residue,
            f"sb_per_opportunity_{tag}": self.per_opportunity,
            f"sb_fraction_engaged_{tag}": self.fraction_charged_residues_engaged,
            f"sb_cutoff_a_{tag}": self.cutoff_a,
            "sb_n_cationic_residues": self.n_cationic_residues,
            "sb_n_anionic_residues": self.n_anionic_residues,
            "sb_n_charged_residues": self.n_charged_residues,
            "sb_n_opportunities": self.n_opportunities,
            "sb_scope": self.scope,
        }


def _charged_atoms(
    model: Model,
    chain_ids: tuple[str, ...],
    params: SaltBridgeParams,
    structure_params: StructureParams,
    definition: str,
) -> tuple[list, list, dict[int, ResidueId], set[ResidueId], set[ResidueId]]:
    """Collect the atoms each definition measures between."""
    clone = submodel(model, chain_ids, structure_params)

    cations: list = []
    anions: list = []
    owner: dict[int, ResidueId] = {}
    cationic_residues: set[ResidueId] = set()
    anionic_residues: set[ResidueId] = set()

    for chain in clone:
        for res in chain:
            name = res.get_resname().strip().upper()
            rid = residue_id(res, chain.id)

            is_cation = name in _CATIONIC_ATOMS and (name != "HIS" or params.include_histidine)
            is_anion = name in _ANIONIC_ATOMS
            if not (is_cation or is_anion):
                continue

            if is_cation:
                cationic_residues.add(rid)
            else:
                anionic_residues.add(rid)

            if definition == "geometric":
                wanted = _CATIONIC_ATOMS[name] if is_cation else _ANIONIC_ATOMS[name]
            else:
                wanted = ("CB",)

            for atom in res:
                if atom.get_id() not in wanted:
                    continue
                if atom.get_altloc() not in structure_params.accepted_altlocs:
                    continue
                owner[id(atom)] = rid
                (cations if is_cation else anions).append(atom)

    return cations, anions, owner, cationic_residues, anionic_residues


def count_salt_bridges(
    model: Model,
    chain_ids: tuple[str, ...],
    params: SaltBridgeParams,
    structure_params: StructureParams,
    definition: Literal["geometric", "cbeta_proxy"] = "geometric",
    cross_chain_only: bool = False,
) -> SaltBridgeResult:
    """Count salt bridges over the given chains.

    Parameters
    ----------
    chain_ids
        Chains to consider. Pass both chains of a complex for a whole-complex
        count, or one chain for an intra-chain count.
    definition
        ``geometric`` measures between charged-group heavy atoms at
        ``params.cutoff_a``. ``cbeta_proxy`` measures between Cbeta atoms at
        ``params.proxy_cbeta_cutoff_a``.
    cross_chain_only
        Count only bridges spanning two different chains. This is the quantity
        of interest for an interface claim; the default counts everything,
        which is dominated by intra-chain pairs.

    Notes
    -----
    A residue pair is counted once however many atom pairs satisfy the cutoff,
    so an arginine forming contacts through both of its terminal nitrogens is
    one bridge, not two. Counting atom pairs instead would weight arginine
    three times more heavily than lysine purely because it has more nitrogens.
    """
    cutoff = params.cutoff_a if definition == "geometric" else params.proxy_cbeta_cutoff_a

    cations, anions, owner, cationic_residues, anionic_residues = _charged_atoms(
        model, chain_ids, params, structure_params, definition
    )

    n_cationic = len(cationic_residues)
    n_anionic = len(anionic_residues)
    scope = "cross_chain" if cross_chain_only else "all"

    if not cations or not anions:
        return SaltBridgeResult(
            definition=definition,
            cutoff_a=cutoff,
            n_bridges=0,
            n_cationic_residues=n_cationic,
            n_anionic_residues=n_anionic,
            n_charged_residues=n_cationic + n_anionic,
            n_opportunities=n_cationic * n_anionic,
            n_engaged_residues=0,
            include_histidine=params.include_histidine,
            scope=scope,
        )

    search = NeighborSearch(anions)
    pairs: set[tuple[ResidueId, ResidueId]] = set()

    for atom in cations:
        cation_rid = owner[id(atom)]
        for other in search.search(atom.get_coord(), cutoff, level="A"):
            anion_rid = owner[id(other)]
            if cross_chain_only and cation_rid.chain == anion_rid.chain:
                continue
            pairs.add((cation_rid, anion_rid))

    engaged = {rid for pair in pairs for rid in pair}

    return SaltBridgeResult(
        definition=definition,
        cutoff_a=cutoff,
        n_bridges=len(pairs),
        n_cationic_residues=n_cationic,
        n_anionic_residues=n_anionic,
        n_charged_residues=n_cationic + n_anionic,
        n_opportunities=n_cationic * n_anionic,
        n_engaged_residues=len(engaged),
        include_histidine=params.include_histidine,
        scope=scope,
    )


def count_salt_bridges_threaded(
    model: Model,
    extract_a: ChainExtract,
    extract_b: ChainExtract,
    params: SaltBridgeParams,
    structure_params: StructureParams,
    cross_chain_only: bool = False,
) -> SaltBridgeResult:
    """Cbeta-proxy salt bridges for sequences threaded onto a fixed backbone.

    Why only the proxy definition is available here
    -----------------------------------------------
    The geometric definition measures between charged-group heavy atoms, and a
    threaded design does not have them: replacing a native aspartate with a
    designed lysine changes which side-chain atoms exist and where they point,
    and recovering that would require side-chain repacking. Cbeta sits on the
    backbone side of the first side-chain bond, so its position is unchanged by
    the substitution and the proxy remains computable.

    That is a real constraint rather than an oversight, and it is why a
    threaded-design analysis is limited to the coarser definition. What it does
    **not** excuse is comparing raw counts between sequences of different charge
    composition, which is what the normalised fields on the result exist for.

    Charge is taken from the ``sequence`` of each extract, so passing a design's
    sequence via :meth:`ChainExtract.with_sequence` measures that design on the
    native backbone.
    """
    cationic = set("KRH") if params.include_histidine else set("KR")
    anionic = set("DE")

    atoms: list = []
    owner: dict[int, ResidueId] = {}
    sign: dict[ResidueId, int] = {}
    clone = submodel(model, (extract_a.chain_id, extract_b.chain_id), structure_params)

    by_chain = {extract_a.chain_id: extract_a, extract_b.chain_id: extract_b}
    for chain in clone:
        extract = by_chain.get(chain.id)
        if extract is None:
            continue
        for res in chain:
            rid = residue_id(res, chain.id)
            position = extract.index_of.get(rid)
            if position is None:
                continue
            aa = extract.sequence[position]
            if aa in cationic:
                sign[rid] = 1
            elif aa in anionic:
                sign[rid] = -1
            else:
                continue
            for atom in res:
                if atom.get_id() != "CB":
                    continue
                if atom.get_altloc() not in structure_params.accepted_altlocs:
                    continue
                owner[id(atom)] = rid
                atoms.append(atom)

    cations = [a for a in atoms if sign[owner[id(a)]] > 0]
    anions = [a for a in atoms if sign[owner[id(a)]] < 0]
    n_cationic = sum(1 for v in sign.values() if v > 0)
    n_anionic = sum(1 for v in sign.values() if v < 0)
    scope = "cross_chain" if cross_chain_only else "all"

    pairs: set[tuple[ResidueId, ResidueId]] = set()
    if cations and anions:
        search = NeighborSearch(anions)
        for atom in cations:
            cation_rid = owner[id(atom)]
            for other in search.search(atom.get_coord(), params.proxy_cbeta_cutoff_a, level="A"):
                anion_rid = owner[id(other)]
                if cross_chain_only and cation_rid.chain == anion_rid.chain:
                    continue
                pairs.add((cation_rid, anion_rid))

    engaged = {rid for pair in pairs for rid in pair}
    return SaltBridgeResult(
        definition="cbeta_proxy",
        cutoff_a=params.proxy_cbeta_cutoff_a,
        n_bridges=len(pairs),
        n_cationic_residues=n_cationic,
        n_anionic_residues=n_anionic,
        n_charged_residues=n_cationic + n_anionic,
        n_opportunities=n_cationic * n_anionic,
        n_engaged_residues=len(engaged),
        include_histidine=params.include_histidine,
        scope=scope,
    )


def compare_salt_bridge_definitions(
    model: Model,
    chain_ids: tuple[str, ...],
    params: SaltBridgeParams,
    structure_params: StructureParams,
    cross_chain_only: bool = False,
) -> dict[str, Any]:
    """Run both definitions on the same coordinates and quantify the gap.

    ``proxy_inflation`` is the ratio of proxy count to geometric count. A value
    well above one means the proxy is counting residue pairs that are near each
    other without forming an ion pair, which is the expected behaviour rather
    than a bug. Reporting it converts a methodological disagreement into a
    measured number.
    """
    geometric = count_salt_bridges(
        model, chain_ids, params, structure_params, "geometric", cross_chain_only
    )
    proxy = count_salt_bridges(
        model, chain_ids, params, structure_params, "cbeta_proxy", cross_chain_only
    )
    return {
        "geometric": geometric,
        "cbeta_proxy": proxy,
        "proxy_inflation": (
            proxy.n_bridges / geometric.n_bridges if geometric.n_bridges else float("nan")
        ),
    }


def charged_composition(extract: ChainExtract, include_histidine: bool = False) -> dict[str, int]:
    """Charged residue counts of a sequence.

    The denominator behind every normalisation, and the quantity that has to be
    matched before two salt-bridge counts can be compared directly.
    """
    cationic = "KRH" if include_histidine else "KR"
    counts = {
        "n_cationic": sum(extract.sequence.count(aa) for aa in cationic),
        "n_anionic": sum(extract.sequence.count(aa) for aa in "DE"),
        "n_residues": len(extract.sequence),
    }
    counts["n_charged"] = counts["n_cationic"] + counts["n_anionic"]
    counts["n_opportunities"] = counts["n_cationic"] * counts["n_anionic"]
    return counts


def bridges_expected_by_chance(
    n_cationic: int,
    n_anionic: int,
    n_residues: int,
    observed_density: float,
) -> float:
    """Bridges a sequence would form if charge were placed without regard to structure.

    A crude null model: the expected count scales with the number of
    cation-anion pairs available. It exists so that a raw-count comparison
    between two sequences of different composition can be corrected to first
    order rather than taken at face value.

    ``observed_density`` is bridges per opportunity measured on a reference set.
    Multiply it by this sequence's opportunity count to get the expectation.
    """
    if n_residues <= 0:
        raise ValueError("n_residues must be positive")
    return float(n_cationic * n_anionic * observed_density)


def summarise(results: list[SaltBridgeResult]) -> dict[str, float]:
    """Mean and spread of a set of counts, for a quick look before the real statistics."""
    if not results:
        return {}
    raw = np.array([r.n_bridges for r in results], dtype=float)
    normalised = np.array([r.per_opportunity for r in results], dtype=float)
    return {
        "n": len(results),
        "mean_raw": float(raw.mean()),
        "median_raw": float(np.median(raw)),
        "mean_per_opportunity": float(normalised.mean()),
        "median_per_opportunity": float(np.median(normalised)),
    }


def require_matched_composition(
    left: dict[str, int],
    right: dict[str, int],
    tolerance: float = 0.10,
) -> None:
    """Raise if two sequences differ enough in charge composition to invalidate a
    raw-count comparison.

    Called before any design-versus-native raw comparison. If the charged
    residue counts differ by more than ``tolerance``, the raw counts are not
    comparable and a normalised statistic has to be used instead. Raising is
    deliberate: the failure mode this guards against produces a plausible number
    rather than an error.
    """
    a, b = left["n_charged"], right["n_charged"]
    if a == 0 or b == 0:
        raise StructureError("cannot compare salt-bridge counts when a sequence carries no charge")
    relative = abs(a - b) / max(a, b)
    if relative > tolerance:
        raise StructureError(
            f"charged residue counts differ by {relative:.1%} ({a} versus {b}), above the "
            f"{tolerance:.0%} tolerance. Raw salt-bridge counts are not comparable across "
            "sequences of different charge composition: more charged residues produce more "
            "bridges whether or not they are better placed. Use per_opportunity or "
            "fraction_charged_residues_engaged instead."
        )
