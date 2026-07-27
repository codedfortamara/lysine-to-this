"""Interface detection under both definitions, and the three-way partition.

These run against the committed complex fixture. The assertions are structural
invariants that correct code must satisfy, plus a small number of checks
against established properties of a real protein-protein interface (an
interface has tens rather than hundreds or zero residues, and it buries several
hundred square angstroms). They are not pinned to numbers copied out of a
previous run, which would only test that the code has not changed rather than
that it is right.
"""

from __future__ import annotations

import pytest

from interface_charge.charge import partition_charge
from interface_charge.config import MAX_ASA_TIEN_2013, InterfaceParams
from interface_charge.interface import (
    Partition,
    compare_definitions,
    interface_by_contact,
    interface_by_delta_sasa,
    interface_positions,
    recompute_on_predicted,
    residue_sasa,
)
from interface_charge.structures import StructureError

# ---------------------------------------------------------------------------
# Both definitions find an interface
# ---------------------------------------------------------------------------


def test_delta_sasa_finds_a_plausible_interface(interface_definition) -> None:
    """A real two-chain complex has an interface of tens of residues."""
    found = interface_definition.delta_sasa_set
    assert 10 < len(found) < 300


def test_contact_definition_finds_a_plausible_interface(interface_definition) -> None:
    found = interface_definition.contact_set
    assert 10 < len(found) < 300


def test_both_definitions_span_both_chains(interface_definition) -> None:
    """An interface with residues on only one side is not an interface."""
    for name, residues in (
        ("delta_sasa", interface_definition.delta_sasa_set),
        ("contact", interface_definition.contact_set),
    ):
        chains = {rid.chain for rid in residues}
        assert chains == {
            interface_definition.chain_a,
            interface_definition.chain_b,
        }, f"{name} definition did not span both chains"


def test_the_two_definitions_substantially_agree(interface_definition) -> None:
    """They should overlap heavily without being identical.

    Identical sets would mean one is computing the other. No overlap would mean
    one of them is wrong.
    """
    agreement = interface_definition.agreement
    assert 0.5 < agreement.jaccard < 1.0
    assert agreement.n_both > 0


def test_agreement_metrics_are_self_consistent(interface_definition) -> None:
    agreement = interface_definition.agreement
    assert agreement.n_both + agreement.n_delta_sasa_only == agreement.n_delta_sasa
    assert agreement.n_both + agreement.n_contact_only == agreement.n_contact


def test_compare_definitions_on_known_sets() -> None:
    """Hand-computed Jaccard: two of three shared, so 2/4."""
    from interface_charge.structures import ResidueId

    a = frozenset({ResidueId("A", 1, " "), ResidueId("A", 2, " "), ResidueId("A", 3, " ")})
    b = frozenset({ResidueId("A", 2, " "), ResidueId("A", 3, " "), ResidueId("A", 4, " ")})
    agreement = compare_definitions(a, b)
    assert agreement.n_both == 2
    assert agreement.n_delta_sasa_only == 1
    assert agreement.n_contact_only == 1
    assert agreement.jaccard == pytest.approx(2 / 4)


def test_compare_definitions_on_empty_sets_does_not_divide_by_zero() -> None:
    agreement = compare_definitions(frozenset(), frozenset())
    assert agreement.jaccard == 0.0


def test_contact_definition_is_more_generous_at_a_larger_cutoff(
    model, structure_case, config
) -> None:
    """Widening the cutoff can only add residues, never remove them."""
    _, chain_a, chain_b = structure_case
    tight = interface_by_contact(
        model, chain_a, chain_b, InterfaceParams(contact_cutoff_a=4.0), config.structure
    )
    loose = interface_by_contact(
        model, chain_a, chain_b, InterfaceParams(contact_cutoff_a=8.0), config.structure
    )
    assert tight <= loose
    assert len(loose) > len(tight)


def test_raising_the_delta_sasa_threshold_shrinks_the_interface(
    model, structure_case, config
) -> None:
    _, chain_a, chain_b = structure_case
    permissive, _ = interface_by_delta_sasa(
        model, chain_a, chain_b, InterfaceParams(delta_sasa_threshold_a2=1.0), config.structure
    )
    strict, _ = interface_by_delta_sasa(
        model, chain_a, chain_b, InterfaceParams(delta_sasa_threshold_a2=25.0), config.structure
    )
    assert strict <= permissive


def test_same_chain_twice_raises(model, structure_case, config) -> None:
    _, chain_a, _ = structure_case
    with pytest.raises(ValueError, match="must differ"):
        interface_by_delta_sasa(model, chain_a, chain_a, config.interface, config.structure)
    with pytest.raises(ValueError, match="must differ"):
        interface_by_contact(model, chain_a, chain_a, config.interface, config.structure)


# ---------------------------------------------------------------------------
# SASA behaviour
# ---------------------------------------------------------------------------


def test_complexation_never_increases_solvent_accessibility(model, structure_case, config) -> None:
    """Physical requirement: adding a partner can only bury surface."""
    _, chain_a, chain_b = structure_case
    _, delta = interface_by_delta_sasa(model, chain_a, chain_b, config.interface, config.structure)
    assert all(value >= 0.0 for value in delta.values())


def test_interface_buries_a_realistic_area(interface_definition) -> None:
    """A genuine protein-protein interface buries hundreds of square angstroms.

    Typical crystallographic interfaces bury 600 to 2000 square angstroms per
    side. This catches a units error or a monomer-and-complex mix-up.
    """
    for classification in (
        interface_definition.classification_a,
        interface_definition.classification_b,
    ):
        area = classification.buried_surface_area_a2()
        assert 200.0 < area < 5000.0


def test_sasa_of_the_complex_is_less_than_the_sum_of_the_parts(
    model, structure_case, config
) -> None:
    _, chain_a, chain_b = structure_case
    complexed = sum(
        residue_sasa(model, (chain_a, chain_b), config.interface, config.structure).values()
    )
    apart = sum(
        sum(residue_sasa(model, (chain,), config.interface, config.structure).values())
        for chain in (chain_a, chain_b)
    )
    assert complexed < apart


def test_sasa_never_exceeds_the_tien_maximum_by_much(
    model, structure_case, config, extracts
) -> None:
    """A residue's SASA should not wildly exceed its Gly-X-Gly maximum.

    A gross violation would mean the radii or the probe are wrong. A modest one
    is expected, since the Tien values are for an extended tripeptide.
    """
    _, chain_a, _ = structure_case
    extract_a, _ = extracts
    sasa = residue_sasa(model, (chain_a,), config.interface, config.structure)
    for position, rid in enumerate(extract_a.residue_ids()):
        maximum = MAX_ASA_TIEN_2013[extract_a.sequence[position]]
        assert sasa[rid] <= maximum * 1.6


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def test_every_residue_lands_in_exactly_one_partition(interface_definition, extracts) -> None:
    for classification, extract in zip(
        (interface_definition.classification_a, interface_definition.classification_b),
        extracts,
        strict=True,
    ):
        counts = classification.counts()
        assert sum(counts.values()) == len(extract.sequence)
        assert set(counts) == {p.value for p in Partition}


def test_partitions_are_disjoint(interface_definition) -> None:
    classification = interface_definition.classification_a
    interface = classification.ids_in(Partition.INTERFACE)
    surface = classification.ids_in(Partition.SURFACE)
    core = classification.ids_in(Partition.CORE)
    assert not interface & surface
    assert not interface & core
    assert not surface & core


def test_all_three_partitions_are_populated(interface_definition) -> None:
    """A folded complex has a core, a surface and an interface."""
    for classification in (
        interface_definition.classification_a,
        interface_definition.classification_b,
    ):
        counts = classification.counts()
        assert counts[Partition.CORE.value] > 0
        assert counts[Partition.SURFACE.value] > 0
        assert counts[Partition.INTERFACE.value] > 0


def test_core_residues_are_less_exposed_than_surface_residues(interface_definition) -> None:
    classification = interface_definition.classification_a
    core = [
        classification.relative_monomer_sasa[rid] for rid in classification.ids_in(Partition.CORE)
    ]
    surface = [
        classification.relative_monomer_sasa[rid]
        for rid in classification.ids_in(Partition.SURFACE)
    ]
    assert max(core) <= min(surface)


def test_interface_positions_map_into_the_sequence(interface_definition, extracts) -> None:
    extract_a, _ = extracts
    interface_idx, core_idx = interface_positions(extract_a, interface_definition.classification_a)
    assert not interface_idx & core_idx
    assert all(0 <= i < len(extract_a.sequence) for i in interface_idx | core_idx)


def test_partition_charge_over_the_real_fixture_sums(interface_definition, extracts) -> None:
    """The identity that everything downstream depends on, on a real structure."""
    for extract, classification in zip(
        extracts,
        (interface_definition.classification_a, interface_definition.classification_b),
        strict=True,
    ):
        interface_idx, core_idx = interface_positions(extract, classification)
        partition = partition_charge(extract.sequence, interface_idx, core_idx)
        assert (
            partition.interface_simple + partition.surface_simple + partition.core_simple
            == partition.total_simple
        )
        assert (partition.interface_ph + partition.surface_ph + partition.core_ph) == pytest.approx(
            partition.total_ph, abs=1e-9
        )
        assert partition.n_total == len(extract.sequence)


# ---------------------------------------------------------------------------
# Native versus predicted
# ---------------------------------------------------------------------------


def test_native_definition_passes_require_native(interface_definition) -> None:
    interface_definition.require_native()  # must not raise


def test_predicted_definition_is_rejected_for_partitioning(model, structure_case, config) -> None:
    """A predicted interface must never be usable to partition charge.

    If it were, the partition boundary would move with beta and any trend in
    interface charge would be confounded with a change in what counts as
    interface.
    """
    path, chain_a, chain_b = structure_case
    definition = recompute_on_predicted(
        model,
        pdb_id=path.stem.upper()[:4],
        chain_a=chain_a,
        chain_b=chain_b,
        structure_sha256="test",
        interface_params=config.interface,
        structure_params=config.structure,
    )
    assert definition.source == "predicted"
    with pytest.raises(ValueError, match="native"):
        definition.require_native()


def test_classification_for_unknown_chain_raises(interface_definition) -> None:
    with pytest.raises(KeyError):
        interface_definition.classification_for("Z")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_interface_definition_is_deterministic(model, structure_case, config) -> None:
    """Two runs over the same structure must give byte-identical sets.

    Any nondeterminism here would make the interface drift between scripts 01
    and 03, which both define it.
    """
    _, chain_a, chain_b = structure_case
    first, _ = interface_by_delta_sasa(model, chain_a, chain_b, config.interface, config.structure)
    second, _ = interface_by_delta_sasa(model, chain_a, chain_b, config.interface, config.structure)
    assert first == second


def test_missing_chain_raises_with_a_useful_message(model, config) -> None:
    with pytest.raises(StructureError, match="not in model"):
        residue_sasa(model, ("Z",), config.interface, config.structure)


# ---------------------------------------------------------------------------
# Partner selection
# ---------------------------------------------------------------------------


def test_partner_is_chosen_by_contact(model, structure_case, config) -> None:
    from interface_charge.interface import choose_contacting_partner

    _, chain_a, chain_b = structure_case
    partner, contacts = choose_contacting_partner(
        model, chain_a, config.interface, config.structure
    )
    assert partner == chain_b
    assert contacts[chain_b] > 0


def test_partner_selection_is_symmetric_on_a_two_chain_complex(
    model, structure_case, config
) -> None:
    from interface_charge.interface import choose_contacting_partner

    _, chain_a, chain_b = structure_case
    assert (
        choose_contacting_partner(model, chain_a, config.interface, config.structure)[0] == chain_b
    )
    assert (
        choose_contacting_partner(model, chain_b, config.interface, config.structure)[0] == chain_a
    )


def test_no_contacting_partner_returns_none_rather_than_a_guess(
    model, structure_case, config
) -> None:
    """A chain with no protein partner must be excludable, not silently zeroed.

    Returning the largest other chain regardless of contact was the original
    behaviour, and it produced empty interfaces that propagated as
    legitimate-looking zeros into every partition and every average.
    """
    from interface_charge.interface import choose_contacting_partner
    from interface_charge.structures import submodel

    _, chain_a, _ = structure_case
    alone = submodel(model, (chain_a,), config.structure)
    partner, contacts = choose_contacting_partner(
        alone, chain_a, config.interface, config.structure
    )
    assert partner is None
    assert contacts == {}


def test_partner_selection_is_deterministic(model, structure_case, config) -> None:
    from interface_charge.interface import choose_contacting_partner

    _, chain_a, _ = structure_case
    first = choose_contacting_partner(model, chain_a, config.interface, config.structure)
    second = choose_contacting_partner(model, chain_a, config.interface, config.structure)
    assert first == second
