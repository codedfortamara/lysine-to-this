"""Salt-bridge counting: the two definitions, and the normalisations.

Run against the committed complex. The assertions are invariants and
established properties of real structures, not numbers lifted from a previous
run.
"""

from __future__ import annotations

import pytest

from interface_charge.config import SaltBridgeParams
from interface_charge.saltbridge import (
    charged_composition,
    compare_salt_bridge_definitions,
    count_salt_bridges,
    require_matched_composition,
    summarise,
)
from interface_charge.structures import StructureError

# ---------------------------------------------------------------------------
# Both definitions find bridges
# ---------------------------------------------------------------------------


def test_geometric_definition_finds_bridges(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    result = count_salt_bridges(
        model, (chain_a, chain_b), config.salt_bridge, config.structure, "geometric"
    )
    assert result.n_bridges > 0
    assert result.n_cationic_residues > 0
    assert result.n_anionic_residues > 0


def test_proxy_counts_more_than_the_geometric_definition(model, structure_case, config) -> None:
    """The proxy is a different quantity, not a conservative approximation.

    A six angstrom Cbeta criterion is indifferent to side-chain orientation, so
    it counts residue pairs that are merely near each other. On a real protein
    it should return substantially more pairs.
    """
    _, chain_a, chain_b = structure_case
    comparison = compare_salt_bridge_definitions(
        model, (chain_a, chain_b), config.salt_bridge, config.structure
    )
    assert comparison["cbeta_proxy"].n_bridges > comparison["geometric"].n_bridges
    assert comparison["proxy_inflation"] > 1.0


def test_both_definitions_see_the_same_charged_residues(model, structure_case, config) -> None:
    """The definitions differ in geometry only, never in composition."""
    _, chain_a, chain_b = structure_case
    comparison = compare_salt_bridge_definitions(
        model, (chain_a, chain_b), config.salt_bridge, config.structure
    )
    geometric, proxy = comparison["geometric"], comparison["cbeta_proxy"]
    assert geometric.n_charged_residues == proxy.n_charged_residues
    assert geometric.n_opportunities == proxy.n_opportunities


def test_a_wider_cutoff_finds_more_bridges(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    tight = count_salt_bridges(
        model, (chain_a, chain_b), SaltBridgeParams(cutoff_a=3.2), config.structure
    )
    loose = count_salt_bridges(
        model, (chain_a, chain_b), SaltBridgeParams(cutoff_a=5.0), config.structure
    )
    assert loose.n_bridges > tight.n_bridges


def test_histidine_can_be_included(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    without = count_salt_bridges(
        model, (chain_a, chain_b), SaltBridgeParams(include_histidine=False), config.structure
    )
    with_his = count_salt_bridges(
        model, (chain_a, chain_b), SaltBridgeParams(include_histidine=True), config.structure
    )
    assert with_his.n_cationic_residues >= without.n_cationic_residues
    assert with_his.n_bridges >= without.n_bridges


def test_counting_is_deterministic(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    args = (model, (chain_a, chain_b), config.salt_bridge, config.structure)
    assert count_salt_bridges(*args).n_bridges == count_salt_bridges(*args).n_bridges


# ---------------------------------------------------------------------------
# Cross-chain scoping
# ---------------------------------------------------------------------------


def test_cross_chain_is_a_subset_of_all(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    every = count_salt_bridges(
        model, (chain_a, chain_b), config.salt_bridge, config.structure, cross_chain_only=False
    )
    across = count_salt_bridges(
        model, (chain_a, chain_b), config.salt_bridge, config.structure, cross_chain_only=True
    )
    assert across.n_bridges <= every.n_bridges
    assert across.scope == "cross_chain"


def test_single_chain_has_no_cross_chain_bridges(model, structure_case, config) -> None:
    _, chain_a, _ = structure_case
    result = count_salt_bridges(
        model, (chain_a,), config.salt_bridge, config.structure, cross_chain_only=True
    )
    assert result.n_bridges == 0


def test_intra_chain_bridges_dominate(model, structure_case, config) -> None:
    """Most salt bridges in a complex are within a chain, not across the interface.

    This is why an interface claim has to be scoped to cross-chain pairs: a
    whole-complex count is swamped by intra-chain bridges that have nothing to
    do with binding.
    """
    _, chain_a, chain_b = structure_case
    every = count_salt_bridges(
        model, (chain_a, chain_b), config.salt_bridge, config.structure, cross_chain_only=False
    )
    across = count_salt_bridges(
        model, (chain_a, chain_b), config.salt_bridge, config.structure, cross_chain_only=True
    )
    assert across.n_bridges < every.n_bridges / 2


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalisations_are_bounded(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    result = count_salt_bridges(model, (chain_a, chain_b), config.salt_bridge, config.structure)
    assert 0.0 <= result.per_opportunity <= 1.0
    assert 0.0 <= result.fraction_charged_residues_engaged <= 1.0
    assert result.per_charged_residue >= 0.0


def test_opportunity_count_is_the_product_of_the_two_populations(
    model, structure_case, config
) -> None:
    _, chain_a, chain_b = structure_case
    result = count_salt_bridges(model, (chain_a, chain_b), config.salt_bridge, config.structure)
    assert result.n_opportunities == result.n_cationic_residues * result.n_anionic_residues
    assert result.n_charged_residues == result.n_cationic_residues + result.n_anionic_residues


def test_bridges_never_exceed_opportunities(model, structure_case, config) -> None:
    """Each residue pair is counted once, so the count is bounded by the pair count."""
    _, chain_a, chain_b = structure_case
    for definition in ("geometric", "cbeta_proxy"):
        result = count_salt_bridges(
            model, (chain_a, chain_b), config.salt_bridge, config.structure, definition
        )
        assert result.n_bridges <= result.n_opportunities


def test_engaged_residues_never_exceed_charged_residues(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    result = count_salt_bridges(model, (chain_a, chain_b), config.salt_bridge, config.structure)
    assert result.n_engaged_residues <= result.n_charged_residues


def test_normalised_columns_carry_the_definition(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    row = count_salt_bridges(
        model, (chain_a, chain_b), config.salt_bridge, config.structure, "geometric"
    ).to_row()
    assert "sb_n_bridges_geometric" in row
    assert "sb_per_opportunity_geometric" in row


# ---------------------------------------------------------------------------
# Composition guard
# ---------------------------------------------------------------------------


def test_charged_composition_is_hand_computable() -> None:
    from interface_charge.structures import ChainExtract

    extract = ChainExtract(
        chain_id="A",
        residues=(),
        sequence="KKRRDDEEAAA",  # four cationic, four anionic, three neutral
        index_of={},
        substitutions=(),
        skipped_heteroatoms=0,
    )
    counts = charged_composition(extract)
    assert counts["n_cationic"] == 4
    assert counts["n_anionic"] == 4
    assert counts["n_charged"] == 8
    assert counts["n_opportunities"] == 16
    assert counts["n_residues"] == 11


def test_histidine_counts_only_when_requested() -> None:
    from interface_charge.structures import ChainExtract

    extract = ChainExtract(
        chain_id="A",
        residues=(),
        sequence="KHHD",
        index_of={},
        substitutions=(),
        skipped_heteroatoms=0,
    )
    assert charged_composition(extract, include_histidine=False)["n_cationic"] == 1
    assert charged_composition(extract, include_histidine=True)["n_cationic"] == 3


def test_matched_composition_passes() -> None:
    require_matched_composition({"n_charged": 40}, {"n_charged": 38})  # 5 percent apart


def test_mismatched_composition_raises_with_the_reason() -> None:
    """The guard exists because the failure mode produces a number, not an error.

    A design carrying half again as much charge as its native will show more
    salt bridges regardless of how well they are placed. Comparing the raw
    counts would read as a real effect.
    """
    with pytest.raises(StructureError) as info:
        require_matched_composition({"n_charged": 60}, {"n_charged": 40})
    message = str(info.value)
    assert "not comparable" in message
    assert "per_opportunity" in message


def test_zero_charge_comparison_raises() -> None:
    with pytest.raises(StructureError, match="no charge"):
        require_matched_composition({"n_charged": 0}, {"n_charged": 10})


def test_summarise_reports_both_raw_and_normalised(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    results = [
        count_salt_bridges(model, (chain_a,), config.salt_bridge, config.structure),
        count_salt_bridges(model, (chain_b,), config.salt_bridge, config.structure),
    ]
    stats = summarise(results)
    assert stats["n"] == 2
    assert stats["mean_raw"] > 0
    assert "mean_per_opportunity" in stats


def test_summarise_of_nothing_is_empty_not_an_error() -> None:
    assert summarise([]) == {}


# ---------------------------------------------------------------------------
# Threaded designs
# ---------------------------------------------------------------------------


def test_threaded_counting_reproduces_the_native_sequence(model, extracts, config) -> None:
    """Threading a chain's own sequence must reproduce its Cbeta-proxy count."""
    from interface_charge.saltbridge import count_salt_bridges_threaded

    extract_a, extract_b = extracts
    threaded = count_salt_bridges_threaded(
        model, extract_a, extract_b, config.salt_bridge, config.structure
    )
    direct = count_salt_bridges(
        model,
        (extract_a.chain_id, extract_b.chain_id),
        config.salt_bridge,
        config.structure,
        "cbeta_proxy",
    )
    assert threaded.n_charged_residues == direct.n_charged_residues
    assert threaded.n_bridges == direct.n_bridges


def test_threading_a_neutral_sequence_removes_every_bridge(model, extracts, config) -> None:
    from interface_charge.saltbridge import count_salt_bridges_threaded

    extract_a, extract_b = extracts
    result = count_salt_bridges_threaded(
        model,
        extract_a.with_sequence("A" * len(extract_a.sequence)),
        extract_b.with_sequence("A" * len(extract_b.sequence)),
        config.salt_bridge,
        config.structure,
    )
    assert result.n_charged_residues == 0
    assert result.n_bridges == 0


def test_threading_more_charge_raises_the_raw_count(model, extracts, config) -> None:
    """The confound this module exists to expose, demonstrated directly.

    Making a sequence uniformly charged raises the raw bridge count without any
    claim that the charge is better placed, which is why a raw count cannot
    support a statement about placement.
    """
    from interface_charge.saltbridge import count_salt_bridges_threaded

    extract_a, extract_b = extracts
    alternating = "".join("K" if i % 2 else "D" for i in range(len(extract_a.sequence)))
    charged = count_salt_bridges_threaded(
        model,
        extract_a.with_sequence(alternating),
        extract_b,
        config.salt_bridge,
        config.structure,
    )
    native = count_salt_bridges_threaded(
        model, extract_a, extract_b, config.salt_bridge, config.structure
    )
    assert charged.n_charged_residues > native.n_charged_residues
    assert charged.n_bridges > native.n_bridges


def test_threaded_cross_chain_is_a_subset(model, extracts, config) -> None:
    from interface_charge.saltbridge import count_salt_bridges_threaded

    extract_a, extract_b = extracts
    every = count_salt_bridges_threaded(
        model, extract_a, extract_b, config.salt_bridge, config.structure, cross_chain_only=False
    )
    across = count_salt_bridges_threaded(
        model, extract_a, extract_b, config.salt_bridge, config.structure, cross_chain_only=True
    )
    assert across.n_bridges <= every.n_bridges
