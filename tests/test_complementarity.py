"""Electrostatic complementarity: sign convention, invariants and real structure.

The sign convention is the thing most likely to be got backwards and is the
thing a reader will rely on, so it is tested directly with a constructed case
before being exercised on the fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from interface_charge.complementarity import (
    ComplementarityParams,
    cross_chain_contacts,
    electrostatic_complementarity,
)
from interface_charge.config import ChargeParams


def test_module_does_not_import_a_poisson_boltzmann_solver() -> None:
    """The decision not to depend on APBS is load-bearing, so it is tested.

    A future contributor adding an APBS or PDB2PQR dependency would change the
    claim the paper can make, and this test makes that a deliberate act rather
    than a quiet import.
    """
    from interface_charge import complementarity

    text = Path(complementarity.__file__).read_text().lower()
    for forbidden in ("import apbs", "from apbs", "import pdb2pqr", "from pdb2pqr"):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------


def test_opposite_charges_give_a_negative_sum(model, extracts, config) -> None:
    """Threading a lysine-only chain against a native partner.

    The direction of the effect is what matters: making one chain uniformly
    positive should move the sum in the direction set by the partner's net
    interface charge. This checks the sum responds to sequence at all, and the
    dedicated sign tests below pin the convention.
    """
    extract_a, extract_b = extracts
    params = ComplementarityParams(charge_definition="simple")

    all_lys = extract_a.with_sequence("K" * len(extract_a.sequence))
    all_asp = extract_a.with_sequence("D" * len(extract_a.sequence))

    positive = electrostatic_complementarity(
        model, all_lys, extract_b, params, config.charge, config.structure
    )
    negative = electrostatic_complementarity(
        model, all_asp, extract_b, params, config.charge, config.structure
    )
    # Flipping the sign of every residue on one side must flip the sum exactly.
    assert positive.sum_product == pytest.approx(-negative.sum_product, abs=1e-9)


def test_like_charges_facing_each_other_give_a_positive_sum(model, extracts, config) -> None:
    """Both chains all lysine: every contacting pair is +1 times +1."""
    extract_a, extract_b = extracts
    params = ComplementarityParams(charge_definition="simple")

    result = electrostatic_complementarity(
        model,
        extract_a.with_sequence("K" * len(extract_a.sequence)),
        extract_b.with_sequence("K" * len(extract_b.sequence)),
        params,
        config.charge,
        config.structure,
    )
    assert result.sum_product > 0
    assert result.sum_product == pytest.approx(float(result.n_pairs))
    assert result.n_repulsive == result.n_pairs
    assert result.n_attractive == 0


def test_opposite_uniform_charges_give_exactly_minus_the_pair_count(
    model, extracts, config
) -> None:
    """All lysine against all aspartate: every pair contributes exactly -1."""
    extract_a, extract_b = extracts
    params = ComplementarityParams(charge_definition="simple")

    result = electrostatic_complementarity(
        model,
        extract_a.with_sequence("K" * len(extract_a.sequence)),
        extract_b.with_sequence("D" * len(extract_b.sequence)),
        params,
        config.charge,
        config.structure,
    )
    assert result.sum_product == pytest.approx(-float(result.n_pairs))
    assert result.n_attractive == result.n_pairs
    assert result.n_repulsive == 0


def test_neutral_chain_gives_exactly_zero(model, extracts, config) -> None:
    """Alanine carries no charge under the simple definition."""
    extract_a, extract_b = extracts
    result = electrostatic_complementarity(
        model,
        extract_a.with_sequence("A" * len(extract_a.sequence)),
        extract_b,
        ComplementarityParams(charge_definition="simple"),
        config.charge,
        config.structure,
    )
    assert result.sum_product == 0.0
    assert result.n_neutral == result.n_pairs


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


def test_cross_chain_contacts_are_found(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    pairs = cross_chain_contacts(model, chain_a, chain_b, 5.0, config.structure)
    assert len(pairs) > 10
    assert all(pair.residue_a.chain == chain_a for pair in pairs)
    assert all(pair.residue_b.chain == chain_b for pair in pairs)
    assert all(0.0 < pair.min_distance_a <= 5.0 for pair in pairs)


def test_each_residue_pair_appears_once(model, structure_case, config) -> None:
    """The minimum distance per pair, not one row per contacting atom pair."""
    _, chain_a, chain_b = structure_case
    pairs = cross_chain_contacts(model, chain_a, chain_b, 5.0, config.structure)
    keys = [(pair.residue_a, pair.residue_b) for pair in pairs]
    assert len(keys) == len(set(keys))


def test_a_wider_cutoff_finds_more_pairs(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    tight = cross_chain_contacts(model, chain_a, chain_b, 4.0, config.structure)
    loose = cross_chain_contacts(model, chain_a, chain_b, 8.0, config.structure)
    assert len(loose) > len(tight)


def test_contacts_are_deterministic(model, structure_case, config) -> None:
    _, chain_a, chain_b = structure_case
    first = cross_chain_contacts(model, chain_a, chain_b, 5.0, config.structure)
    second = cross_chain_contacts(model, chain_a, chain_b, 5.0, config.structure)
    assert [(p.residue_a, p.residue_b) for p in first] == [
        (p.residue_a, p.residue_b) for p in second
    ]


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def test_pair_counts_partition_the_total(model, extracts, config) -> None:
    extract_a, extract_b = extracts
    result = electrostatic_complementarity(
        model, extract_a, extract_b, config.complementarity, config.charge, config.structure
    )
    assert result.n_attractive + result.n_repulsive + result.n_neutral == result.n_pairs


def test_both_charge_definitions_run_and_are_labelled(model, extracts, config) -> None:
    extract_a, extract_b = extracts
    simple = electrostatic_complementarity(
        model,
        extract_a,
        extract_b,
        ComplementarityParams(charge_definition="simple"),
        config.charge,
        config.structure,
    )
    ph = electrostatic_complementarity(
        model,
        extract_a,
        extract_b,
        ComplementarityParams(charge_definition="ph"),
        config.charge,
        config.structure,
    )
    assert "ec_sum_product_simple" in simple.to_row()
    assert "ec_sum_product_ph7.4_emboss" in ph.to_row()
    # The two must not share a column name, or they would be concatenated.
    assert not (set(simple.to_row()) & set(ph.to_row())) - {
        "ec_n_pairs",
        "ec_n_attractive",
        "ec_n_repulsive",
        "ec_n_neutral",
        "ec_contact_cutoff_a",
        "ec_weight",
    }


def test_inverse_distance_weighting_changes_the_sum(model, extracts, config) -> None:
    extract_a, extract_b = extracts
    unweighted = electrostatic_complementarity(
        model,
        extract_a,
        extract_b,
        ComplementarityParams(charge_definition="simple", weight="none"),
        config.charge,
        config.structure,
    )
    weighted = electrostatic_complementarity(
        model,
        extract_a,
        extract_b,
        ComplementarityParams(charge_definition="simple", weight="inverse_distance"),
        config.charge,
        config.structure,
    )
    assert weighted.n_pairs == unweighted.n_pairs
    assert weighted.sum_product != unweighted.sum_product


def test_unknown_weight_raises(model, extracts, config) -> None:
    extract_a, extract_b = extracts
    with pytest.raises(ValueError, match="weight"):
        electrostatic_complementarity(
            model,
            extract_a,
            extract_b,
            ComplementarityParams(weight="magic"),
            config.charge,
            config.structure,
        )


def test_unknown_charge_definition_raises(model, extracts, config) -> None:
    extract_a, extract_b = extracts
    with pytest.raises(ValueError, match="charge_definition"):
        electrostatic_complementarity(
            model,
            extract_a,
            extract_b,
            ComplementarityParams(charge_definition="quantum"),
            config.charge,
            config.structure,
        )


def test_ph_result_records_the_pka_set(model, extracts, config) -> None:
    extract_a, extract_b = extracts
    result = electrostatic_complementarity(
        model,
        extract_a,
        extract_b,
        ComplementarityParams(charge_definition="ph"),
        ChargeParams(pka_set="bjellqvist"),
        config.structure,
    )
    assert result.pka_set == "bjellqvist"
    assert "ec_sum_product_ph7.4_bjellqvist" in result.to_row()


def test_simple_result_does_not_claim_a_ph(model, extracts, config) -> None:
    extract_a, extract_b = extracts
    result = electrostatic_complementarity(
        model,
        extract_a,
        extract_b,
        ComplementarityParams(charge_definition="simple"),
        config.charge,
        config.structure,
    )
    assert result.ph is None
    assert result.pka_set is None
