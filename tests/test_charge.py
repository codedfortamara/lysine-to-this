"""The two charge definitions, checked against values worked out by hand.

The pH-based tests lean on a property that makes exact hand calculation
possible: when the pH equals a group's pKa, Henderson-Hasselbalch gives that
group exactly half a charge. Those anchors are exact rational numbers, not
values copied out of a previous run, so they would catch a wrong pKa, a sign
error, or a flipped inequality. The general case is then checked against an
independent re-derivation of the formula written out in the test with literal
constants.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from interface_charge.charge import (
    ChargePartition,
    SequenceError,
    net_charge_ph,
    net_charge_simple,
    partition_charge,
    per_residue_charge_ph,
    per_residue_charge_simple,
    validate_sequence,
)
from interface_charge.config import PKA_BJELLQVIST, PKA_EMBOSS

# ---------------------------------------------------------------------------
# Definition one: the integer count the controller targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        # Worked by hand: count K and R, subtract D and E.
        ("KRDE", 0),  # +2 - 2
        ("KKKK", 4),
        ("DDEE", -4),
        ("ACFGILMNPQSTVWY", 0),  # no ionisable side chains under this definition
        ("H", 0),  # histidine is deliberately ignored by the simple definition
        ("KRKRDEDE", 0),  # +4 - 4
        ("KRRKK", 5),
        ("DEDED", -5),
        ("MKVLAAGIVGLLLLLAAYCYA", 1),  # one K, nothing else ionisable
    ],
)
def test_net_charge_simple_hand_computed(sequence: str, expected: int) -> None:
    assert net_charge_simple(sequence) == expected


def test_net_charge_simple_returns_int_not_float() -> None:
    """The simple definition is a count and must stay an integer.

    If it ever returns a float it will silently start being averaged with the
    pH-based definition somewhere downstream.
    """
    assert isinstance(net_charge_simple("KRDE"), int)


def test_net_charge_simple_is_additive_over_concatenation() -> None:
    left, right = "KKRD", "EEDDR"
    assert net_charge_simple(left + right) == net_charge_simple(left) + net_charge_simple(right)


# ---------------------------------------------------------------------------
# Definition two: Henderson-Hasselbalch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("residue", "expected_sign"),
    [("K", 1.0), ("R", 1.0), ("H", 1.0), ("D", -1.0), ("E", -1.0), ("C", -1.0), ("Y", -1.0)],
)
def test_at_ph_equal_to_pka_the_group_carries_exactly_half_a_charge(
    residue: str, expected_sign: float
) -> None:
    """Exact hand-computed anchor: 1/(1 + 10^0) = 1/2, precisely.

    Termini are excluded so the side chain is the only contributor. A wrong pKa
    in the table would move this off 0.5 immediately.
    """
    pka = PKA_EMBOSS[residue]
    value = net_charge_ph(residue, ph=pka, pka_set="emboss", include_termini=False)
    assert value == pytest.approx(0.5 * expected_sign, abs=1e-12)


def test_terminus_anchors_at_their_own_pka() -> None:
    """The N-terminus at its pKa contributes exactly +0.5, the C-terminus -0.5."""
    # A glycine has no ionisable side chain, so only the termini contribute.
    n_only = net_charge_ph("G", ph=PKA_EMBOSS["Nterm"], pka_set="emboss")
    c_term_at_n_pka = -1.0 / (1.0 + 10.0 ** (PKA_EMBOSS["Cterm"] - PKA_EMBOSS["Nterm"]))
    assert n_only == pytest.approx(0.5 + c_term_at_n_pka, abs=1e-12)


def test_sequence_without_ionisable_groups_or_termini_is_exactly_zero() -> None:
    """Hand-computed: no K, R, H, D, E, C or Y, and no termini, so exactly zero."""
    assert net_charge_ph("AAAGGGVVV", include_termini=False) == 0.0


def test_ph_definition_matches_independent_rederivation() -> None:
    """Re-derive the formula in the test with literal constants.

    This is deliberately a separate implementation rather than a call into the
    module's own helpers, so that an error in the module cannot be mirrored in
    the expectation.
    """
    sequence = "MKHRDECYGA"
    ph = 7.4
    pka = PKA_EMBOSS

    expected = 0.0
    for residue in sequence:
        if residue in ("K", "R", "H"):
            expected += 1.0 / (1.0 + 10.0 ** (ph - pka[residue]))
        elif residue in ("D", "E", "C", "Y"):
            expected -= 1.0 / (1.0 + 10.0 ** (pka[residue] - ph))
    expected += 1.0 / (1.0 + 10.0 ** (ph - pka["Nterm"]))
    expected -= 1.0 / (1.0 + 10.0 ** (pka["Cterm"] - ph))

    assert net_charge_ph(sequence, ph=ph, pka_set="emboss") == pytest.approx(expected, abs=1e-12)


def test_the_two_definitions_genuinely_disagree() -> None:
    """Guard against the two definitions being accidentally wired together.

    Histidine separates them by construction: the simple definition ignores it
    entirely, the pH-based one counts it at about a tenth of a charge each at
    pH 7.4. If a refactor ever made one delegate to the other, this fails.
    """
    sequence = "H" * 20
    assert net_charge_simple(sequence) == 0
    assert net_charge_ph(sequence, ph=7.4) > 1.0


def test_the_two_definitions_can_coincide_by_accident() -> None:
    """A worked example where the two definitions land within 0.21 of each other.

    This is not a property of the definitions, it is a coincidence of this
    sequence, and it is recorded here because it is the trap: a spot check on
    one sequence can make the two look interchangeable when they are not. That
    is the argument for keeping them in separately named columns rather than
    relying on anyone noticing a discrepancy.
    """
    sequence = "MKHHHRDDEECYGAKKRRDE"
    assert net_charge_simple(sequence) == 0
    assert abs(net_charge_ph(sequence, ph=7.4)) < 0.25


def test_pka_sets_give_different_answers() -> None:
    """EMBOSS and Bjellqvist must not silently be the same table."""
    sequence = "MKHRDECYGAKKRR"
    emboss = net_charge_ph(sequence, pka_set="emboss")
    bjellqvist = net_charge_ph(sequence, pka_set="bjellqvist")
    assert emboss != bjellqvist
    assert PKA_EMBOSS != PKA_BJELLQVIST


def test_charge_is_monotonically_decreasing_in_ph() -> None:
    """A titration curve only ever goes down as pH rises."""
    sequence = "MKHRDECYGAKKRR"
    values = [net_charge_ph(sequence, ph=ph) for ph in (2.0, 4.0, 6.0, 7.4, 9.0, 11.0, 13.0)]
    assert all(later <= earlier for earlier, later in pairwise(values))


def test_termini_can_be_switched_off() -> None:
    sequence = "AAAA"
    with_termini = net_charge_ph(sequence, include_termini=True)
    without = net_charge_ph(sequence, include_termini=False)
    assert without == 0.0
    assert with_termini != 0.0


def test_cys_tyr_can_be_switched_off() -> None:
    sequence = "CYCY"
    assert net_charge_ph(sequence, include_cys_tyr=False, include_termini=False) == 0.0
    assert net_charge_ph(sequence, include_cys_tyr=True, include_termini=False) < 0.0


def test_unknown_pka_set_raises() -> None:
    with pytest.raises(ValueError, match="unknown pka_set"):
        net_charge_ph("AAAA", pka_set="not_a_real_set")


# ---------------------------------------------------------------------------
# Sequence validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["ACDX", "ACD-", "acd*", "ACDB", "ACDZ", "ACD U"])
def test_non_standard_residues_raise(bad: str) -> None:
    """An X treated as neutral would shift a net charge invisibly."""
    with pytest.raises(SequenceError):
        validate_sequence(bad)


def test_empty_sequence_raises() -> None:
    with pytest.raises(SequenceError, match="empty"):
        validate_sequence("")


def test_non_string_sequence_raises() -> None:
    with pytest.raises(SequenceError):
        validate_sequence(None)  # type: ignore[arg-type]


def test_error_message_names_the_offending_positions() -> None:
    with pytest.raises(SequenceError) as info:
        validate_sequence("AAXAA")
    assert "X" in str(info.value)
    assert "2" in str(info.value)


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def test_partitions_sum_to_the_whole_sequence_total_simple() -> None:
    sequence = "MKHRDECYGAKKRRDDEEHH"
    partition = partition_charge(sequence, interface_idx={0, 1, 2, 3}, core_idx={10, 11, 12})
    assert (
        partition.interface_simple + partition.surface_simple + partition.core_simple
        == partition.total_simple
    )
    assert partition.total_simple == net_charge_simple(sequence)


def test_partitions_sum_to_the_whole_sequence_total_ph() -> None:
    """The pH sum must be exact too, which is why the termini are folded in.

    If the termini were held outside the partitions this identity would fail by
    close to one charge unit at each end, and every partition table would carry
    an unexplained residual.
    """
    sequence = "MKHRDECYGAKKRRDDEEHH"
    partition = partition_charge(sequence, interface_idx={0, 1, 2, 3}, core_idx={10, 11, 12})
    total = partition.interface_ph + partition.surface_ph + partition.core_ph
    assert total == pytest.approx(partition.total_ph, abs=1e-12)
    assert partition.total_ph == pytest.approx(net_charge_ph(sequence), abs=1e-12)


def test_partition_counts_sum_to_sequence_length() -> None:
    sequence = "MKHRDECYGAKKRRDDEEHH"
    partition = partition_charge(sequence, interface_idx={0, 1}, core_idx={5, 6, 7})
    assert partition.n_interface + partition.n_surface + partition.n_core == partition.n_total
    assert partition.n_total == len(sequence)


def test_partition_with_termini_split_across_regions_still_sums() -> None:
    """The N-terminal and C-terminal residues in different partitions."""
    sequence = "KKKDDDEEE"
    partition = partition_charge(sequence, interface_idx={0}, core_idx={len(sequence) - 1})
    assert (partition.interface_ph + partition.surface_ph + partition.core_ph) == pytest.approx(
        partition.total_ph, abs=1e-12
    )


def test_empty_interface_is_allowed_and_sums() -> None:
    """A complex with no interface residues is degenerate but must not crash."""
    sequence = "KKKDDD"
    partition = partition_charge(sequence, interface_idx=set(), core_idx=set())
    assert partition.n_interface == 0
    assert partition.interface_simple == 0
    assert partition.surface_simple == partition.total_simple


def test_overlapping_interface_and_core_raises() -> None:
    with pytest.raises(ValueError, match="overlap"):
        partition_charge("KKKDDD", interface_idx={1, 2}, core_idx={2, 3})


def test_out_of_range_index_raises() -> None:
    with pytest.raises(IndexError, match="outside the sequence"):
        partition_charge("KKKDDD", interface_idx={99}, core_idx=set())


def test_negative_index_raises() -> None:
    """A negative index would silently wrap round to the other end of the chain."""
    with pytest.raises(IndexError):
        partition_charge("KKKDDD", interface_idx={-1}, core_idx=set())


def test_to_row_keeps_the_two_definitions_in_separate_columns() -> None:
    partition = partition_charge("MKHRDECYGA", interface_idx={0}, core_idx={9})
    row = partition.to_row()
    simple_columns = [c for c in row if c.endswith("_simple")]
    ph_columns = [c for c in row if "ph7.4" in c]
    assert len(simple_columns) == 4
    assert len(ph_columns) == 4
    assert not set(simple_columns) & set(ph_columns)
    # The pH columns must name the pKa set, so two runs under different sets
    # cannot be concatenated into one column.
    assert all("emboss" in c for c in ph_columns)


def test_charge_density_divides_by_partition_size() -> None:
    partition = partition_charge("KKKKDDDD", interface_idx={0, 1}, core_idx={6, 7})
    densities = partition.charge_density_simple()
    assert densities["interface"] == pytest.approx(1.0)
    assert densities["core"] == pytest.approx(-1.0)


def test_charge_density_of_empty_partition_is_zero_not_nan() -> None:
    partition = partition_charge("KKKK", interface_idx=set(), core_idx=set())
    densities = partition.charge_density_simple()
    assert densities["interface"] == 0.0
    assert not math.isnan(densities["core"])


def test_partition_is_frozen() -> None:
    partition = partition_charge("KKKK", interface_idx={0}, core_idx=set())
    assert isinstance(partition, ChargePartition)
    with pytest.raises((AttributeError, TypeError)):
        partition.interface_simple = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Per-residue arrays
# ---------------------------------------------------------------------------


def test_per_residue_simple_matches_the_total() -> None:
    sequence = "MKHRDECYGAKKRR"
    assert int(per_residue_charge_simple(sequence).sum()) == net_charge_simple(sequence)


def test_per_residue_ph_matches_the_total() -> None:
    sequence = "MKHRDECYGAKKRR"
    assert float(per_residue_charge_ph(sequence).sum()) == pytest.approx(
        net_charge_ph(sequence), abs=1e-12
    )


def test_per_residue_simple_places_charges_on_the_right_residues() -> None:
    charges = per_residue_charge_simple("KADE R".replace(" ", ""))
    assert list(charges) == [1, 0, -1, -1, 1]
