"""Charge-hydropathy placement.

The assertions are properties of the published scale and boundary, or invariants
of the placement, so they hold without reference to any particular run.
"""

from __future__ import annotations

import pytest

from interface_charge.uversky import (
    BOUNDARY_INTERCEPT,
    BOUNDARY_SLOPE,
    KYTE_DOOLITTLE,
    boundary_charge,
    mean_net_charge,
    mean_scaled_hydropathy,
    place_on_diagram,
)

# ---------------------------------------------------------------------------
# The scale
# ---------------------------------------------------------------------------


def test_every_standard_amino_acid_has_a_hydropathy() -> None:
    assert set(KYTE_DOOLITTLE) == set("ACDEFGHIKLMNPQRSTVWY")


def test_the_scale_extremes_map_to_the_ends_of_the_unit_interval() -> None:
    """Isoleucine is the most hydrophobic residue on this scale, arginine the least.

    Uversky's normalisation puts them at exactly one and exactly zero, which is
    the check that the shift and divisor match the scale being used.
    """
    assert mean_scaled_hydropathy("I") == pytest.approx(1.0)
    assert mean_scaled_hydropathy("R") == pytest.approx(0.0)


def test_scaled_hydropathy_stays_inside_the_unit_interval() -> None:
    for aa in KYTE_DOOLITTLE:
        assert 0.0 <= mean_scaled_hydropathy(aa) <= 1.0


def test_hydropathy_is_a_mean_not_a_sum() -> None:
    """Doubling a sequence must not move it on the diagram."""
    assert mean_scaled_hydropathy("ACDEFG") == pytest.approx(mean_scaled_hydropathy("ACDEFGACDEFG"))


def test_an_unknown_residue_raises() -> None:
    with pytest.raises(ValueError):
        mean_scaled_hydropathy("ACDZ")


# ---------------------------------------------------------------------------
# The charge axis
# ---------------------------------------------------------------------------


def test_charge_axis_is_absolute() -> None:
    """The published axis is mean *absolute* net charge, so sign is discarded.

    This is the limitation that has to be reported alongside any use of the
    diagram, and it is worth pinning in a test so nobody quietly "fixes" it into
    a signed quantity and changes what the boundary means.
    """
    assert mean_net_charge("KKKKA") == mean_net_charge("DDDDA")


def test_charge_axis_is_per_residue() -> None:
    assert mean_net_charge("KA") == pytest.approx(0.5)
    assert mean_net_charge("KAAA") == pytest.approx(0.25)


def test_a_balanced_sequence_sits_on_the_charge_floor() -> None:
    assert mean_net_charge("KDRE") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


def test_boundary_is_the_published_line() -> None:
    assert boundary_charge(0.0) == pytest.approx(BOUNDARY_INTERCEPT)
    assert boundary_charge(1.0) == pytest.approx(BOUNDARY_SLOPE + BOUNDARY_INTERCEPT)


def test_boundary_rises_with_hydropathy() -> None:
    """A more hydrophobic sequence tolerates more charge before it is called disordered."""
    assert boundary_charge(0.6) > boundary_charge(0.3)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def test_a_polyalanine_sequence_is_predicted_globular() -> None:
    """No charge at moderate hydrophobicity sits well below the line."""
    point = place_on_diagram("A" * 40)
    assert point.mean_net_charge == 0.0
    assert not point.predicted_disordered
    assert point.distance_from_boundary < 0.0


def test_a_polylysine_sequence_is_predicted_disordered() -> None:
    """Maximum charge at minimum hydrophobicity is the corner of the diagram."""
    point = place_on_diagram("K" * 40)
    assert point.mean_net_charge == pytest.approx(1.0)
    assert point.predicted_disordered
    assert point.distance_from_boundary > 0.0


def test_distance_is_signed_and_agrees_with_the_flag() -> None:
    for sequence in ("A" * 30, "K" * 30, "ACDEFGHIKLMNPQRSTVWY"):
        point = place_on_diagram(sequence)
        assert point.predicted_disordered == (point.distance_from_boundary > 0.0)


def test_adding_charge_moves_a_sequence_towards_the_boundary() -> None:
    """The monotonicity the whole argument rests on.

    Biasing a sequence towards lysine raises its charge and lowers its
    hydropathy at once, so it must move towards and then across the line. If
    this did not hold, the diagram could not explain a charge-driven collapse.
    """
    base = "ACDEFGHIKLMNPQRSTVWY" * 2
    biased = base.replace("A", "K").replace("S", "K").replace("T", "K")
    assert place_on_diagram(biased).distance_from_boundary > (
        place_on_diagram(base).distance_from_boundary
    )


def test_placement_records_the_signed_charge_the_axis_discards() -> None:
    """The row keeps the sign even though the diagram cannot use it.

    Without this the sign asymmetry seen elsewhere in the project could not be
    reported against the placement at all.
    """
    positive = place_on_diagram("K" * 10 + "A" * 30)
    negative = place_on_diagram("D" * 10 + "A" * 30)
    assert positive.signed_net_charge == 10
    assert negative.signed_net_charge == -10
    assert positive.mean_net_charge == negative.mean_net_charge


def test_row_round_trips_the_properties() -> None:
    point = place_on_diagram("ACDEFGHIKLMNPQRSTVWY")
    row = point.to_row()
    assert row["n_residues"] == 20
    assert row["uversky_boundary"] == pytest.approx(point.boundary)
    assert row["distance_from_boundary"] == pytest.approx(point.distance_from_boundary)
    assert row["predicted_disordered"] is point.predicted_disordered
