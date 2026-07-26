"""Bootstrap intervals and window membership, against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from interface_charge.statistics import (
    best_of_n_bias_warning,
    bootstrap_ci,
    fraction_in_window,
    paired_bootstrap_ci,
    window_membership_disagreement,
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_interval_brackets_the_point_estimate() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    interval = bootstrap_ci(values)
    assert interval.low <= interval.estimate <= interval.high
    assert interval.estimate == pytest.approx(4.5)


def test_constant_sample_gives_a_zero_width_interval() -> None:
    """Hand-computed: every resample of a constant sample has the same mean."""
    interval = bootstrap_ci([3.0] * 20)
    assert interval.estimate == 3.0
    assert interval.low == 3.0
    assert interval.high == 3.0


def test_interval_is_reproducible_for_a_given_seed() -> None:
    values = list(np.linspace(0, 10, 40))
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_different_seeds_give_different_intervals() -> None:
    values = list(np.linspace(0, 10, 40))
    assert bootstrap_ci(values, seed=1).low != bootstrap_ci(values, seed=2).low


def test_larger_samples_give_tighter_intervals() -> None:
    rng = np.random.default_rng(0)
    narrow = bootstrap_ci(rng.normal(size=2000).tolist(), seed=0)
    wide = bootstrap_ci(rng.normal(size=20).tolist(), seed=0)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_higher_confidence_gives_a_wider_interval() -> None:
    values = list(np.linspace(0, 10, 60))
    tight = bootstrap_ci(values, confidence=0.50, seed=0)
    loose = bootstrap_ci(values, confidence=0.99, seed=0)
    assert (loose.high - loose.low) > (tight.high - tight.low)


def test_median_statistic_is_supported() -> None:
    interval = bootstrap_ci([1.0, 2.0, 3.0, 100.0], statistic=np.median)
    assert interval.estimate == pytest.approx(2.5)


def test_empty_sample_raises() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_ci([])


def test_invalid_confidence_raises() -> None:
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_ci([1.0, 2.0], confidence=bad)


def test_excludes_zero_flag() -> None:
    assert bootstrap_ci([5.0] * 30).excludes_zero
    assert not bootstrap_ci([-1.0, 1.0] * 30, seed=0).excludes_zero


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def test_paired_difference_of_identical_samples_is_zero() -> None:
    values = [1.0, 5.0, 9.0, 2.0]
    interval = paired_bootstrap_ci(values, values)
    assert interval.estimate == 0.0
    assert interval.low == 0.0 and interval.high == 0.0


def test_paired_difference_detects_a_constant_offset() -> None:
    """Hand-computed: adding two to every element gives a paired difference of two."""
    left = [1.0, 5.0, 9.0, 2.0, 7.0]
    right = [value - 2.0 for value in left]
    interval = paired_bootstrap_ci(left, right)
    assert interval.estimate == pytest.approx(2.0)
    assert interval.low == pytest.approx(2.0)
    assert interval.excludes_zero


def test_pairing_survives_large_between_subject_variance() -> None:
    """The reason to pair: a small consistent effect under huge subject spread.

    Unpaired, the effect is invisible. Paired, it is unambiguous. This is
    exactly the design-versus-native situation, where complexes differ
    enormously and the treatment effect is small.
    """
    rng = np.random.default_rng(0)
    subject = rng.normal(0, 100, size=40)
    left = subject + 1.0
    right = subject

    paired = paired_bootstrap_ci(left.tolist(), right.tolist(), seed=0)
    assert paired.estimate == pytest.approx(1.0)
    assert paired.excludes_zero

    unpaired_left = bootstrap_ci(left.tolist(), seed=0)
    unpaired_right = bootstrap_ci(right.tolist(), seed=0)
    # The unpaired intervals overlap heavily, so the effect would be missed.
    assert unpaired_left.low < unpaired_right.high


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_ci([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Window membership
# ---------------------------------------------------------------------------


def test_fraction_inside_is_hand_computable() -> None:
    """Three of six values lie within [-4, 4], so the fraction is exactly one half."""
    achieved = [-6.0, -2.0, 0.0, 3.0, 5.0, 9.0]
    summary = fraction_in_window(achieved, -4.0, 4.0)
    assert summary.n == 6
    assert summary.n_inside == 3
    assert summary.fraction_inside == pytest.approx(0.5)


def test_window_boundaries_are_inclusive() -> None:
    summary = fraction_in_window([-4.0, 4.0], -4.0, 4.0)
    assert summary.n_inside == 2


def test_mae_exceeds_window_flag() -> None:
    """The case the flag exists for: an average miss wider than the whole window.

    A window of eight units and a mean absolute error near seven means the
    average design misses by almost the width of the target, so quoting the
    error as evidence the window can be hit is misleading.
    """
    achieved = [-9.0, 9.0, -8.0, 8.0]
    summary = fraction_in_window(achieved, -4.0, 4.0, target=0.0)
    assert summary.mean_absolute_error == pytest.approx(8.5)
    assert summary.mae_exceeds_window()
    assert summary.fraction_inside == 0.0


def test_small_error_inside_a_wide_window_does_not_flag() -> None:
    summary = fraction_in_window([0.5, -0.5, 1.0], -4.0, 4.0, target=0.0)
    assert not summary.mae_exceeds_window()
    assert summary.fraction_inside == 1.0


def test_per_design_targets_are_supported() -> None:
    summary = fraction_in_window([1.0, 2.0, 3.0], -4.0, 4.0, target=[0.0, 2.0, 6.0])
    # Absolute errors are 1, 0 and 3, so the mean is 4/3.
    assert summary.mean_absolute_error == pytest.approx(4.0 / 3.0)


def test_inverted_window_raises() -> None:
    with pytest.raises(ValueError, match="window_low must be below"):
        fraction_in_window([1.0], 4.0, -4.0)


def test_mismatched_target_length_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        fraction_in_window([1.0, 2.0], -4.0, 4.0, target=[1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Definition disagreement
# ---------------------------------------------------------------------------


def test_identical_definitions_never_disagree() -> None:
    values = [-6.0, -1.0, 0.0, 2.0, 7.0]
    report = window_membership_disagreement(values, values, -4.0, 4.0)
    assert report["n_disagree"] == 0
    assert report["fraction_disagree"] == 0.0
    assert "does not affect any conclusion" in report["interpretation"]


def test_disagreement_is_counted_and_attributed() -> None:
    """Hand-computed: two designs straddle the boundary in opposite directions."""
    simple = [3.0, 5.0, 0.0]  # inside, outside, inside
    ph = [5.0, 3.0, 0.0]  # outside, inside, inside
    report = window_membership_disagreement(simple, ph, -4.0, 4.0)
    assert report["n_disagree"] == 2
    assert report["n_only_inside_simple"] == 1
    assert report["n_only_inside_ph"] == 1
    assert "change window membership" in report["interpretation"]


def test_constant_offset_is_detected() -> None:
    simple = [1.0, 2.0, 3.0]
    ph = [0.0, 1.0, 2.0]
    report = window_membership_disagreement(simple, ph, -4.0, 4.0)
    assert report["mean_offset"] == pytest.approx(1.0)
    assert report["offset_is_constant"] is True


def test_variable_offset_is_flagged_as_not_constant() -> None:
    """The important case: the gap between definitions is not a fixed shift.

    If it were constant it could be corrected with one subtraction. It is not,
    which is why the two definitions cannot be interconverted after the fact.
    """
    simple = [1.0, 2.0, 3.0]
    ph = [0.4, 1.9, 2.1]
    report = window_membership_disagreement(simple, ph, -4.0, 4.0)
    assert report["offset_is_constant"] is False


def test_mismatched_definition_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same designs"):
        window_membership_disagreement([1.0, 2.0], [1.0], -4.0, 4.0)


# ---------------------------------------------------------------------------
# Best-of-N
# ---------------------------------------------------------------------------


def test_equal_sample_sizes_produce_no_warning() -> None:
    assert best_of_n_bias_warning([16, 16, 16]) is None


def test_unequal_sample_sizes_produce_a_warning() -> None:
    message = best_of_n_bias_warning([28, 24, 16, 16, 16])
    assert message is not None
    assert "16" in message and "28" in message
    assert "expected maximum grows with N" in message


def test_best_of_n_bias_is_real_not_theoretical() -> None:
    """Demonstrate the bias the warning describes, on identical distributions.

    Both groups are drawn from the same distribution, so any difference in the
    best value is purely a sample-size artefact.
    """
    rng = np.random.default_rng(0)
    best_large = np.mean([rng.normal(size=28).max() for _ in range(400)])
    best_small = np.mean([rng.normal(size=16).max() for _ in range(400)])
    assert best_large > best_small
