"""Uncertainty, and the difference between an average miss and a useful hit.

Two things live here.

**Bootstrap intervals.** At the sample sizes in play, between roughly sixteen
and sixty per cell, a point estimate on its own is not interpretable. The
bootstrap is used rather than an analytic standard error because none of the
quantities involved (a median RMSD, a best-of-N interface pTM, a bridges-per-
opportunity ratio) has a distribution anyone should assume.

**Window membership.** A mean absolute error answers "how far off is the
controller on average". It does not answer "does the design land in the range
that matters", and those come apart badly when the tolerated range is narrow
relative to the error. If a developability window spans eight charge units and
the mean absolute error is close to seven, then a good-looking MAE is
compatible with missing the window most of the time. :func:`fraction_in_window`
reports the quantity the application actually cares about, and
:func:`window_membership_disagreement` reports something easier to overlook:
how often the answer flips depending on which charge definition the window is
expressed in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "BootstrapInterval",
    "WindowSummary",
    "bootstrap_ci",
    "fraction_in_window",
    "paired_bootstrap_ci",
    "spearman",
    "spearman_bootstrap_ci",
    "spearman_permutation_p",
    "window_membership_disagreement",
]


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A point estimate with a percentile bootstrap interval."""

    estimate: float
    low: float
    high: float
    confidence: float
    n: int
    n_resamples: int
    seed: int

    def __str__(self) -> str:
        return f"{self.estimate:.3g} [{self.low:.3g}, {self.high:.3g}]"

    def to_row(self, prefix: str) -> dict[str, Any]:
        return {
            prefix: self.estimate,
            f"{prefix}_ci_low": self.low,
            f"{prefix}_ci_high": self.high,
            f"{prefix}_ci_confidence": self.confidence,
            f"{prefix}_n": self.n,
        }

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies wholly on one side of zero."""
        return (self.low > 0.0) or (self.high < 0.0)


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile bootstrap interval for any statistic of one sample.

    The seed is explicit and recorded on the result, because an interval that
    moves between runs is not something anyone should have to wonder about.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    replicates = np.array([statistic(array[row]) for row in indices], dtype=np.float64)

    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(statistic(array)),
        low=float(np.quantile(replicates, tail)),
        high=float(np.quantile(replicates, 1.0 - tail)),
        confidence=confidence,
        n=int(array.size),
        n_resamples=n_resamples,
        seed=seed,
    )


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Interval for the paired difference ``left - right``.

    Pairs are resampled together, which is what makes this valid for a
    design-versus-native comparison on the same structures: the between-protein
    variance is enormous next to the within-protein effect, and resampling the
    two sets independently would drown the effect in it.
    """
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must have equal length, got {a.shape} and {b.shape}")
    return bootstrap_ci(a - b, statistic, confidence, n_resamples, seed)


@dataclass(frozen=True, slots=True)
class WindowSummary:
    """How often a set of values lands inside a target window."""

    n: int
    n_inside: int
    fraction_inside: float
    window_low: float
    window_high: float
    mean_absolute_error: float
    median_absolute_error: float
    interval: BootstrapInterval

    def to_row(self, prefix: str = "window") -> dict[str, Any]:
        return {
            f"{prefix}_n": self.n,
            f"{prefix}_n_inside": self.n_inside,
            f"{prefix}_fraction_inside": self.fraction_inside,
            f"{prefix}_fraction_inside_ci_low": self.interval.low,
            f"{prefix}_fraction_inside_ci_high": self.interval.high,
            f"{prefix}_low": self.window_low,
            f"{prefix}_high": self.window_high,
            f"{prefix}_mean_absolute_error": self.mean_absolute_error,
            f"{prefix}_median_absolute_error": self.median_absolute_error,
        }

    def mae_exceeds_window(self) -> bool:
        """Whether the mean absolute error is wider than the window itself.

        When this is true, an average miss is larger than the whole range being
        aimed at, and the mean absolute error should not be presented as
        evidence that the window can be hit. Report
        :attr:`fraction_inside` instead.
        """
        return self.mean_absolute_error > (self.window_high - self.window_low)


def fraction_in_window(
    achieved: Sequence[float],
    window_low: float,
    window_high: float,
    target: Sequence[float] | float | None = None,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> WindowSummary:
    """Fraction of achieved values landing inside a target window, with an interval.

    Parameters
    ----------
    achieved
        The realised values, for example achieved net charge per design.
    window_low, window_high
        The window that matters for the application.
    target
        Optional per-design target, or a single target for all. Used only to
        compute the absolute errors reported alongside, so that the window
        statistic and the error statistic can be read together.

    Returns
    -------
    WindowSummary
        Carries both the fraction inside and the absolute errors, plus
        :meth:`WindowSummary.mae_exceeds_window`, which flags the case where an
        average-error summary is misleading about whether the window is
        reachable at all.
    """
    values = np.asarray(achieved, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarise an empty sample")
    if window_low >= window_high:
        raise ValueError(f"window_low must be below window_high, got {window_low}, {window_high}")

    inside = (values >= window_low) & (values <= window_high)

    if target is None:
        centre = (window_low + window_high) / 2.0
        errors = np.abs(values - centre)
    else:
        targets = np.asarray(target, dtype=np.float64)
        if targets.ndim == 0:
            targets = np.full_like(values, float(targets))
        if targets.shape != values.shape:
            raise ValueError("target must be scalar or the same length as achieved")
        errors = np.abs(values - targets)

    return WindowSummary(
        n=int(values.size),
        n_inside=int(inside.sum()),
        fraction_inside=float(inside.mean()),
        window_low=window_low,
        window_high=window_high,
        mean_absolute_error=float(errors.mean()),
        median_absolute_error=float(np.median(errors)),
        interval=bootstrap_ci(
            inside.astype(np.float64),
            np.mean,
            confidence,
            n_resamples,
            seed,
        ),
    )


def window_membership_disagreement(
    charge_definition_a: Sequence[float],
    charge_definition_b: Sequence[float],
    window_low: float,
    window_high: float,
    label_a: str = "simple",
    label_b: str = "ph",
) -> dict[str, Any]:
    """How often two charge definitions disagree about window membership.

    A developability window quoted in the literature was computed under some
    particular charge definition. Plotting achieved charge under a *different*
    definition against that window silently shifts every point relative to the
    band, by an amount that is neither constant nor small.

    This function makes the consequence concrete: it reports how many designs
    are called compliant under one definition and non-compliant under the other.
    If that number is zero the choice is immaterial and the point can be
    dropped. If it is large, the band and the axis have to be brought into the
    same definition before anything is concluded from their relationship.
    """
    a = np.asarray(charge_definition_a, dtype=np.float64)
    b = np.asarray(charge_definition_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"both definitions must cover the same designs, got {a.shape} and {b.shape}"
        )
    if a.size == 0:
        raise ValueError("cannot compare an empty sample")

    inside_a = (a >= window_low) & (a <= window_high)
    inside_b = (b >= window_low) & (b <= window_high)
    disagree = inside_a != inside_b
    offset = a - b

    return {
        "n": int(a.size),
        "window": [window_low, window_high],
        f"n_inside_{label_a}": int(inside_a.sum()),
        f"n_inside_{label_b}": int(inside_b.sum()),
        f"fraction_inside_{label_a}": float(inside_a.mean()),
        f"fraction_inside_{label_b}": float(inside_b.mean()),
        "n_disagree": int(disagree.sum()),
        "fraction_disagree": float(disagree.mean()),
        f"n_only_inside_{label_a}": int((inside_a & ~inside_b).sum()),
        f"n_only_inside_{label_b}": int((inside_b & ~inside_a).sum()),
        "mean_offset": float(offset.mean()),
        "median_offset": float(np.median(offset)),
        "max_absolute_offset": float(np.abs(offset).max()),
        "offset_is_constant": bool(np.ptp(offset) < 1e-9),
        "interpretation": (
            "The two definitions never disagree about this window, so the choice "
            "does not affect any conclusion drawn from it."
            if not disagree.any()
            else (
                f"{int(disagree.sum())} of {int(a.size)} designs change window membership "
                f"depending on which charge definition is used. The window and the axis "
                f"must be expressed in the same definition before the relationship "
                f"between them supports any claim."
            )
        ),
    }


def best_of_n_bias_warning(sample_sizes: Sequence[int]) -> str | None:
    """Flag a best-of-N comparison made across unequal N.

    The expected maximum of a sample grows with sample size, so comparing the
    best value from twenty-eight draws against the best from sixteen favours the
    larger group for reasons that have nothing to do with the treatment.
    Returns a message when the sizes differ, otherwise None.
    """
    sizes = sorted(set(int(n) for n in sample_sizes))
    if len(sizes) <= 1:
        return None
    return (
        f"best-of-N values are being compared across unequal sample sizes {sizes}. "
        "The expected maximum grows with N, so the largest group is favoured "
        "independently of any real effect. Subsample to a common N, or report a "
        "statistic that does not depend on it."
    )


#: Permutations for a rank-correlation significance test. Ten thousand gives a
#: resolution of about 1e-4, which is finer than any claim made from it.
N_PERMUTATIONS = 10_000


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation, without pulling in scipy for one number.

    Returns NaN below three points rather than the ill-defined value numpy
    would produce, so a caller cannot report a correlation from two.
    """
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            f"paired samples must have equal length, got {first.shape} and {second.shape}"
        )
    if first.size < 3:
        return float("nan")
    return float(np.corrcoef(_ranks(first), _ranks(second))[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, matching pandas' default and scipy's ``rankdata``.

    Ties get the mean of the ranks they span. Ranking them arbitrarily instead
    would make the correlation depend on input order, which for a table sorted
    by one of the two variables is not a small effect.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(unique.size, dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def spearman_permutation_p(
    a: Sequence[float],
    b: Sequence[float],
    *,
    seed: int,
    n_permutations: int = N_PERMUTATIONS,
) -> float:
    """Two-sided permutation p-value for a rank correlation.

    Shuffling one side breaks the pairing while keeping both marginal
    distributions intact, which is the right null: the two variables really do
    have the spread they have, and the question is only whether they are paired.

    Uses an add-one correction, so a p-value is never reported as exactly zero
    when the truth is only that it lies below the resolution of the test.
    """
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    observed = spearman(first, second)
    if not np.isfinite(observed):
        return float("nan")

    rng = np.random.default_rng(seed)
    shuffled = second.copy()
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        if abs(spearman(first, shuffled)) >= abs(observed):
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (n_permutations + 1)


def spearman_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile interval for a rank correlation, resampling pairs together.

    Separate from :func:`bootstrap_ci`, which resamples one sample of scalars.
    Here the two values move as a unit: resampling them independently would
    destroy the very association being estimated and return an interval centred
    on zero regardless of the data.

    Resamples that degenerate (every x tied, say) yield NaN and are dropped, and
    the surviving count is reported on the interval so that a quietly thinned
    bootstrap is visible.
    """
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            f"paired samples must have equal length, got {first.shape} and {second.shape}"
        )
    if first.size < 3:
        raise ValueError("a rank correlation needs at least three pairs")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, first.size, size=(n_resamples, first.size))
    replicates = np.array([spearman(first[row], second[row]) for row in draws], dtype=np.float64)
    replicates = replicates[np.isfinite(replicates)]
    if replicates.size == 0:
        raise ValueError("every bootstrap resample was degenerate; no interval to report")

    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=spearman(first, second),
        low=float(np.quantile(replicates, tail)),
        high=float(np.quantile(replicates, 1.0 - tail)),
        confidence=confidence,
        n=int(first.size),
        n_resamples=int(replicates.size),
        seed=seed,
    )
