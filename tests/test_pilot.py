"""Pilot selection and extrapolation.

A pilot exists to replace a planning assumption with a measurement. These tests
pin the two ways that goes wrong quietly: sampling only the largest complexes,
which extrapolates high, and measuring only jobs that paid for their own MSA
search, which also extrapolates high.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))

from af2_multimer import Job, extrapolate_from_pilot, select_pilot

BETAS = [-3.0, -1.5, 0.0, 1.5, 3.0]


def make_jobs(sizes: dict[str, int]) -> list[Job]:
    """One complex per entry, at every beta, with the given total residue count."""
    jobs = []
    for pdb_id, size in sizes.items():
        for beta in BETAS:
            half = size // 2
            jobs.append(
                Job(
                    pdb_id=pdb_id,
                    beta=beta,
                    replicate=0,
                    designed_chain="A",
                    chains={"A": "A" * half, "B": "A" * (size - half)},
                    msa_mode={"A": "single_sequence", "B": "msa"},
                )
            )
    return jobs


SIZES = {f"c{i:02d}": 100 + 40 * i for i in range(10)}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_pilot_spans_the_size_distribution_rather_than_the_top_of_it() -> None:
    """Sampling only the largest complexes would extrapolate a cost far too high."""
    selected = select_pilot(make_jobs(SIZES), n_complexes=4, betas_each=2)
    picked = sorted({job.total_residues() for job in selected})
    smallest_possible = min(SIZES.values())
    largest = max(SIZES.values())

    assert picked[0] < (smallest_possible + largest) / 2, (
        f"pilot clustered at the large end: {picked}"
    )
    assert picked[-1] == largest, "the largest complex must be kept for the risk check"


def test_pilot_always_includes_the_largest_complex() -> None:
    """Out-of-memory and timeout are driven by size, so the pilot must provoke them."""
    for n in (1, 2, 3, 5):
        selected = select_pilot(make_jobs(SIZES), n_complexes=n, betas_each=1)
        assert max(job.total_residues() for job in selected) == max(SIZES.values())


def test_each_pilot_complex_runs_more_than_one_beta() -> None:
    """Otherwise every job pays its own MSA search and none measures the cached case."""
    selected = select_pilot(make_jobs(SIZES), n_complexes=4, betas_each=2)
    by_complex: dict[str, int] = {}
    for job in selected:
        by_complex[job.pdb_id] = by_complex.get(job.pdb_id, 0) + 1
    assert set(by_complex.values()) == {2}


def test_the_reference_beta_is_run_first_within_a_complex() -> None:
    selected = select_pilot(make_jobs(SIZES), n_complexes=2, betas_each=2)
    for pdb_id in {job.pdb_id for job in selected}:
        betas = [job.beta for job in selected if job.pdb_id == pdb_id]
        assert betas[0] == 0.0


def test_asking_for_more_complexes_than_exist_is_not_an_error() -> None:
    selected = select_pilot(make_jobs({"a": 200, "b": 300}), n_complexes=10, betas_each=1)
    assert len({job.pdb_id for job in selected}) == 2


def test_an_empty_job_list_gives_an_empty_pilot() -> None:
    assert select_pilot([], n_complexes=4) == []


def test_a_nonsensical_pilot_size_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_pilot(make_jobs(SIZES), n_complexes=0)


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------


def result(pdb_id: str, minutes: float) -> dict:
    return {"pdb_id": pdb_id, "wall_clock_s": minutes * 60.0}


def test_the_msa_search_is_charged_once_per_complex_not_once_per_job() -> None:
    """The trap this function exists for.

    Ten minutes cold, seven warm. A grid of 100 jobs over 20 complexes pays
    twenty cold and eighty warm, not a hundred cold. Costing it all at the cold
    rate would overstate the compute by 20 percent.
    """
    pilot = [result("a", 10.0), result("a", 7.0), result("b", 10.0), result("b", 7.0)]
    projection = extrapolate_from_pilot(pilot, n_grid_jobs=100, n_grid_complexes=20)

    assert projection["observed_cold_minutes_median"] == 10.0
    assert projection["observed_warm_minutes_median"] == 7.0
    assert projection["msa_overhead_observed_minutes"] == 3.0
    assert projection["n_grid_cold_jobs"] == 20
    assert projection["n_grid_warm_jobs"] == 80

    all_cold = extrapolate_from_pilot(
        [result("a", 10.0), result("b", 10.0)], n_grid_jobs=100, n_grid_complexes=20
    )
    assert projection["estimated_usd"] < all_cold["estimated_usd"]


def test_without_a_warm_job_everything_is_costed_cold_and_says_so() -> None:
    """The conservative direction: an overstated estimate stops a run, it does not blow a budget."""
    projection = extrapolate_from_pilot(
        [result("a", 10.0), result("b", 10.0)], n_grid_jobs=100, n_grid_complexes=20
    )
    assert not projection["warm_measured"]
    assert projection["observed_warm_minutes_median"] == 10.0
    assert "overstates" in projection["basis"]


def test_cost_scales_with_the_grid() -> None:
    pilot = [result("a", 10.0), result("a", 7.0)]
    small = extrapolate_from_pilot(pilot, n_grid_jobs=50, n_grid_complexes=10)
    large = extrapolate_from_pilot(pilot, n_grid_jobs=200, n_grid_complexes=40)
    assert large["estimated_usd"] > small["estimated_usd"]


def test_a_slower_pilot_costs_more() -> None:
    fast = extrapolate_from_pilot([result("a", 5.0), result("a", 4.0)], 100, 20)
    slow = extrapolate_from_pilot([result("a", 20.0), result("a", 18.0)], 100, 20)
    assert slow["estimated_usd"] > fast["estimated_usd"]


def test_the_worst_case_job_is_reported_alongside_the_median() -> None:
    """The median costs the grid; the maximum is what trips the two-hour timeout."""
    pilot = [result("a", 8.0), result("a", 7.0), result("b", 95.0)]
    assert extrapolate_from_pilot(pilot, 100, 20)["observed_max_minutes"] == 95.0


def test_failed_jobs_are_ignored_rather_than_counted_as_free() -> None:
    pilot = [result("a", 10.0), {"pdb_id": "b", "wall_clock_s": None}, result("a", 7.0)]
    assert extrapolate_from_pilot(pilot, 100, 20)["n_pilot_jobs_timed"] == 2


def test_a_pilot_that_returned_nothing_refuses_to_guess() -> None:
    with pytest.raises(ValueError, match="no timed job"):
        extrapolate_from_pilot([{"pdb_id": "a", "wall_clock_s": None}], 100, 20)


def test_more_complexes_than_jobs_does_not_overcount_cold_jobs() -> None:
    projection = extrapolate_from_pilot([result("a", 10.0)], n_grid_jobs=5, n_grid_complexes=20)
    assert projection["n_grid_cold_jobs"] == 5
    assert projection["n_grid_warm_jobs"] == 0
