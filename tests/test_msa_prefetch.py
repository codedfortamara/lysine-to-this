"""The homology search must not happen on a GPU.

This exists because it did. The first Modal pilot ran ``run_mmseqs2`` inside
the A100 containers, and three of eight jobs hit the two-hour timeout having
folded nothing. The search is an HTTP poll against api.colabfold.com, which is
free, shared, and queues under load; ``run_mmseqs2`` waits until it is served.
With twenty containers submitting at once the queue was the whole job, and it
was billed at 2.10 USD an hour.

Nothing about the alignment changes when it is fetched on CPU instead. Same
server, same flags, same cache path. What changes is that the wait costs cents.

These tests cover the planning side, which is pure Python and needs no network:
which searches the grid requires, and that the count is per complex rather than
per job. The guard that stops ``predict`` searching lives inside the Modal
block and is checked by reading the source, since importing it needs colabfold.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))

from af2_multimer import Job, searches_required

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "modal_app" / "af2_multimer.py").read_text()


def make_job(pdb_id: str, beta: float, designed: str = "A", partner: str = "B") -> Job:
    return Job(
        pdb_id=pdb_id,
        beta=beta,
        replicate=0,
        designed_chain=designed,
        chains={designed: "MKV" * 10, partner: "GGAL" * 12},
        msa_mode={designed: "single_sequence", partner: "msa"},
    )


# ---------------------------------------------------------------------------
# What the grid actually has to search for
# ---------------------------------------------------------------------------


def test_one_search_per_native_chain_not_per_job() -> None:
    """The saving that makes the prefetch worth doing at all.

    The partner is held at its native sequence at every charge setting, so all
    five betas of a complex want one identical alignment. Counted per job this
    grid would be 275 searches; per complex it is 55.
    """
    jobs = [make_job("1ABC", beta) for beta in (-3.0, -1.5, 0.0, 1.5, 3.0)]
    assert len(searches_required(jobs)) == 1


def test_designed_chains_are_never_searched_for() -> None:
    """A design has no evolutionary history, and searching would invent one."""
    searches = searches_required([make_job("1ABC", 0.0)])
    assert [chain for _, chain, _ in searches] == ["B"]


def test_distinct_complexes_are_separate_searches() -> None:
    jobs = [make_job("1ABC", 0.0), make_job("2XYZ", 0.0), make_job("1ABC", 1.5)]
    assert {pdb for pdb, _, _ in searches_required(jobs)} == {"1ABC", "2XYZ"}


def test_the_sequence_travels_with_the_request() -> None:
    """The prefetch runs remotely, so it cannot look the sequence up itself."""
    ((_, _, sequence),) = searches_required([make_job("1ABC", 0.0)])
    assert sequence == "GGAL" * 12


def test_the_order_is_deterministic() -> None:
    """A resumed prefetch should ask for the same things in the same order."""
    jobs = [make_job(pdb, 0.0) for pdb in ("2XYZ", "1ABC", "3QQQ")]
    assert searches_required(jobs) == searches_required(list(reversed(jobs)))


def test_a_chain_with_two_sequences_is_refused() -> None:
    """One cache entry cannot serve two sequences, and picking either is wrong.

    If it happened, some jobs would be folded against an alignment built from a
    sequence they do not contain. AlphaFold would consume that without
    complaint and return confident numbers.
    """
    first = make_job("1ABC", 0.0)
    second = make_job("1ABC", 1.5)
    second.chains["B"] = "WWWW" * 9
    with pytest.raises(ValueError, match="two different"):
        searches_required([first, second])


def test_the_invariants_of_msa_plan_still_apply() -> None:
    """searches_required goes through msa_plan, so it inherits its refusals."""
    job = make_job("1ABC", 0.0)
    job.msa_mode["A"] = "msa"
    with pytest.raises(ValueError, match="designed chain"):
        searches_required([job])


# ---------------------------------------------------------------------------
# The guard on the GPU path
# ---------------------------------------------------------------------------


def function_source(name: str) -> str:
    """Source of a function by name, wherever it is nested in the module."""
    tree = ast.parse(SOURCE)
    lines = SOURCE.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{name} not found in af2_multimer.py")


def test_predict_does_not_search_by_default() -> None:
    """A cache miss on the GPU path must raise, not queue."""
    assert "allow_msa_search: bool = False" in function_source("predict")


def test_the_search_call_is_behind_the_guard() -> None:
    source = function_source("build_mixed_a3m")
    assert "if not allow_search:" in source
    assert source.index("if not allow_search:") < source.index("result = run_mmseqs2(")


def test_the_launcher_checks_the_cache_before_dispatching() -> None:
    """Discovering a cold cache after the GPUs are allocated is the expensive way."""
    source = function_source("run")
    assert "cached_alignments()" in source
    assert source.index("cached_alignments()") < source.index("predict.map")


def test_the_prefetch_asks_for_no_gpu() -> None:
    """Paying A100 rates for an HTTP poll is the whole bug being fixed."""
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fetch_msa":
            decorator = node.decorator_list[0]
            keywords = {k.arg for k in decorator.keywords}  # type: ignore[attr-defined]
            assert "gpu" not in keywords, "fetch_msa must not request a GPU"
            return
    raise AssertionError("fetch_msa not found")


def test_the_prefetch_is_gentle_with_a_free_server() -> None:
    """api.colabfold.com is run by people who did not agree to absorb this grid.

    It is also self-interest: the server throttles, so more containers means
    every one of them waits longer.
    """
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fetch_msa":
            decorator = node.decorator_list[0]
            limits = [
                k.value.value
                for k in decorator.keywords  # type: ignore[attr-defined]
                if k.arg == "max_containers"
            ]
            assert limits and limits[0] <= 8, (
                f"fetch_msa runs {limits} at a time against a free shared server"
            )
            return
    raise AssertionError("fetch_msa not found")


def test_the_prefetch_and_the_gpu_path_use_the_same_cache_names() -> None:
    """Two naming schemes would mean a full cache that never gets hit."""
    assert 'f"{pdb_id}_{chain_id}.a3m"' in function_source("fetch_msa")
    assert 'f"{job.pdb_id}_{chain_id}.a3m"' in function_source("build_mixed_a3m")


def test_the_prefetch_and_the_gpu_path_search_with_the_same_flags() -> None:
    """A different search would make prefetched and inline runs different experiments."""
    for name in ("fetch_msa", "build_mixed_a3m"):
        source = function_source(name)
        assert "use_env=True" in source
        assert "use_filter=True" in source
        assert "use_templates=False" in source
        assert "use_pairing=False" in source


def test_an_empty_alignment_is_never_cached() -> None:
    """A cached empty file would silently degrade every beta of that complex."""
    assert "Refusing to write an empty cache entry" in function_source("fetch_msa")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_the_alignment_wait_is_recorded_apart_from_the_folding() -> None:
    """In the first pilot both were "the job took two hours" and nothing said which."""
    source = function_source("predict")
    assert '"msa_seconds"' in source
    assert '"fold_seconds"' in source


def test_the_recost_uses_folding_time_not_wall_clock() -> None:
    """With the cache warm the alignment wait is off the GPU, so it is not billed."""
    source = function_source("timings")
    assert "msa_overhead_minutes=0.0" in source
    assert "fold_seconds" in source


# ---------------------------------------------------------------------------
# Orchestration that does not depend on the launching machine
# ---------------------------------------------------------------------------


def test_the_grid_can_be_driven_from_inside_modal() -> None:
    """The laptop was load bearing, and it failed three times in one evening.

    ``run`` submits batch by batch from the client and holds the budget guard
    in memory, so a dropped connection, a closed lid or a reboot stops the
    grid. ``--detach`` keeps containers alive but not the loop that decides
    what to start next, so it still halts at the end of the batch in flight.
    """
    source = function_source("drive")
    assert "predict.map(" in source, "drive must dispatch the work itself"
    assert "BudgetGuard(" in source, "the ceiling must be enforced beside the work"


def test_the_driver_asks_for_no_gpu() -> None:
    """It waits on other containers; paying A100 rates to do that is the old bug."""
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "drive":
            keywords = {k.arg for k in node.decorator_list[0].keywords}  # type: ignore[attr-defined]
            assert "gpu" not in keywords
            return
    raise AssertionError("drive not found")


def test_the_driver_outlives_the_whole_grid() -> None:
    """A driver timing out mid-grid would strand the run with no orchestrator."""
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "drive":
            timeouts = [
                k.value.value
                for k in node.decorator_list[0].keywords  # type: ignore[attr-attr,attr-defined]
                if k.arg == "timeout"
            ]
            assert timeouts and timeouts[0] >= 8 * 3600, (
                f"drive times out after {timeouts}, which is inside the plausible "
                "range for this grid"
            )
            return
    raise AssertionError("drive not found")


def test_the_launcher_spawns_rather_than_waits() -> None:
    """.remote() would block the client and reintroduce the dependency."""
    source = function_source("launch")
    assert "drive.spawn(" in source
    assert "drive.remote(" not in source


def test_progress_survives_the_client_going_away() -> None:
    """Status has to be readable from a machine that was switched off throughout."""
    source = function_source("drive")
    assert "progress.json" in source
    assert "results_volume.commit()" in source
    assert "publish(" in source

    status_source = function_source("progress")
    assert 'read_file("progress.json")' in status_source


def test_the_launcher_checks_the_cache_before_spawning() -> None:
    """Same reason as run: a cold cache means every job fails on the GPU path."""
    source = function_source("launch")
    assert "cached_alignments()" in source
    assert source.index("cached_alignments()") < source.index("drive.spawn(")
