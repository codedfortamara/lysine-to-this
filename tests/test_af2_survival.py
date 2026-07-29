"""The AF2 analysis, exercised on a synthetic table before any real grid exists.

Written at the same time as the script it tests, and for the same reason: the
analysis of the AlphaFold arm was fixed before the numbers arrived, so that no
choice in it could be made to suit them. Tests are the only way to know it
works until then, and "it ran on the real data" is not a thing that can be
checked at 2am on the deadline.

The synthetic table here is a fixture, not a result. It exists to drive code
paths and its numbers are constructed to make a known answer come out.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "12_af2_interface_survival.py"


def load_module():
    spec = importlib.util.spec_from_file_location("af2_survival", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def survival():
    return load_module()


BETAS = [-3.0, -1.5, 0.0, 1.5, 3.0]


def synthetic_metrics(
    n_complexes: int = 12,
    effect_per_unit_beta: float = -0.05,
    plddt_at_extremes: float = 40.0,
) -> pd.DataFrame:
    """A grid where ipSAE falls linearly with |beta| and the extremes fold badly.

    Both are put in deliberately: the first is the effect the primary endpoint
    should recover, the second is the confound the fold-quality control exists
    to separate from it.
    """
    rows = []
    for index in range(n_complexes):
        pdb_id = f"{index + 1}ABC"
        # A large per-complex offset, so a test that accidentally drops the
        # pairing fails loudly rather than approximately passing.
        offset = 0.30 * index
        for beta in BETAS:
            folded = abs(beta) < 3.0
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "designed_chain": "A",
                    "replicate": 0,
                    "beta": beta,
                    "key": f"{pdb_id}_A_b{beta}_r0",
                    "ipsae_d0res": 0.5 + offset + effect_per_unit_beta * abs(beta),
                    "interface_ptm": 0.6 + offset,
                    "interface_pae": 10.0 + abs(beta),
                    "designed_chain_plddt": 85.0 if folded else plddt_at_extremes,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_the_change_is_measured_against_the_same_complex(survival) -> None:
    """The between-complex spread here is six times the effect being measured.

    If the comparison were not paired, the offset would swamp it entirely.
    """
    changes = survival.paired_change(synthetic_metrics(), "ipsae_d0res")
    at_three = changes[changes["beta"] == 3.0]["delta"]
    assert at_three.std() < 1e-9, "paired deltas should be identical by construction"
    assert at_three.mean() == pytest.approx(-0.15)


def test_the_reference_rows_are_dropped(survival) -> None:
    """Their delta is zero by construction, and keeping them dilutes every mean."""
    changes = survival.paired_change(synthetic_metrics(), "ipsae_d0res")
    assert (changes["beta"] != 0.0).all()


def test_a_complex_with_no_reference_is_dropped_not_paired_elsewhere(survival) -> None:
    """Pairing across complexes would be worse than dropping the row."""
    table = synthetic_metrics()
    table = table[~((table["pdb_id"] == "1ABC") & (table["beta"] == 0.0))]
    changes = survival.paired_change(table, "ipsae_d0res")
    assert "1ABC" not in set(changes["pdb_id"])
    assert len(set(changes["pdb_id"])) == 11


# ---------------------------------------------------------------------------
# The primary endpoint
# ---------------------------------------------------------------------------


def test_the_primary_endpoint_is_fixed_in_advance(survival) -> None:
    """One metric, named in the source, not chosen from the output."""
    assert survival.PRIMARY_METRIC == "ipsae_d0res"
    assert survival.PRIMARY_METRIC not in survival.SECONDARY_METRICS


def test_it_recovers_an_effect_that_is_there(survival) -> None:
    rows = survival.summarise_metric(synthetic_metrics(), "ipsae_d0res", seed=0, label="t")
    at_three = next(r for r in rows if r["beta"] == 3.0)
    assert at_three["mean_change_vs_beta0"] == pytest.approx(-0.15)
    assert at_three["excludes_zero"]
    assert at_three["fraction_degraded"] == 1.0


def test_it_reports_no_effect_when_there_is_none(survival) -> None:
    """The negative result has to be reportable, or the analysis only finds effects."""
    rows = survival.summarise_metric(
        synthetic_metrics(effect_per_unit_beta=0.0), "ipsae_d0res", seed=0, label="t"
    )
    assert all(not r["excludes_zero"] for r in rows)


def test_degraded_means_worse_whichever_direction_the_metric_runs(survival) -> None:
    """interface_pae rises as the interface gets worse; ipSAE falls."""
    table = synthetic_metrics()
    pae = survival.summarise_metric(table, "interface_pae", seed=0, label="t")
    ipsae = survival.summarise_metric(table, "ipsae_d0res", seed=0, label="t")
    assert next(r for r in pae if r["beta"] == 3.0)["lower_is_better"]
    assert not next(r for r in ipsae if r["beta"] == 3.0)["lower_is_better"]
    # Both got worse at beta = 3, so both should count every complex as degraded.
    assert next(r for r in pae if r["beta"] == 3.0)["fraction_degraded"] == 1.0
    assert next(r for r in ipsae if r["beta"] == 3.0)["fraction_degraded"] == 1.0


def test_complexes_are_the_resampling_unit_not_rows(survival) -> None:
    """Replicates within a complex are not independent draws.

    Counting them as such would narrow every interval by roughly the square
    root of the replicate count, for free and for nothing.
    """
    single = synthetic_metrics()
    doubled = pd.concat([single, single.assign(replicate=1)], ignore_index=True)
    rows_single = survival.summarise_metric(single, "ipsae_d0res", seed=0, label="t")
    rows_doubled = survival.summarise_metric(doubled, "ipsae_d0res", seed=0, label="t")
    for a, b in zip(rows_single, rows_doubled, strict=True):
        assert a["n_complexes"] == b["n_complexes"]


# ---------------------------------------------------------------------------
# The confound
# ---------------------------------------------------------------------------


def test_the_fold_quality_subset_uses_the_designed_chain(survival) -> None:
    """The complex mean is half native partner and would hide a collapsed design."""
    source = SCRIPT.read_text()
    assert "designed_chain_plddt" in source
    assert 'table["mean_plddt"]' not in source


def test_the_reading_distinguishes_a_broken_interface_from_a_broken_monomer(
    survival,
) -> None:
    """All four outcomes are written out in advance, so none can be talked into."""
    frame = pd.DataFrame(
        [{"excludes_zero": True, "mean_change_vs_beta0": -0.2}],
    )
    none = pd.DataFrame([{"excludes_zero": False, "mean_change_vs_beta0": -0.01}])

    assert "not explained by monomer quality" in survival.fold_quality_reading(frame, frame)
    assert "breaking monomers rather than" in survival.fold_quality_reading(frame, none)
    assert "collider" in survival.fold_quality_reading(none, frame)
    assert "No measurable degradation" in survival.fold_quality_reading(none, none)


def test_without_the_control_it_says_so_rather_than_concluding(survival) -> None:
    reading = survival.fold_quality_reading(pd.DataFrame([{"a": 1}]), pd.DataFrame())
    assert "cannot be distinguished" in reading


# ---------------------------------------------------------------------------
# Missing jobs
# ---------------------------------------------------------------------------


def test_a_depleted_beta_is_flagged(survival) -> None:
    """Refolding failures concentrate at the extremes, so this will happen."""
    table = synthetic_metrics()
    drop = table[(table["beta"] == 3.0)].index[:8]
    report = survival.completeness(table.drop(index=drop), expected_complexes=12)
    assert 3.0 in report["betas_materially_depleted"]
    assert not report["comparable_across_beta"]


def test_a_complete_grid_is_not_flagged(survival) -> None:
    report = survival.completeness(synthetic_metrics(), expected_complexes=12)
    assert report["comparable_across_beta"]
    assert report["betas_materially_depleted"] == []


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_it_refuses_a_table_with_no_primary_endpoint(survival, tmp_path: Path) -> None:
    """Silently falling back to ipTM would answer a different question."""
    table = synthetic_metrics().drop(columns=["ipsae_d0res"])
    path = tmp_path / "af2_metrics.csv"
    table.to_csv(path, index=False)
    with pytest.raises(KeyError, match="primary endpoint"):
        survival.load_metrics(path)


def test_it_refuses_a_table_it_cannot_pair(survival, tmp_path: Path) -> None:
    table = synthetic_metrics().drop(columns=["designed_chain"])
    path = tmp_path / "af2_metrics.csv"
    table.to_csv(path, index=False)
    with pytest.raises(KeyError, match="paired on"):
        survival.load_metrics(path)


def test_it_fails_cleanly_with_no_data_at_all(tmp_path: Path) -> None:
    """The project rule: a useful message, not a traceback."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--metrics", str(tmp_path / "absent.csv")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "prefetch" in result.stderr, "the message should say how to produce the input"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_it_runs_end_to_end_and_writes_a_manifest(tmp_path: Path) -> None:
    metrics = tmp_path / "af2_metrics.csv"
    synthetic_metrics().to_csv(metrics, index=False)
    out = tmp_path / "results"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--metrics",
            str(metrics),
            "--results-dir",
            str(out),
            "--analysis",
            str(tmp_path / "absent.csv"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads((out / "af2_interface_survival_summary.json").read_text())
    assert summary["primary_endpoint"] == "ipsae_d0res"
    assert summary["primary_all_returned"]
    assert summary["primary_folded_only"], "the fold-quality subset should be populated"
    assert (out / "af2_interface_survival.csv").is_file()
    assert (out / "af2_interface_survival.csv.manifest.json").is_file()


def test_the_exploratory_link_is_labelled_as_exploratory(tmp_path: Path, survival) -> None:
    """One number over 55 complexes, from two experiments designed for other things."""
    changes = survival.paired_change(synthetic_metrics(n_complexes=20), "ipsae_d0res")
    analysis = pd.DataFrame(
        [
            {
                "pdb_id": f"{i + 1}ABC",
                "partition": part,
                "buffering_ratio": 1.0 + (0.03 * i if part == "interface" else 0.0),
            }
            for i in range(20)
            for part in ("interface", "surface")
        ]
    )
    path = tmp_path / "analysis_per_complex.csv"
    analysis.to_csv(path, index=False)
    report = survival.buffering_versus_survival(changes, path, seed=0)
    assert report["status"].startswith("EXPLORATORY")


def test_the_exploratory_link_refuses_too_few_complexes(tmp_path: Path, survival) -> None:
    changes = survival.paired_change(synthetic_metrics(n_complexes=4), "ipsae_d0res")
    analysis = pd.DataFrame(
        [
            {
                "pdb_id": f"{i + 1}ABC",
                "partition": part,
                "buffering_ratio": 1.0 + (0.03 * i if part == "interface" else 0.0),
            }
            for i in range(4)
            for part in ("interface", "surface")
        ]
    )
    path = tmp_path / "analysis_per_complex.csv"
    analysis.to_csv(path, index=False)
    report = survival.buffering_versus_survival(changes, path, seed=0)
    assert "not reported" in report["note"]


def test_a_half_populated_control_is_refused(tmp_path: Path) -> None:
    """The dangerous case: the control present on one backend and absent on the other.

    A folded subset drawn only from the machine that happened to record pLDDT
    differs from the full set by backend as much as by fold quality. That is a
    confound wearing the costume of a control, and unlike a wholly missing
    column it produces output that looks fine.
    """
    table = synthetic_metrics()
    table.loc[table["beta"] < 0, "designed_chain_plddt"] = None
    table["source"] = ["colab" if b < 0 else "modal" for b in table["beta"]]
    metrics = tmp_path / "af2_metrics.csv"
    table.to_csv(metrics, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--metrics",
            str(metrics),
            "--results-dir",
            str(tmp_path / "results"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "colab" in result.stderr, "the message should name which rows are short"


def test_the_colab_notebook_records_the_same_control(tmp_path: Path) -> None:
    """Two backends must not disagree about the primary control.

    collect_metrics is not in the notebook's shared block, so a change on the
    Modal side does not propagate and nothing else would catch it.
    """
    notebook = json.loads((REPO_ROOT / "notebooks" / "af2_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert '"designed_chain_plddt"' in source
    assert '"partner_chain_plddt"' in source


def test_it_says_why_the_control_produced_nothing(tmp_path: Path) -> None:
    """ "No fold-quality control was possible" alone is useless.

    It cannot distinguish every design failing the pLDDT floor from a subset
    with no beta = 0 rows left to pair against, and those call for opposite
    responses: one says the designs are bad, the other says the floor is
    filtering out the references.
    """
    table = synthetic_metrics()
    table["designed_chain_plddt"] = 40.0  # nothing clears the floor
    metrics = tmp_path / "af2_metrics.csv"
    table.to_csv(metrics, index=False)
    out = tmp_path / "results"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--metrics", str(metrics), "--results-dir", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads((out / "af2_interface_survival_summary.json").read_text())
    assert "no design cleared pLDDT" in summary["fold_quality_control_note"]
    assert summary["designed_chain_plddt_summary"]["n_at_or_above_floor"] == 0
    assert "reason:" in result.stderr


def test_the_plddt_distribution_is_always_reported(tmp_path: Path) -> None:
    """The floor is a judgement call, so the reader needs the spread it sits in."""
    metrics = tmp_path / "af2_metrics.csv"
    synthetic_metrics().to_csv(metrics, index=False)
    out = tmp_path / "results"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--metrics", str(metrics), "--results-dir", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )
    summary = json.loads((out / "af2_interface_survival_summary.json").read_text())
    spread = summary["designed_chain_plddt_summary"]
    assert spread["min"] <= spread["median"] <= spread["max"]
    assert spread["floor"] == 70.0


# ---------------------------------------------------------------------------
# Backend mixing inside a paired comparison
# ---------------------------------------------------------------------------


def test_a_lineage_split_across_backends_is_flagged(survival) -> None:
    """A paired difference assumes only the charge differs between the two rows.

    AlphaFold on CPU and on GPU do not produce identical numbers, which is what
    oneDNN warns about at every container start. 37 jobs in this grid ran on CPU
    because of a cuDNN mismatch, so any grid mixing those with later GPU jobs
    mixes backends inside the pairs that carry the claim.
    """
    table = synthetic_metrics(n_complexes=3)
    table["jax_device_kind"] = "NVIDIA L4"
    table.loc[table["beta"] == 0.0, "jax_device_kind"] = None  # references on CPU

    report = survival.backend_consistency(table)

    assert report["checkable"]
    assert report["n_lineages_spanning_more_than_one_device"] == 3
    assert "hardware" in report["note"]


def test_a_single_backend_grid_is_not_flagged(survival) -> None:
    table = synthetic_metrics(n_complexes=3)
    table["jax_device_kind"] = "NVIDIA L4"
    report = survival.backend_consistency(table)
    assert report["n_lineages_spanning_more_than_one_device"] == 0
    assert "same backend" in report["note"]


def test_a_grid_with_no_device_column_says_it_cannot_be_checked(survival) -> None:
    """Silence is not the same as a clean bill of health."""
    report = survival.backend_consistency(synthetic_metrics())
    assert report["checkable"] is False
    assert "cannot be ruled out" in report["note"]


def test_the_warning_reaches_the_operator(tmp_path: Path) -> None:
    table = synthetic_metrics(n_complexes=3)
    table["jax_device_kind"] = "NVIDIA L4"
    table.loc[table["beta"] == 0.0, "jax_device_kind"] = None
    metrics = tmp_path / "af2_metrics.csv"
    table.to_csv(metrics, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--metrics",
            str(metrics),
            "--results-dir",
            str(tmp_path / "results"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "different hardware" in result.stderr


# ---------------------------------------------------------------------------
# The fold-quality control has to be calibrated for the regime it measures
# ---------------------------------------------------------------------------


def test_the_relative_control_survives_a_uniformly_low_pLDDT_regime(survival) -> None:
    """The defect: an absolute floor imported from a different measurement regime.

    The designed chain is folded from its own sequence with no alignment, because
    a design has no evolutionary history worth searching. Single-sequence
    AlphaFold predictions sit systematically lower than MSA-backed ones, and the
    conventional floor of 70 comes from the MSA regime. On the real data it
    rejected 30 of 31 rows, which says nothing about the designs.
    """
    table = synthetic_metrics(n_complexes=6)
    # Everything low, as single-sequence predictions are, but internally stable.
    table["designed_chain_plddt"] = 55.0

    kept = survival.retained_confidence(table, drop_limit=10.0)

    assert len(kept) == len(table), "a stable low-confidence regime must not be filtered out"
    assert (table["designed_chain_plddt"] < survival.PLDDT_FLOOR).all(), (
        "and the absolute floor would have rejected all of it"
    )


def test_a_design_that_lost_confidence_against_its_own_reference_is_excluded(survival) -> None:
    """What the control is actually for: the design stopped folding."""
    table = synthetic_metrics(n_complexes=4)
    table["designed_chain_plddt"] = 80.0
    collapsed = (table["pdb_id"] == "1ABC") & (table["beta"] == 3.0)
    table.loc[collapsed, "designed_chain_plddt"] = 40.0

    kept = survival.retained_confidence(table, drop_limit=10.0)

    assert len(kept) == len(table) - 1
    assert not ((kept["pdb_id"] == "1ABC") & (kept["beta"] == 3.0)).any()


def test_a_design_more_confident_than_its_reference_is_kept(survival) -> None:
    """The filter is one-sided. Rising confidence is not a failure to fold."""
    table = synthetic_metrics(n_complexes=3)
    table["designed_chain_plddt"] = 70.0
    table.loc[table["beta"] == 1.5, "designed_chain_plddt"] = 95.0
    assert len(survival.retained_confidence(table, drop_limit=10.0)) == len(table)


def test_rows_without_a_reference_are_kept_not_quietly_dropped(survival) -> None:
    """Their absence is a completeness problem, reported as one elsewhere.

    Dropping them here as well would count the same gap twice, and would do it
    invisibly under a label that says fold quality.
    """
    table = synthetic_metrics(n_complexes=3)
    table["designed_chain_plddt"] = 80.0
    table = table[~((table["pdb_id"] == "1ABC") & (table["beta"] == 0.0))]

    kept = survival.retained_confidence(table, drop_limit=10.0)

    assert (kept["pdb_id"] == "1ABC").sum() == (table["pdb_id"] == "1ABC").sum()


def test_both_controls_are_reported(tmp_path: Path) -> None:
    """The absolute floor stays visible, clearly labelled, because readers expect it."""
    table = synthetic_metrics(n_complexes=6)
    table["designed_chain_plddt"] = 55.0
    metrics = tmp_path / "af2_metrics.csv"
    table.to_csv(metrics, index=False)
    out = tmp_path / "results"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--metrics", str(metrics), "--results-dir", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    written = pd.read_csv(out / "af2_interface_survival.csv")
    subsets = set(written["subset"])
    assert any(s.startswith("plddt_within_") for s in subsets), subsets

    # The absolute floor rejects everything in this regime, so it contributes no
    # rows. That must be said rather than left as a silently absent subset: a
    # reader who expects the conventional threshold needs to know it was applied
    # and found nothing, not wonder whether it ran.
    summary = json.loads((out / "af2_interface_survival_summary.json").read_text())
    assert "no design cleared pLDDT" in summary["fold_quality_control_note"]
    spread = summary["designed_chain_plddt_summary"]
    assert spread["n_at_or_above_floor"] == 0
    assert spread["median"] < spread["floor"]
