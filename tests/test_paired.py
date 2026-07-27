"""Within-complex pairing of the refolding metrics.

The fixtures here are small hand-built tables rather than real refolding output,
because the properties being checked are properties of the pairing, and a real
table would make them harder to read rather than more convincing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from interface_charge.paired import balanced_panel, paired_deltas, pairing_report

BETAS = [-3.0, -1.5, 0.0, 1.5, 3.0]


def grid(n_complexes: int = 3, betas: list[float] | None = None) -> pd.DataFrame:
    """A complete grid where each complex has its own baseline and its own trend.

    Complex ``i`` sits at a different absolute level, which is the between-
    complex variance the pairing is meant to remove, and every complex loses the
    same amount per unit of absolute charge, which is the effect it is meant to
    recover.
    """
    rows = []
    for index in range(n_complexes):
        for beta in betas if betas is not None else BETAS:
            rows.append(
                {
                    "pdb_id": f"complex{index}",
                    "designed_chain": "A",
                    "replicate": 0,
                    "beta": beta,
                    "interface_ptm": 10.0 * index + 0.8 - 0.1 * abs(beta),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# What pairing is for
# ---------------------------------------------------------------------------


def test_pairing_removes_the_between_complex_offset() -> None:
    """Complexes an order of magnitude apart in level give identical deltas.

    Without pairing the spread across complexes is about 20 units and the charge
    effect is about 0.3, so the effect is invisible. After pairing the spread is
    zero and the effect is all that remains.
    """
    table = grid()
    assert table["interface_ptm"].std() > 5.0

    paired = paired_deltas(table, ["interface_ptm"])
    at_three = paired[paired["beta"] == 3.0]["delta_interface_ptm"]
    assert at_three.std() == pytest.approx(0.0, abs=1e-12)
    assert at_three.iloc[0] == pytest.approx(-0.3)


def test_the_reference_row_has_a_zero_delta() -> None:
    paired = paired_deltas(grid(), ["interface_ptm"])
    reference = paired[paired["beta"] == 0.0]
    assert (reference["delta_interface_ptm"] == 0.0).all()


def test_lineages_are_paired_only_to_themselves() -> None:
    """A delta must never be computed against a different complex's baseline."""
    table = grid(n_complexes=4)
    paired = paired_deltas(table, ["interface_ptm"])
    for pdb_id, group in paired.groupby("pdb_id"):
        baseline = table[(table["pdb_id"] == pdb_id) & (table["beta"] == 0.0)][
            "interface_ptm"
        ].iloc[0]
        expected = group["interface_ptm"] - baseline
        assert group["delta_interface_ptm"].to_numpy() == pytest.approx(expected.to_numpy())


def test_replicates_are_separate_lineages() -> None:
    table = grid(n_complexes=1)
    other = table.copy()
    other["replicate"] = 1
    other["interface_ptm"] = other["interface_ptm"] + 5.0
    paired = paired_deltas(pd.concat([table, other], ignore_index=True), ["interface_ptm"])
    for _, group in paired.groupby("replicate"):
        assert group[group["beta"] == 0.0]["delta_interface_ptm"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Missing references
# ---------------------------------------------------------------------------


def test_a_missing_reference_gives_no_delta_rather_than_a_wrong_one() -> None:
    """The failure this function exists to prevent.

    If a complex failed to refold at beta = 0, nothing it did at other charge
    settings can be expressed as a change. Filling that with a zero, or with
    another complex's baseline, would read as a measurement.
    """
    table = grid(n_complexes=2)
    table = table[~((table["pdb_id"] == "complex1") & (table["beta"] == 0.0))]

    paired = paired_deltas(table, ["interface_ptm"])
    orphaned = paired[paired["pdb_id"] == "complex1"]
    assert orphaned["delta_interface_ptm"].isna().all()
    assert not orphaned["has_reference"].any()

    kept = paired[paired["pdb_id"] == "complex0"]
    assert kept["delta_interface_ptm"].notna().all()
    assert kept["has_reference"].all()


def test_a_duplicated_reference_raises_instead_of_picking_one() -> None:
    table = grid(n_complexes=1)
    duplicate = table[table["beta"] == 0.0]
    with pytest.raises(ValueError, match="more than one row"):
        paired_deltas(pd.concat([table, duplicate], ignore_index=True), ["interface_ptm"])


def test_asking_for_an_absent_metric_raises() -> None:
    with pytest.raises(KeyError, match="absent metric"):
        paired_deltas(grid(), ["ipsae_d0res"])


def test_a_table_without_pairing_keys_raises() -> None:
    table = grid().drop(columns=["replicate"])
    with pytest.raises(KeyError, match="replicate"):
        paired_deltas(table, ["interface_ptm"])


# ---------------------------------------------------------------------------
# The balanced panel, and survivorship
# ---------------------------------------------------------------------------


def test_a_complete_grid_is_entirely_balanced() -> None:
    table = grid(n_complexes=3)
    assert len(balanced_panel(table)) == len(table)


def test_a_lineage_missing_one_beta_leaves_the_panel_entirely() -> None:
    """Partial lineages are dropped whole, not just at the missing setting.

    Keeping the rows that did come back would be exactly the survivorship bias
    the panel exists to remove.
    """
    table = grid(n_complexes=3)
    table = table[~((table["pdb_id"] == "complex2") & (table["beta"] == 3.0))]
    panel = balanced_panel(table, BETAS)
    assert set(panel["pdb_id"]) == {"complex0", "complex1"}
    assert len(panel) == 10


def test_survivorship_shows_up_as_a_gap_in_the_report() -> None:
    """The scenario from the upstream data: hard complexes fail at extreme charge."""
    table = grid(n_complexes=5)
    table = table[~(table["pdb_id"].isin(["complex3", "complex4"]) & (table["beta"].abs() == 3.0))]

    report = pairing_report(table, BETAS)
    assert report["n_lineages"] == 5
    assert report["n_lineages_complete_across_beta"] == 3
    counts = {row["beta"]: row["n_returned"] for row in report["per_beta"]}
    assert counts[0.0] == 5
    assert counts[3.0] == 3


def test_an_empty_panel_is_empty_not_an_error() -> None:
    table = grid(n_complexes=2)
    table = table[table["beta"] != 3.0]
    assert len(balanced_panel(table, BETAS)) == 0
    assert pairing_report(table, BETAS)["n_lineages_complete_across_beta"] == 0
