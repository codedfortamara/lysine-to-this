"""Within-complex pairing for the refolding metrics.

Why the absolute numbers are the wrong thing to report
------------------------------------------------------
Interface pTM, interface PAE and interface RMSD vary far more between complexes
than they do across the charge grid within one complex. A small tight interface
and a large flat one score differently at every beta for reasons that have
nothing to do with charge. Averaging across 55 complexes at each beta therefore
buries the effect under between-complex variance, and the standard error is
dominated by which complexes happen to be in the set.

Every metric is instead reported as a difference against the *same complex* at
beta = 0. Each complex is then its own control, the between-complex term cancels
exactly, and what is left is the quantity the paper is actually about.

The survivorship problem this also fixes
----------------------------------------
Refolding failures are not random. Large complexes and heavily charged designs
fail more often, so the set of complexes that returned a result at beta = +/-3
is easier than the set that returned one at beta = 0. Comparing the mean at one
beta against the mean at another then compares two different populations, and
the charge effect is confounded with which complexes survived.

Pairing removes this only if the pairing is enforced: a delta exists solely when
both the design and its own beta = 0 reference came back. ``balanced_panel``
goes further and restricts to complexes present at *every* beta, which is the
set the headline numbers should be computed on. The two counts are reported
side by side, because a large gap between them is itself a finding about how
the dial affects foldability.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

__all__ = [
    "REFERENCE_BETA",
    "balanced_panel",
    "paired_deltas",
    "pairing_report",
]

#: The charge setting every other setting is measured against: no bias applied.
REFERENCE_BETA: Final[float] = 0.0

#: Columns identifying one design lineage across the beta grid. A delta is only
#: meaningful between rows agreeing on all of these.
PAIR_KEYS: Final[tuple[str, ...]] = ("pdb_id", "designed_chain", "replicate")


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise KeyError(
            f"{what} requires column(s) {missing}, which are not in the table. "
            f"Present: {sorted(table.columns)}"
        )


def paired_deltas(
    table: pd.DataFrame,
    metrics: list[str],
    reference_beta: float = REFERENCE_BETA,
) -> pd.DataFrame:
    """Difference every metric against the same design lineage at the reference beta.

    Returns the table with a ``delta_<metric>`` column per metric, plus
    ``has_reference`` recording whether a reference row existed at all.

    A row whose reference is missing gets a null delta, never a value computed
    against some other complex's reference and never a zero. That is the whole
    point of the function: a missing reference means the comparison cannot be
    made, and saying so is the correct output.
    """
    _require_columns(table, (*PAIR_KEYS, "beta"), "paired_deltas")
    missing_metrics = [m for m in metrics if m not in table.columns]
    if missing_metrics:
        raise KeyError(f"paired_deltas asked for absent metric(s) {missing_metrics}")

    keys = list(PAIR_KEYS)
    reference = table[table["beta"] == reference_beta]

    duplicated = reference.duplicated(subset=keys, keep=False)
    if duplicated.any():
        offenders = reference.loc[duplicated, keys].drop_duplicates().to_dict("records")
        raise ValueError(
            f"more than one row at beta = {reference_beta} for {offenders[:5]}. "
            "The reference must be unique per design lineage, otherwise the "
            "delta depends on row order."
        )

    reference = reference[keys + metrics].rename(columns={m: f"_ref_{m}" for m in metrics})
    merged = table.merge(reference, on=keys, how="left", validate="many_to_one")

    for metric in metrics:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"_ref_{metric}"]
    merged["has_reference"] = merged[[f"_ref_{m}" for m in metrics]].notna().any(axis=1)
    return merged.drop(columns=[f"_ref_{m}" for m in metrics])


def balanced_panel(table: pd.DataFrame, betas: list[float] | None = None) -> pd.DataFrame:
    """Restrict to design lineages that returned a result at every beta.

    This is the set on which a comparison across beta is like for like. Anything
    else compares populations that differ in composition as well as in charge.
    """
    _require_columns(table, (*PAIR_KEYS, "beta"), "balanced_panel")
    wanted = set(betas) if betas is not None else set(table["beta"].unique())
    if not wanted:
        return table.iloc[0:0]

    keys = list(PAIR_KEYS)
    present = table.groupby(keys)["beta"].agg(lambda values: wanted.issubset(set(values)))
    complete = present[present].index
    if len(complete) == 0:
        return table.iloc[0:0]
    return table.set_index(keys).loc[complete].reset_index()


def pairing_report(table: pd.DataFrame, betas: list[float] | None = None) -> dict:
    """How much of the grid actually came back, stated plainly.

    The shortfall belongs in the supplement. A charge setting where a third of
    the complexes failed to refold has not produced a mean that can be compared
    against one where all of them did, and the only way a reader can tell is if
    the counts are printed.
    """
    _require_columns(table, (*PAIR_KEYS, "beta"), "pairing_report")
    balanced = balanced_panel(table, betas)
    keys = list(PAIR_KEYS)
    per_beta = (
        table.groupby("beta").size().rename("n_returned").reset_index().to_dict(orient="records")
    )
    return {
        "n_rows": len(table),
        "n_lineages": len(table[keys].drop_duplicates()),
        "n_lineages_complete_across_beta": len(balanced[keys].drop_duplicates())
        if len(balanced)
        else 0,
        "n_rows_in_balanced_panel": len(balanced),
        "per_beta": per_beta,
        "note": (
            "Compare n_lineages against n_lineages_complete_across_beta. A gap "
            "means some designs refolded at some charge settings and not at "
            "others, so any statistic computed on all returned rows compares "
            "different sets of complexes at different beta."
        ),
    }
