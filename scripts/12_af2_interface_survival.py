#!/usr/bin/env python
"""Does the interface survive the charge dial, under complex-aware refolding?

What this answers
-----------------
The upstream work scores each designed chain on its own with ESMFold, which
cannot see a partner and therefore cannot say anything about whether the
interface is still there. This arm refolds the whole complex with
AlphaFold2-Multimer and asks the question the monomer score cannot: as net
charge is dialled away from native, does the predicted interface degrade, and
how fast.

Written before the grid finished
--------------------------------
Deliberately. Every choice below was fixed while the results volume held eight
jobs, so none of it can have been selected to suit the numbers. Where a decision
could have gone either way it is stated here rather than left to the reader to
infer from the output.

The primary endpoint
--------------------
**ipSAE with d0 taken from the interface residue count** (``ipsae_d0res``),
as a within-complex paired change against the same complex at beta = 0.

Two parts to that, both load bearing.

*Why ipSAE and not ipTM.* ipTM's normalisation uses the total length of the
chains, so the same quality of interface scores differently on a 200-residue
complex and a 1200-residue one. This set spans 165 to 1257 residues, so pooling
raw ipTM across it compares numbers that are not on the same scale. ipSAE takes
d0 from the number of residues actually at the interface, which is the quantity
the score is about. ipTM is reported alongside as a secondary endpoint because
it is what most readers will recognise, not because it is the better measure
here.

*Why paired.* Between-complex variation in every one of these metrics is far
larger than the within-complex effect of the charge dial. An unpaired
comparison of beta = +3 designs against beta = 0 designs across different
complexes would be dominated by which complexes happened to land in each group.
Each design is compared against the same complex at beta = 0, and complexes are
resampled as units.

The confound that has to be controlled
--------------------------------------
A design that does not fold at all will score badly at the interface, and that
tells us nothing about the interface. Charge and foldability are not
independent: the upstream monomer data has only 10 to 18 percent of beta = +/-3
designs folding.

So the primary endpoint is reported twice: over every design that came back,
and over the subset whose *designed chain* cleared a pLDDT floor. The chain
pLDDT is used rather than the complex mean because the complex is roughly half
native partner, held identical at every beta, which damps the very signal being
controlled for.

Neither number is the answer on its own. If the interface degrades in the
unfiltered set and not in the folded subset, the charge dial is breaking
monomers rather than interfaces, and the paper should say so. If it degrades in
both, the interface is losing something over and above fold quality.

Missing jobs are not missing at random
--------------------------------------
Refolding failures concentrate in large complexes and extreme betas. Analysing
only what came back would flatter the extremes. The completeness of every beta
is reported, and a beta whose return rate is far below the others is flagged,
because a mean taken over the survivors of a 40 percent return rate is not
comparable to one taken over 95 percent.

Connecting the two halves of the paper
--------------------------------------
The charge arm finds that interface positions absorb less imposed charge than
bulk surface does, and that this depends on partner context. If that buffering
is doing protective work, complexes that buffer more should lose less
interface confidence. The Spearman correlation between the two is reported and
labelled **exploratory**: it is one number over 55 complexes, it was not the
reason for either experiment, and it is not powered to carry a claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from interface_charge.cli import add_common_arguments, banner, fail, resolve_config
from interface_charge.config import RESULTS_DIR
from interface_charge.paired import PAIR_KEYS, REFERENCE_BETA
from interface_charge.provenance import manifest_path_for, run_manifest
from interface_charge.statistics import (
    bootstrap_ci,
    spearman_bootstrap_ci,
    spearman_permutation_p,
)

#: The endpoint the claim rests on. One, chosen in advance.
PRIMARY_METRIC = "ipsae_d0res"

#: Reported alongside, in this order. Not corrected for multiplicity, and not
#: to be promoted to the headline if one of them happens to look better.
SECONDARY_METRICS = [
    "interface_ptm",
    "interface_pae",
    "interface_rmsd_a",
    "designed_chain_plddt",
]

#: Metrics where a larger number is a worse interface, so that "degraded" means
#: the same thing in every row of the output.
LOWER_IS_BETTER = {"interface_pae", "interface_rmsd_a"}

#: pLDDT floor for "the designed chain folded". 70 is the conventional boundary
#: between confident and low-confidence prediction and is used here because it
#: is conventional, not because anything in this data suggested it.
PLDDT_FLOOR = 70.0

#: A beta whose return rate falls this far below the best beta's is flagged.
#: Not a threshold for excluding anything, only for saying so out loud.
RETURN_RATE_GAP = 0.15


def load_metrics(path: Path) -> pd.DataFrame:
    """Read the collected AF2 table, refusing anything it cannot analyse."""
    table = pd.read_csv(path)
    required = {*PAIR_KEYS, "beta"}
    missing = required - set(table.columns)
    if missing:
        raise KeyError(
            f"{path} is missing column(s) {sorted(missing)}, which are the keys "
            "every design is paired on. Regenerate it with modal_app/collect.py."
        )
    if PRIMARY_METRIC not in table.columns:
        raise KeyError(
            f"{path} has no {PRIMARY_METRIC!r} column, which is the primary "
            "endpoint. Either the grid predates ipSAE or collection went wrong; "
            "either way this script will not silently fall back to a secondary "
            "metric and present it as the result."
        )
    return table


def completeness(table: pd.DataFrame, expected_complexes: int) -> dict:
    """Return rate per beta, and whether the betas are comparable to each other.

    A mean over the survivors of a 40 percent return rate and a mean over 95
    percent are not the same measurement, and nothing downstream of here can
    tell them apart.
    """
    counts = table.groupby("beta").size()
    rates = (counts / expected_complexes).sort_index()
    best = float(rates.max()) if len(rates) else 0.0
    depleted = sorted(float(b) for b, r in rates.items() if best - float(r) > RETURN_RATE_GAP)
    return {
        "expected_complexes_per_beta": expected_complexes,
        "returned_per_beta": {float(b): int(n) for b, n in counts.items()},
        "return_rate_per_beta": {float(b): round(float(r), 3) for b, r in rates.items()},
        "betas_materially_depleted": depleted,
        "comparable_across_beta": not depleted,
        "note": (
            "refolding failures concentrate in large complexes and extreme "
            "betas, so a depleted beta is biased towards its easy cases"
        ),
    }


def paired_change(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Change in ``metric`` against the same design lineage at beta = 0.

    Returns one row per (lineage, beta) that has a reference, with the reference
    rows themselves dropped: their delta is zero by construction and including
    them would drag every average towards no effect.
    """
    keys = list(PAIR_KEYS)
    reference = (
        table[table["beta"] == REFERENCE_BETA].set_index(keys)[metric].rename("reference").dropna()
    )
    joined = table[table["beta"] != REFERENCE_BETA].join(reference, on=keys, how="left")
    joined = joined.dropna(subset=["reference", metric])
    joined = joined.assign(delta=joined[metric] - joined["reference"])
    return joined[[*keys, "beta", metric, "reference", "delta"]]


def summarise_metric(
    table: pd.DataFrame,
    metric: str,
    seed: int,
    label: str,
) -> list[dict]:
    """Paired change per beta, with a bootstrap interval over complexes.

    Complexes are the resampling unit, not rows, because the rows within a
    complex are not independent of each other.
    """
    changes = paired_change(table, metric)
    lower_is_better = metric in LOWER_IS_BETTER
    rows: list[dict] = []

    for beta, block in changes.groupby("beta"):
        # One value per complex, so a complex with several replicates does not
        # count more than once in the bootstrap.
        per_complex = block.groupby("pdb_id")["delta"].mean()
        if per_complex.empty:
            continue
        interval = bootstrap_ci(per_complex.tolist(), seed=seed)
        degraded = (per_complex > 0) if lower_is_better else (per_complex < 0)
        rows.append(
            {
                "subset": label,
                "metric": metric,
                "beta": float(beta),
                "n_complexes": int(per_complex.size),
                "mean_change_vs_beta0": interval.estimate,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "excludes_zero": interval.excludes_zero,
                "n_degraded": int(degraded.sum()),
                "fraction_degraded": float(degraded.mean()),
                "lower_is_better": lower_is_better,
            }
        )
    return rows


def buffering_versus_survival(changes: pd.DataFrame, analysis_path: Path, seed: int) -> dict | None:
    """Do complexes that buffer more lose less interface confidence?

    Exploratory, and labelled as such wherever it is reported. The two
    experiments were designed independently and neither was powered for this.
    """
    if not analysis_path.is_file():
        return None
    per_complex = pd.read_csv(analysis_path)
    if not {"pdb_id", "partition", "buffering_ratio"}.issubset(per_complex.columns):
        return None

    wide = per_complex.pivot_table(index="pdb_id", columns="partition", values="buffering_ratio")
    if not {"interface", "surface"}.issubset(wide.columns):
        return None
    gap = (wide["interface"] - wide["surface"]).dropna()

    # Averaged over the non-zero betas: the question is whether a complex that
    # buffers holds up across the dial, not at one particular setting.
    survival = changes.groupby("pdb_id")["delta"].mean()
    # Both sides come from separately parsed CSVs, and an identifier read as a
    # string on one side and a number on the other intersects to nothing. That
    # failure is silent: it looks exactly like two tables with no complexes in
    # common, and the correlation is quietly dropped from the output.
    gap.index = gap.index.astype(str)
    survival.index = survival.index.astype(str)
    common = sorted(set(gap.index) & set(survival.index))
    if len(common) < 10:
        return {
            "n_complexes": len(common),
            "note": "too few complexes in both tables to correlate; not reported",
        }

    x = gap.loc[common].to_numpy()
    y = survival.loc[common].to_numpy()
    interval = spearman_bootstrap_ci(x, y, seed=seed)

    return {
        "status": "EXPLORATORY, not a pre-registered endpoint",
        "n_complexes": len(common),
        "spearman_rho": interval.estimate,
        "permutation_p": spearman_permutation_p(x, y, seed=seed),
        "ci": [interval.low, interval.high],
        "excludes_zero": interval.excludes_zero,
        "x": "interface minus surface buffering ratio (script 04)",
        "y": f"mean paired change in {PRIMARY_METRIC} across non-zero beta",
        "interpretation": (
            "a negative rho would mean complexes that buffer more (more negative "
            "gap) lose less interface confidence (less negative change), which is "
            "the direction the protective reading predicts"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/af2_metrics.csv"),
        help="Collected AF2 table from modal_app/collect.py",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("results/analysis_per_complex.csv"),
        help="Per-complex charge table from script 04, for the exploratory link",
    )
    parser.add_argument(
        "--plddt-floor",
        type=float,
        default=PLDDT_FLOOR,
        help="Designed-chain pLDDT at or above which a design counts as folded",
    )
    parser.add_argument("--seed", type=int, default=0)
    return add_common_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_config(args)
    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.metrics.is_file():
        fail(
            f"{args.metrics} not found. Run the grid and collect it first:\n"
            "  modal run modal_app/af2_multimer.py::prefetch\n"
            "  modal run modal_app/af2_multimer.py::run\n"
            "  modal volume get interface-charge-af2-results / ./modal_output\n"
            "  python modal_app/collect.py --from-dir modal_output"
        )

    try:
        table = load_metrics(args.metrics)
    except KeyError as exc:
        fail(str(exc))
        return 2

    n_complexes = int(table["pdb_id"].nunique())
    banner("12_af2_interface_survival", args, {"rows": len(table), "complexes": n_complexes})

    folded_available = "designed_chain_plddt" in table.columns
    if not folded_available:
        print(
            "note: no designed_chain_plddt column, so the fold-quality control "
            "cannot be run. The unfiltered result alone cannot separate a broken "
            "interface from a broken monomer.",
            file=sys.stderr,
        )

    with run_manifest(
        script=Path(__file__),
        parameters={
            "primary_endpoint": PRIMARY_METRIC,
            "primary_endpoint_rationale": (
                "ipTM normalises on total chain length, so it is not comparable "
                "across a set spanning 165 to 1257 residues; ipSAE takes d0 from "
                "the interface residue count"
            ),
            "secondary_endpoints": SECONDARY_METRICS,
            "comparison": f"within-complex paired change against beta = {REFERENCE_BETA}",
            "resampling_unit": "complex",
            "fold_quality_control": (
                f"designed-chain pLDDT >= {args.plddt_floor}, reported as a second "
                "subset rather than applied to the primary"
            ),
            "multiplicity": (
                "no correction applied; the primary endpoint is one metric fixed "
                "in advance and the secondaries are descriptive"
            ),
            "preregistered": ("this analysis was written and committed before the grid completed"),
        },
        seeds={"seed": args.seed},
        inputs=[p for p in (args.metrics, args.analysis) if p.is_file()],
    ) as manifest:
        subsets: list[tuple[str, pd.DataFrame]] = [("all_returned", table)]
        if folded_available:
            folded = table[table["designed_chain_plddt"] >= args.plddt_floor]
            subsets.append((f"designed_chain_plddt_ge_{args.plddt_floor:g}", folded))

        rows: list[dict] = []
        for label, subset in subsets:
            if subset.empty:
                continue
            for metric in [PRIMARY_METRIC, *SECONDARY_METRICS]:
                if metric in subset.columns:
                    rows.extend(summarise_metric(subset, metric, args.seed, label))

        if not rows:
            fail(
                "no beta had both a design and a beta = 0 reference for the same "
                "complex, so nothing could be paired. Check the grid completeness "
                "before reading anything into this."
            )

        summary_table = pd.DataFrame(rows)
        out_path = results_dir / "af2_interface_survival.csv"
        summary_table.to_csv(out_path, index=False)
        manifest.add_output(out_path)

        primary = summary_table[
            (summary_table["metric"] == PRIMARY_METRIC)
            & (summary_table["subset"] == "all_returned")
        ]
        folded_primary = summary_table[
            (summary_table["metric"] == PRIMARY_METRIC)
            & (summary_table["subset"] != "all_returned")
        ]

        changes = paired_change(table, PRIMARY_METRIC)
        exploratory = buffering_versus_survival(changes, args.analysis, args.seed)

        summary = {
            "primary_endpoint": PRIMARY_METRIC,
            "n_complexes": n_complexes,
            "completeness": completeness(table, n_complexes),
            "primary_all_returned": primary.to_dict(orient="records"),
            "primary_folded_only": folded_primary.to_dict(orient="records"),
            "fold_quality_reading": fold_quality_reading(primary, folded_primary),
            "exploratory_buffering_versus_survival": exploratory,
        }
        summary_path = results_dir / "af2_interface_survival_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        manifest.add_output(summary_path)
        manifest.note("summary", summary)
        manifest_target = manifest_path_for(out_path)

    manifest.write(manifest_target)

    print(f"\nwrote {out_path}", file=sys.stderr)
    for record in summary["primary_all_returned"]:
        mark = "*" if record["excludes_zero"] else " "
        print(
            f"  beta {record['beta']:+5.1f}  {PRIMARY_METRIC} change "
            f"{record['mean_change_vs_beta0']:+.4f} "
            f"[{record['ci_low']:+.4f}, {record['ci_high']:+.4f}]{mark}  "
            f"{record['n_degraded']}/{record['n_complexes']} degraded",
            file=sys.stderr,
        )
    print(f"\n{summary['fold_quality_reading']}", file=sys.stderr)
    if not summary["completeness"]["comparable_across_beta"]:
        print(
            f"\nWARNING: beta {summary['completeness']['betas_materially_depleted']} "
            "returned materially fewer jobs than the others, so their means are\n"
            "taken over easier complexes than the rest. Do not compare them "
            "directly without saying so.",
            file=sys.stderr,
        )
    print(f"manifest hash: {manifest.manifest_hash}", file=sys.stderr)
    return 0


def fold_quality_reading(primary: pd.DataFrame, folded: pd.DataFrame) -> str:
    """State which of the two readings the numbers support, in words.

    Written as a lookup over the four possible outcomes rather than composed
    after the fact, so the conclusion cannot drift to fit whichever result
    arrives.
    """
    if folded.empty:
        return (
            "No fold-quality control was possible, so a degraded interface here "
            "cannot be distinguished from a design that simply did not fold."
        )
    degrades_overall = bool(
        (primary["excludes_zero"] & (primary["mean_change_vs_beta0"] < 0)).any()
    )
    degrades_folded = bool((folded["excludes_zero"] & (folded["mean_change_vs_beta0"] < 0)).any())

    if degrades_overall and degrades_folded:
        return (
            "The interface degrades with charge among designs that folded, so the "
            "loss is not explained by monomer quality alone."
        )
    if degrades_overall and not degrades_folded:
        return (
            "The interface degrades overall but not among designs that folded, "
            "which points at the charge dial breaking monomers rather than "
            "interfaces. The paper should say so."
        )
    if not degrades_overall and degrades_folded:
        return (
            "No degradation overall but degradation within the folded subset, "
            "which is what selection on a collider looks like. Treat with care: "
            "conditioning on folding is not a neutral filter."
        )
    return (
        "No measurable degradation of the interface at any beta, in either "
        "subset. That is a result, and a useful one for a developability claim."
    )


if __name__ == "__main__":
    sys.exit(main())
