#!/usr/bin/env python
"""Join the tables and test where the imposed charge actually lands.

This answers question 1, and question 2 when AlphaFold2-Multimer metrics are
present. It runs without them, because the charge analysis needs no GPU and
should not wait on one.

The null model, stated before any data was seen
-----------------------------------------------
Interface residues are a minority of the surface, so in absolute terms they will
absorb less imposed charge than the bulk surface no matter what the model does.
Testing an interface slope against zero would therefore "confirm" buffering on
any dataset whatsoever, which is the single easiest way to get a wrong headline
out of this analysis.

The null is instead the partition's **residue-count share**. If charge were
placed without regard to structural context, a partition holding a fraction *f*
of the chain's residues would absorb a fraction *f* of any imposed charge shift,
giving a slope of exactly *f* when its charge shift is regressed on the whole
chain's. Buffering means a slope **below** that share.

The comparison that carries the claim
-------------------------------------
Core residues will always look strongly buffered, because inverse-folding models
do not put lysine in a hydrophobic core. That is a statement about protein
folding, not about interfaces, and it is reported but not interpreted.

The load-bearing comparison is **interface against non-interface surface**. Both
are solvent exposed, both are available to the model, and the only difference is
whether a residue sits in the binding site. Each partition's slope is divided by
its own residue share to give a buffering ratio, where one means "absorbs its
fair share" and below one means buffered, and the two ratios are compared as a
paired difference across complexes. Pairing matters because complexes differ
enormously in size and composition, and that between-complex variance would
otherwise swamp a real within-complex effect.

Statistics
----------
Bootstrap intervals throughout rather than analytic ones: none of these
quantities (a per-complex regression slope, a ratio of slopes, a median
interface pTM) has a distribution worth assuming at this sample size. Every
interval is seeded and the seed is recorded.

Outputs
-------
``results/analysis_per_complex.csv``
    One row per complex per partition: slope, residue share, buffering ratio.
``results/analysis_summary.json``
    The paired tests, effect sizes, intervals and sample sizes behind every
    claim, so a number in the paper can be traced to the test that produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from interface_charge.cli import add_common_arguments, banner, fail, resolve_config
from interface_charge.config import RESULTS_DIR
from interface_charge.provenance import manifest_path_for, run_manifest
from interface_charge.statistics import bootstrap_ci, paired_bootstrap_ci

PARTITIONS = ("interface", "surface", "core")

#: A complex needs at least this many distinct beta values for a slope to mean
#: anything. Three is already thin; two would just join two points.
MIN_BETAS_FOR_SLOPE = 3


def fit_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """Least-squares slope of y on x, or None if x does not vary."""
    if x.size < 2 or np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def per_complex_slopes(designs: pd.DataFrame, definition: str) -> pd.DataFrame:
    """Regress each partition's charge shift on the whole chain's, per complex.

    ``definition`` selects the charge definition, and it is carried into every
    output column name so that two runs under different definitions cannot be
    concatenated by accident.
    """
    rows: list[dict] = []

    for (pdb_id, chain), group in designs.groupby(["pdb_id", "designed_chain"], sort=True):
        if group["beta"].nunique() < MIN_BETAS_FOR_SLOPE:
            continue

        total_shift = group[f"charge_shift_total_{definition}"].to_numpy(dtype=float)
        n_total = float(group["n_residues_total"].iloc[0])
        if n_total <= 0:
            continue

        for partition in PARTITIONS:
            n_partition = float(group[f"n_residues_{partition}"].iloc[0])
            share = n_partition / n_total
            partition_shift = group[f"charge_shift_{partition}_{definition}"].to_numpy(dtype=float)
            slope = fit_slope(total_shift, partition_shift)
            if slope is None:
                continue

            rows.append(
                {
                    "pdb_id": pdb_id,
                    "designed_chain": chain,
                    "partition": partition,
                    "charge_definition": definition,
                    "n_residues_partition": int(n_partition),
                    "n_residues_total": int(n_total),
                    "residue_share": share,
                    "slope": slope,
                    # Below zero means the partition absorbs less than a
                    # structure-blind placement would give it.
                    "slope_minus_share": slope - share,
                    # One means "absorbs exactly its fair share". Undefined for
                    # an empty partition, which is left as NaN rather than
                    # filled, since a missing ratio is not a ratio of zero.
                    "buffering_ratio": (slope / share) if share > 0 else np.nan,
                    "n_betas": int(group["beta"].nunique()),
                }
            )

    return pd.DataFrame(rows)


def paired_partition_test(per_complex: pd.DataFrame, left: str, right: str, seed: int) -> dict:
    """Paired comparison of two partitions' buffering ratios across complexes."""
    pivot = per_complex.pivot_table(
        index=["pdb_id", "designed_chain"], columns="partition", values="buffering_ratio"
    ).dropna(subset=[left, right])

    if pivot.empty:
        return {"n": 0, "note": f"no complex has both a {left} and a {right} ratio"}

    a = pivot[left].to_numpy(dtype=float)
    b = pivot[right].to_numpy(dtype=float)
    difference = paired_bootstrap_ci(a.tolist(), b.tolist(), seed=seed)

    return {
        "n_complexes": len(a),
        f"mean_{left}_ratio": float(a.mean()),
        f"mean_{right}_ratio": float(b.mean()),
        "mean_paired_difference": difference.estimate,
        "ci_low": difference.low,
        "ci_high": difference.high,
        "excludes_zero": difference.excludes_zero,
        "n_complexes_where_left_is_lower": int((a < b).sum()),
        "interpretation": (
            f"{left} absorbs "
            + ("LESS" if difference.estimate < 0 else "MORE")
            + f" than {right} relative to its residue share"
            + (
                ", and the interval excludes zero"
                if difference.excludes_zero
                else ", but the interval includes zero so this is not resolved at this sample size"
            )
        ),
    }


def against_share_test(per_complex: pd.DataFrame, partition: str, seed: int) -> dict:
    """Is this partition's slope below its residue-count share?"""
    subset = per_complex[per_complex["partition"] == partition]
    if subset.empty:
        return {"n": 0}
    values = subset["slope_minus_share"].to_numpy(dtype=float)
    interval = bootstrap_ci(values.tolist(), seed=seed)
    return {
        "n_complexes": int(values.size),
        "mean_slope_minus_share": interval.estimate,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "excludes_zero": interval.excludes_zero,
        "n_below_share": int((values < 0).sum()),
        "interpretation": (
            f"{partition} absorbs "
            + ("less" if interval.estimate < 0 else "more")
            + " charge than a structure-blind placement would give it"
            + ("" if interval.excludes_zero else ", but the interval includes zero")
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--charge-partitions", type=Path, default=Path("results/charge_partitions.csv")
    )
    parser.add_argument("--complementarity", type=Path, default=Path("results/complementarity.csv"))
    parser.add_argument(
        "--af2-metrics",
        type=Path,
        default=Path("results/af2_metrics.csv"),
        help="Optional. Question 1 is analysed without it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return add_common_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.charge_partitions.is_file():
        fail(
            f"{args.charge_partitions} not found. Run scripts/02_partition_charge.py first; "
            "this script analyses its output."
        )

    designs = pd.read_csv(args.charge_partitions)
    banner("04_join_and_analyse", args, {"designs": len(designs)})

    definitions = sorted(
        {
            column.rsplit("charge_shift_total_", 1)[1]
            for column in designs.columns
            if column.startswith("charge_shift_total_")
        }
    )
    if not definitions:
        fail(
            "no charge_shift_total_* columns found. This file does not look like "
            "the output of 02_partition_charge.py."
        )

    inputs = [args.charge_partitions]
    complementarity = None
    if args.complementarity.is_file():
        complementarity = pd.read_csv(args.complementarity)
        inputs.append(args.complementarity)

    af2 = None
    if args.af2_metrics.is_file():
        af2 = pd.read_csv(args.af2_metrics)
        inputs.append(args.af2_metrics)

    with run_manifest(
        script=Path(__file__),
        parameters={
            "definitions": definitions,
            "min_betas_for_slope": MIN_BETAS_FOR_SLOPE,
            "partitions": list(PARTITIONS),
            "charge": config.to_dict()["charge"],
            "af2_metrics_present": af2 is not None,
            "complementarity_present": complementarity is not None,
        },
        seeds={"bootstrap_seed": args.seed},
        inputs=inputs,
    ) as manifest:
        frames = []
        summary: dict = {"definitions": {}}

        for definition in definitions:
            per_complex = per_complex_slopes(designs, definition)
            if per_complex.empty:
                summary["definitions"][definition] = {
                    "note": (
                        f"no complex had at least {MIN_BETAS_FOR_SLOPE} distinct beta "
                        "values, so no slope could be fitted"
                    )
                }
                continue
            frames.append(per_complex)

            summary["definitions"][definition] = {
                "n_complexes": int(per_complex["pdb_id"].nunique()),
                "against_residue_share": {
                    partition: against_share_test(per_complex, partition, args.seed)
                    for partition in PARTITIONS
                },
                "interface_versus_surface": paired_partition_test(
                    per_complex, "interface", "surface", args.seed
                ),
            }

        if not frames:
            fail(
                f"no slopes could be fitted. Every complex needs at least "
                f"{MIN_BETAS_FOR_SLOPE} distinct beta values."
            )

        per_complex_all = pd.concat(frames, ignore_index=True)

        if complementarity is not None:
            ec_columns = [c for c in complementarity.columns if c.startswith("ec_sum_product_")]
            summary["complementarity"] = {
                "n_designs": len(complementarity),
                "columns": ec_columns,
                "by_beta": {
                    column: complementarity.groupby("beta")[column].median().round(4).to_dict()
                    for column in ec_columns
                },
            }

        if af2 is not None:
            summary["af2"] = {
                "n_designs": len(af2),
                "by_beta": {
                    metric: af2.groupby("beta")[metric].median().round(4).to_dict()
                    for metric in ("interface_ptm", "interface_pae", "interface_rmsd_a")
                    if metric in af2.columns
                },
            }
        else:
            summary["af2"] = {
                "note": (
                    "AlphaFold2-Multimer metrics absent. Question 1 is complete without "
                    "them; question 2 needs modal_app/collect.py to have run."
                )
            }

        per_complex_path = results_dir / "analysis_per_complex.csv"
        summary_path = results_dir / "analysis_summary.json"
        per_complex_all.to_csv(per_complex_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")

        manifest.add_output(per_complex_path)
        manifest.add_output(summary_path)
        manifest.note("n_complexes", int(per_complex_all["pdb_id"].nunique()))
        manifest_target = manifest_path_for(per_complex_path)

    manifest.write(manifest_target)

    print(f"\nwrote {per_complex_path} and {summary_path}", file=sys.stderr)
    for definition, block in summary["definitions"].items():
        if "note" in block:
            print(f"\n[{definition}] {block['note']}", file=sys.stderr)
            continue
        print(f"\n=== {definition} ({block['n_complexes']} complexes) ===", file=sys.stderr)
        for partition in PARTITIONS:
            test = block["against_residue_share"][partition]
            print(
                f"  {partition:<9} slope minus share {test['mean_slope_minus_share']:+.4f} "
                f"[{test['ci_low']:+.4f}, {test['ci_high']:+.4f}]  "
                f"{'resolved' if test['excludes_zero'] else 'includes zero'}",
                file=sys.stderr,
            )
        comparison = block["interface_versus_surface"]
        if comparison.get("n_complexes"):
            print(
                f"\n  interface vs surface buffering ratio: "
                f"{comparison['mean_paired_difference']:+.4f} "
                f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}] "
                f"over {comparison['n_complexes']} complexes",
                file=sys.stderr,
            )
            print(f"  {comparison['interpretation']}", file=sys.stderr)

    print(f"\nmanifest hash: {manifest.manifest_hash}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
