#!/usr/bin/env python
"""Place the designs on the charge-hydropathy diagram, and test it against folding.

The upstream work reports that foldability collapses at large charge bias
without saying why. Uversky, Gillespie and Fink (2000) give a mechanism:
sequences with high mean net charge and low mean hydrophobicity fall outside the
region of that plane occupied by folded proteins. If the designs that stop
folding are the designs that cross that boundary, the collapse is explained
rather than merely observed, and it becomes predictable from sequence alone with
no structure prediction at all.

That last part is the useful bit. A boundary crossing costs nothing to compute
and needs no GPU, so it can screen a charge setting before any folding is
attempted.

The test
--------
With ``--fold-results`` pointing at a table of self-consistency RMSD per
(pdb_id, beta), this checks the correspondence directly: are designs on the
disorder-prone side of the boundary the ones that fail to fold? Reported as the
fold-success rate either side of the boundary, plus the rank correlation between
signed distance from the boundary and RMSD.

Without that table the script still places every design on the diagram, which is
the figure, but the correspondence is untested and it says so.

An honest limit worth reporting alongside
-----------------------------------------
The diagram uses the *absolute* mean net charge, so a design pushed equally far
positive and negative lands in the same place. Any asymmetry between the two
directions is therefore something this diagram cannot explain, and if the data
show one, that is a genuine gap rather than a detail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from interface_charge.cli import add_common_arguments, banner, fail, resolve_config
from interface_charge.config import RESULTS_DIR
from interface_charge.contracts import SchemaError, load_designs
from interface_charge.provenance import manifest_path_for, run_manifest
from interface_charge.statistics import bootstrap_ci
from interface_charge.uversky import place_on_diagram

#: Self-consistency RMSD below which a design is treated as having folded. Five
#: angstroms is the threshold the upstream work reports against.
FOLD_SUCCESS_RMSD_A = 5.0


#: Permutations for the within-beta significance test.
N_PERMUTATIONS: Final[int] = 10_000


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, without pulling in scipy for one number."""
    if a.size < 3:
        return float("nan")
    rank_a = pd.Series(a).rank().to_numpy()
    rank_b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def spearman_permutation_p(
    a: np.ndarray, b: np.ndarray, *, seed: int, n_permutations: int = N_PERMUTATIONS
) -> float:
    """Two-sided permutation p-value for a rank correlation.

    Shuffling one side breaks the pairing while keeping both marginal
    distributions intact, which is the right null here: the designs at a given
    beta really do have this spread of boundary distances and this spread of
    RMSDs, and the question is only whether they are paired.
    """
    observed = spearman(a, b)
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    shuffled = b.copy()
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        if abs(spearman(a, shuffled)) >= abs(observed):
            at_least_as_extreme += 1
    # Add-one correction, so a p-value is never reported as exactly zero when
    # the truth is only that it is below the resolution of the test.
    return (at_least_as_extreme + 1) / (n_permutations + 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--designs", type=Path, default=Path("data/raw/designs.csv"))
    parser.add_argument(
        "--fold-results",
        type=Path,
        default=None,
        help=(
            "Optional table with pdb_id, beta and scRMSD, to test whether crossing "
            "the boundary predicts the observed folding collapse."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return add_common_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_config(args)
    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        designs = load_designs(args.designs, allow_extra_columns=args.allow_extra_columns)
    except SchemaError as exc:
        fail(str(exc))
        return 2

    banner("08_uversky", args, {"designs": len(designs)})

    inputs = [args.designs]
    fold = None
    if args.fold_results is not None:
        if not args.fold_results.is_file():
            fail(f"--fold-results not found: {args.fold_results}")
        fold = pd.read_csv(args.fold_results)
        missing = {"pdb_id", "beta", "scRMSD"} - set(fold.columns)
        if missing:
            fail(f"--fold-results is missing column(s) {sorted(missing)}")
        inputs.append(args.fold_results)

    with run_manifest(
        script=Path(__file__),
        parameters={
            "boundary": "Uversky, Gillespie and Fink (2000): charge = 2.785 * hydropathy - 1.151",
            "hydropathy_scale": "Kyte and Doolittle (1982), normalised onto [0, 1]",
            "fold_success_rmsd_a": FOLD_SUCCESS_RMSD_A,
            "charge_definition": "simple",
        },
        seeds={"seed": args.seed},
        inputs=inputs,
    ) as manifest:
        rows = []
        for row in designs.itertuples():
            point = place_on_diagram(row.sequence)
            rows.append(
                {
                    "pdb_id": row.pdb_id,
                    "designed_chain": row.designed_chain[0],
                    "beta": row.beta,
                    **point.to_row(),
                }
            )
        table = pd.DataFrame(rows)

        by_beta = (
            table.groupby("beta")
            .agg(
                n=("pdb_id", "size"),
                mean_hydropathy=("mean_scaled_hydropathy", "mean"),
                mean_net_charge=("mean_net_charge", "mean"),
                mean_distance=("distance_from_boundary", "mean"),
                fraction_predicted_disordered=("predicted_disordered", "mean"),
            )
            .reset_index()
        )

        summary: dict = {
            "n_designs": len(table),
            "by_beta": by_beta.round(4).to_dict(orient="records"),
            "overall_fraction_predicted_disordered": float(table["predicted_disordered"].mean()),
        }

        if fold is not None:
            merged = table.merge(fold[["pdb_id", "beta", "scRMSD"]], on=["pdb_id", "beta"])
            if merged.empty:
                summary["fold_test"] = {"note": "no rows joined on (pdb_id, beta)"}
            else:
                merged["folded"] = merged["scRMSD"] < FOLD_SUCCESS_RMSD_A
                inside = merged[~merged["predicted_disordered"]]
                outside = merged[merged["predicted_disordered"]]

                block = {
                    "n_joined": len(merged),
                    "n_inside_boundary": len(inside),
                    "n_outside_boundary": len(outside),
                    "fold_rate_inside": float(inside["folded"].mean()) if len(inside) else None,
                    "fold_rate_outside": float(outside["folded"].mean()) if len(outside) else None,
                    "median_rmsd_inside": float(inside["scRMSD"].median()) if len(inside) else None,
                    "median_rmsd_outside": (
                        float(outside["scRMSD"].median()) if len(outside) else None
                    ),
                    "spearman_distance_vs_rmsd": spearman(
                        merged["distance_from_boundary"].to_numpy(float),
                        merged["scRMSD"].to_numpy(float),
                    ),
                }
                if len(inside) and len(outside):
                    difference = bootstrap_ci(
                        inside["folded"].astype(float).tolist(), seed=args.seed
                    )
                    outside_interval = bootstrap_ci(
                        outside["folded"].astype(float).tolist(), seed=args.seed
                    )
                    block["fold_rate_inside_ci"] = [difference.low, difference.high]
                    block["fold_rate_outside_ci"] = [outside_interval.low, outside_interval.high]

                # The confound that decides whether any of this is worth
                # reporting. Distance from the boundary rises with |beta| almost
                # by construction, so a correlation pooled across beta can be
                # nothing more than the dial setting read back out. If the
                # diagram carries sequence-level information the dial does not,
                # it has to show up *within* a fixed beta, where every design
                # received the same bias and only the sequences differ.
                block["within_beta"] = [
                    {
                        "beta": float(beta),
                        "n": len(group),
                        "fold_rate": float(group["folded"].mean()),
                        "median_rmsd": float(group["scRMSD"].median()),
                        "spearman_distance_vs_rmsd": spearman(
                            group["distance_from_boundary"].to_numpy(float),
                            group["scRMSD"].to_numpy(float),
                        ),
                        "permutation_p": spearman_permutation_p(
                            group["distance_from_boundary"].to_numpy(float),
                            group["scRMSD"].to_numpy(float),
                            seed=args.seed,
                        ),
                    }
                    for beta, group in merged.groupby("beta")
                ]
                within = [
                    entry["spearman_distance_vs_rmsd"]
                    for entry in block["within_beta"]
                    if entry["n"] >= 3 and np.isfinite(entry["spearman_distance_vs_rmsd"])
                ]
                block["mean_within_beta_spearman"] = float(np.mean(within)) if within else None
                block["confound_note"] = (
                    "spearman_distance_vs_rmsd is pooled across beta and is "
                    "inflated by the beta effect itself. mean_within_beta_spearman "
                    "is the part not attributable to the dial setting."
                )
                summary["fold_test"] = block

                merged.to_csv(results_dir / "uversky_with_folding.csv", index=False)
                manifest.add_output(results_dir / "uversky_with_folding.csv")

        # The diagram is blind to sign, so report the asymmetry it cannot see.
        positive = table[table["signed_net_charge"] > 0]
        negative = table[table["signed_net_charge"] < 0]
        summary["sign_asymmetry_the_diagram_cannot_see"] = {
            "n_positive": len(positive),
            "n_negative": len(negative),
            "mean_distance_positive": (
                float(positive["distance_from_boundary"].mean()) if len(positive) else None
            ),
            "mean_distance_negative": (
                float(negative["distance_from_boundary"].mean()) if len(negative) else None
            ),
            "note": (
                "The diagram uses absolute net charge, so equally positive and "
                "negative designs coincide on it. Any difference in outcome between "
                "the two directions is therefore outside what this boundary explains."
            ),
        }

        table_path = results_dir / "uversky.csv"
        summary_path = results_dir / "uversky_summary.json"
        table.to_csv(table_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
        manifest.add_output(table_path)
        manifest.add_output(summary_path)
        manifest.note("n_designs", len(table))
        manifest_target = manifest_path_for(table_path)

    manifest.write(manifest_target)

    print(f"\n=== charge-hydropathy placement, {len(table)} designs ===", file=sys.stderr)
    print(by_beta.round(4).to_string(index=False), file=sys.stderr)
    if fold is not None and "fold_test" in summary and "n_joined" in summary["fold_test"]:
        block = summary["fold_test"]
        print(
            f"\n=== does crossing the boundary predict folding failure? "
            f"({block['n_joined']} designs joined) ===",
            file=sys.stderr,
        )
        print(
            f"  inside the boundary  (n={block['n_inside_boundary']:>3}): "
            f"{block['fold_rate_inside']:.1%} fold under {FOLD_SUCCESS_RMSD_A} A, "
            f"median RMSD {block['median_rmsd_inside']:.2f}",
            file=sys.stderr,
        )
        print(
            f"  outside the boundary (n={block['n_outside_boundary']:>3}): "
            f"{block['fold_rate_outside']:.1%} fold under {FOLD_SUCCESS_RMSD_A} A, "
            f"median RMSD {block['median_rmsd_outside']:.2f}",
            file=sys.stderr,
        )
        print(
            f"  Spearman, distance from boundary vs RMSD: "
            f"{block['spearman_distance_vs_rmsd']:+.3f} (pooled, inflated by beta)",
            file=sys.stderr,
        )
        print("\n  within each beta, where only the sequences differ:", file=sys.stderr)
        for entry in block["within_beta"]:
            print(
                f"    beta {entry['beta']:+.1f} (n={entry['n']:>3}): "
                f"{entry['fold_rate']:6.1%} fold, median RMSD {entry['median_rmsd']:5.2f}, "
                f"Spearman {entry['spearman_distance_vs_rmsd']:+.3f} "
                f"(p={entry['permutation_p']:.4f})",
                file=sys.stderr,
            )
        if block["mean_within_beta_spearman"] is not None:
            print(
                f"  mean within-beta Spearman: {block['mean_within_beta_spearman']:+.3f} "
                f"(the part not attributable to the dial setting)",
                file=sys.stderr,
            )
    print(f"\nmanifest hash: {manifest.manifest_hash}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
