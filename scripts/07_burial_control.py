#!/usr/bin/env python
"""Is interface buffering an interface effect, or just a burial effect?

The problem this exists to settle
---------------------------------
ProteinMPNN's sequence recovery is strongly graded by burial: roughly 90 to 95
percent in the deep core against about 35 percent on the surface, with interface
residues intermediate. A position the model is more confident about absorbs less
of an imposed logit bias. So an interface that takes up less charge than bulk
surface is exactly what burial alone predicts, with no interface-specific
mechanism required at all.

Reporting the raw interface-versus-surface gap without conditioning on burial
would therefore be reporting a restatement of ProteinMPNN's recovery gradient
and calling it a finding about interfaces.

The test
--------
Per-residue response is additive. For position *i*, fit the slope of that
position's charge against the whole-chain charge shift across the beta grid. Any
set of positions then has a slope equal to the sum of its members' slopes, and a
buffering ratio equal to that slope divided by the set's share of the chain.

The null resamples **which positions count as interface**, drawing sets of the
same size from non-interface surface residues, matched on relative SASA. If the
observed interface ratio sits inside that null distribution, the effect is
burial, not interface. If it sits outside, there is something about the
interface beyond how buried it is.

Matching is done by nearest available relative SASA without replacement, so the
resampled set has a burial profile as close to the real interface as the
available surface residues allow. The realised matching error is reported: a
null that could not actually match burial is not a control, and saying so is
more useful than a p-value computed over a bad match.

A second, simpler view is also reported: the per-residue slope regressed on
relative SASA, with and without an interface indicator, so the size of any
residual interface effect can be read directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from interface_charge.charge import per_residue_charge_simple
from interface_charge.cli import add_common_arguments, banner, fail, resolve_config
from interface_charge.config import RESULTS_DIR
from interface_charge.contracts import SchemaError, load_designs
from interface_charge.provenance import manifest_path_for, run_manifest
from interface_charge.statistics import bootstrap_ci

#: Resamples per chain. Ten thousand gives a stable tail at the resolution the
#: claim needs without making the script slow enough to discourage re-running.
N_PERMUTATIONS = 10_000

#: A chain needs this many distinct beta values before a per-residue slope means
#: anything.
MIN_BETAS = 3


def per_residue_slopes(
    sequences: list[str], total_shift: np.ndarray, native: str
) -> np.ndarray | None:
    """Slope of each position's charge against the whole-chain charge shift.

    Returns one slope per position. These are additive: the slope of any set of
    positions is the sum of its members' slopes, which is what makes the
    permutation test cheap and exact rather than approximate.
    """
    if np.ptp(total_shift) == 0:
        return None
    native_charge = per_residue_charge_simple(native).astype(float)
    per_design = np.array(
        [per_residue_charge_simple(s).astype(float) - native_charge for s in sequences]
    )
    centred_x = total_shift - total_shift.mean()
    denominator = float((centred_x**2).sum())
    if denominator == 0:
        return None
    return (per_design - per_design.mean(axis=0)).T @ centred_x / denominator


def matched_sample(
    rng: np.random.Generator,
    pool: np.ndarray,
    pool_rsasa: np.ndarray,
    target_rsasa: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Draw len(target) positions from ``pool``, matched on relative SASA.

    Nearest available value without replacement, with the pool shuffled first so
    that ties and the order of matching do not bias the draw. Returns the sample
    and the mean absolute matching error, which is what says whether the control
    is worth anything.
    """
    available = list(rng.permutation(len(pool)))
    chosen: list[int] = []
    errors: list[float] = []
    for wanted in target_rsasa:
        if not available:
            break
        distances = [abs(pool_rsasa[i] - wanted) for i in available]
        best = int(np.argmin(distances))
        errors.append(distances[best])
        chosen.append(available.pop(best))
    return pool[chosen], float(np.mean(errors)) if errors else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--designs", type=Path, default=Path("data/raw/designs.csv"))
    parser.add_argument(
        "--interface-definitions", type=Path, default=Path("results/interface_definitions.json")
    )
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=0)
    return add_common_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_config(args)  # validates any TOML override before work begins
    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.interface_definitions.is_file():
        fail(f"{args.interface_definitions} not found. Run scripts/01_define_interfaces.py first.")

    try:
        designs = load_designs(args.designs, allow_extra_columns=args.allow_extra_columns)
    except SchemaError as exc:
        fail(str(exc))
        return 2

    definitions = json.loads(args.interface_definitions.read_text())
    banner("07_burial_control", args, {"permutations": args.permutations})

    rows: list[dict] = []
    per_residue_frames: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []

    with run_manifest(
        script=Path(__file__),
        parameters={
            "permutations": args.permutations,
            "min_betas": MIN_BETAS,
            "matching": "nearest available relative SASA, without replacement",
            "charge_definition": "simple",
        },
        seeds={"seed": args.seed},
        inputs=[args.designs, args.interface_definitions],
    ) as manifest:
        rng = np.random.default_rng(args.seed)

        for (pdb_id, chain_tuple), group in designs.groupby(
            ["pdb_id", "designed_chain"], sort=True
        ):
            chain = chain_tuple[0] if isinstance(chain_tuple, tuple) else chain_tuple
            entry = definitions.get(pdb_id, {}).get("chains", {}).get(chain)
            if entry is None or "relative_sasa" not in entry:
                skipped.append((pdb_id, "no interface definition or no relative SASA"))
                continue
            if group["beta"].nunique() < MIN_BETAS:
                skipped.append((pdb_id, "too few beta values"))
                continue

            native = entry["sequence"]
            rsasa = np.array(entry["relative_sasa"], dtype=float)
            n_total = len(native)
            interface = np.array(sorted(entry["interface_positions"]), dtype=int)
            core = set(entry["core_positions"])
            surface = np.array(
                sorted(set(range(n_total)) - set(interface.tolist()) - core), dtype=int
            )

            if interface.size == 0 or surface.size < interface.size:
                skipped.append((pdb_id, "interface empty or surface pool too small to match"))
                continue

            ordered = group.sort_values("beta")
            sequences = ordered["sequence"].tolist()
            if any(len(s) != n_total for s in sequences):
                skipped.append((pdb_id, "design length differs from native"))
                continue

            native_total = float(per_residue_charge_simple(native).sum())
            total_shift = np.array(
                [float(per_residue_charge_simple(s).sum()) - native_total for s in sequences]
            )
            slopes = per_residue_slopes(sequences, total_shift, native)
            if slopes is None:
                skipped.append((pdb_id, "charge shift does not vary"))
                continue

            share = interface.size / n_total
            observed_ratio = float(slopes[interface].sum() / share)

            null_ratios = np.empty(args.permutations, dtype=float)
            match_errors = np.empty(args.permutations, dtype=float)
            for index in range(args.permutations):
                sample, error = matched_sample(rng, surface, rsasa[surface], rsasa[interface])
                null_ratios[index] = slopes[sample].sum() / (sample.size / n_total)
                match_errors[index] = error

            # Two-sided empirical p, with the observation counted so that p is
            # never zero, which would overstate what a finite null can support.
            centred = np.abs(null_ratios - null_ratios.mean())
            extreme = int((centred >= abs(observed_ratio - null_ratios.mean())).sum())
            p_value = (extreme + 1) / (args.permutations + 1)

            rows.append(
                {
                    "pdb_id": pdb_id,
                    "designed_chain": chain,
                    "n_residues": n_total,
                    "n_interface": int(interface.size),
                    "n_surface_pool": int(surface.size),
                    "mean_rsasa_interface": float(rsasa[interface].mean()),
                    "mean_rsasa_surface": float(rsasa[surface].mean()),
                    "observed_ratio": observed_ratio,
                    "null_mean_ratio": float(null_ratios.mean()),
                    "null_sd_ratio": float(null_ratios.std(ddof=1)),
                    "observed_minus_null": observed_ratio - float(null_ratios.mean()),
                    "p_permutation": p_value,
                    "mean_rsasa_match_error": float(match_errors.mean()),
                }
            )

            per_residue_frames.append(
                pd.DataFrame(
                    {
                        "pdb_id": pdb_id,
                        "position": np.arange(n_total),
                        "relative_sasa": rsasa,
                        "slope": slopes,
                        "is_interface": np.isin(np.arange(n_total), interface),
                        "is_core": [i in core for i in range(n_total)],
                    }
                )
            )

        if not rows:
            fail("no chain could be tested. First skips: " + str(skipped[:5]))

        table = pd.DataFrame(rows)
        per_residue = pd.concat(per_residue_frames, ignore_index=True)

        difference = bootstrap_ci(table["observed_minus_null"].tolist(), seed=args.seed)

        exposed = per_residue[~per_residue["is_core"]]
        correlation = float(np.corrcoef(exposed["relative_sasa"], exposed["slope"])[0, 1])
        design = np.column_stack(
            [
                np.ones(len(exposed)),
                exposed["relative_sasa"].to_numpy(float),
                exposed["is_interface"].to_numpy(float),
            ]
        )
        coefficients, *_ = np.linalg.lstsq(design, exposed["slope"].to_numpy(float), rcond=None)

        summary = {
            "n_chains": len(table),
            "permutations": args.permutations,
            "observed_minus_null": {
                "mean": difference.estimate,
                "ci_low": difference.low,
                "ci_high": difference.high,
                "excludes_zero": difference.excludes_zero,
            },
            "n_chains_p_below_0.05": int((table["p_permutation"] < 0.05).sum()),
            "mean_rsasa_interface": float(table["mean_rsasa_interface"].mean()),
            "mean_rsasa_surface": float(table["mean_rsasa_surface"].mean()),
            "mean_rsasa_match_error": float(table["mean_rsasa_match_error"].mean()),
            "per_residue": {
                "n_exposed_residues": len(exposed),
                "correlation_slope_with_rsasa": correlation,
                "regression_intercept": float(coefficients[0]),
                "regression_rsasa_coefficient": float(coefficients[1]),
                "regression_interface_coefficient": float(coefficients[2]),
            },
            "interpretation": (
                "The interface effect survives matching on burial"
                if difference.excludes_zero
                else "The interface effect does NOT survive matching on burial: once "
                "interface-sized residue sets are drawn from surface residues of the "
                "same relative SASA, they buffer the imposed charge just as much. The "
                "observed gap is a burial effect, not an interface effect."
            ),
        }

        table_path = results_dir / "burial_control.csv"
        residue_path = results_dir / "burial_control_per_residue.csv"
        summary_path = results_dir / "burial_control_summary.json"
        table.to_csv(table_path, index=False)
        per_residue.to_csv(residue_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")

        for path in (table_path, residue_path, summary_path):
            manifest.add_output(path)
        manifest.note("n_chains", len(table))
        manifest.note("n_skipped", len(skipped))
        manifest_target = manifest_path_for(table_path)

    manifest.write(manifest_target)

    print(f"\n=== burial-matched control, {summary['n_chains']} chains ===", file=sys.stderr)
    print(
        f"  mean relative SASA: interface {summary['mean_rsasa_interface']:.3f} "
        f"vs surface {summary['mean_rsasa_surface']:.3f} "
        f"(matching error {summary['mean_rsasa_match_error']:.4f})",
        file=sys.stderr,
    )
    block = summary["observed_minus_null"]
    print(
        f"\n  observed ratio minus burial-matched null: {block['mean']:+.4f} "
        f"[{block['ci_low']:+.4f}, {block['ci_high']:+.4f}]  "
        f"{'RESOLVED' if block['excludes_zero'] else 'includes zero'}",
        file=sys.stderr,
    )
    print(
        f"  chains individually significant at p < 0.05: "
        f"{summary['n_chains_p_below_0.05']} of {summary['n_chains']}",
        file=sys.stderr,
    )
    residue_block = summary["per_residue"]
    print(
        f"\n  per-residue, exposed positions only ({residue_block['n_exposed_residues']}):"
        f"\n    correlation of response with relative SASA: "
        f"{residue_block['correlation_slope_with_rsasa']:+.4f}"
        f"\n    regression coefficient on relative SASA:    "
        f"{residue_block['regression_rsasa_coefficient']:+.5f}"
        f"\n    regression coefficient on interface flag:   "
        f"{residue_block['regression_interface_coefficient']:+.5f}",
        file=sys.stderr,
    )
    print(f"\n  {summary['interpretation']}", file=sys.stderr)
    print(f"\nmanifest hash: {manifest.manifest_hash}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
