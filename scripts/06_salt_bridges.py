#!/usr/bin/env python
"""Compare designed and native salt-bridge counts, with and without normalisation.

The upstream claim is that ProteinMPNN designs are, if anything, *more*
salt-bridged than native structures, supporting the framing that the work adds a
capability rather than repairing a weakness. The evidence is a raw count of
oppositely charged residue pairs whose Cbeta atoms lie within six angstroms.

Two things need separating before that count means what it is being asked to
mean.

**The Cbeta proxy is a constraint, not a shortcut.** A threaded design has no
side-chain atoms to measure between, because substituting a lysine for an
aspartate changes which atoms exist and where they point. Cbeta sits on the
backbone side of the first side-chain bond, so it survives the substitution.
Anything applied to a threaded design is therefore limited to the coarse
definition, and comparing design against native under the same coarse definition
is a fair comparison.

**The raw count is not.** Inverse-folding models emit more charged surface
residues than occur natively. More charged residues produce more pairs within
any fixed distance whether or not the model has placed a single one
thoughtfully, so a raw count cannot distinguish *more charge* from *better
arranged charge*, and only the second is a claim about the model.

This script reports both, plus the charge composition that drives the
difference, so the claim can be stated at whatever strength the data supports.
It also reports the cross-chain subset, since bridges spanning the interface are
the ones relevant to binding and are heavily outnumbered by intra-chain ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from interface_charge.cli import add_common_arguments, banner, fail, resolve_config
from interface_charge.config import NATIVE_DIR, RESULTS_DIR
from interface_charge.contracts import SchemaError, load_designs, load_test_set
from interface_charge.provenance import manifest_path_for, run_manifest
from interface_charge.saltbridge import charged_composition, count_salt_bridges_threaded
from interface_charge.statistics import paired_bootstrap_ci
from interface_charge.structures import StructureError, extract_chain, load_model


def beta_tag(beta: float) -> str:
    """Filename-safe rendering of a charge setting, matching the Modal job keys."""
    return f"{beta:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--designs", type=Path, default=Path("data/raw/designs.csv"))
    parser.add_argument("--test-set", type=Path, default=Path("data/raw/test_set.csv"))
    parser.add_argument("--native-dir", type=Path, default=None)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Which beta to compare against native. 0.0 is vanilla ProteinMPNN.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return add_common_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)
    native_dir = args.native_dir or NATIVE_DIR
    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        designs = load_designs(args.designs, allow_extra_columns=args.allow_extra_columns)
        test_set = load_test_set(args.test_set, allow_extra_columns=args.allow_extra_columns)
    except SchemaError as exc:
        fail(str(exc))
        return 2

    banner("06_salt_bridges", args, {"beta": args.beta, "designs": len(designs)})

    chains_by_id = dict(zip(test_set["pdb_id"], test_set["chains"], strict=True))
    suffix = config.structure.native_format
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    # Buried interface area, for the one normalisation whose denominator does
    # not move with the charge being added. Optional: without it the run still
    # reports the other three, it just cannot report density.
    interfaces_path = results_dir / "interfaces.csv"
    buried_area: dict[tuple[str, str], float] = {}
    if interfaces_path.is_file():
        interfaces = pd.read_csv(interfaces_path)
        needed = {"pdb_id", "chain", "buried_surface_area_a2"}
        if needed.issubset(interfaces.columns):
            buried_area = {
                (str(r.pdb_id), str(r.chain)): float(r.buried_surface_area_a2)
                for r in interfaces.itertuples()
                if float(r.buried_surface_area_a2) > 0
            }
        else:
            print(
                f"note: {interfaces_path} lacks {sorted(needed - set(interfaces.columns))}, "
                "so area-normalised density is not reported",
                file=sys.stderr,
            )
    else:
        print(
            f"note: {interfaces_path} not found, so area-normalised density is not "
            "reported. Run scripts/01_define_interfaces.py to enable it.",
            file=sys.stderr,
        )

    with run_manifest(
        script=Path(__file__),
        parameters={
            "salt_bridge": config.to_dict()["salt_bridge"],
            "structure": config.to_dict()["structure"],
            "beta_compared": args.beta,
            "definition": "cbeta_proxy (the only one computable on a threaded design)",
        },
        seeds={"bootstrap_seed": args.seed},
        inputs=[args.designs, args.test_set],
    ) as manifest:
        for row in designs[designs["beta"] == args.beta].itertuples():
            designed = row.designed_chain[0]
            pair = chains_by_id.get(row.pdb_id)
            if pair is None:
                continue
            partner = next((c for c in pair if c != designed), None)
            native_path = native_dir / f"{row.pdb_id}.{suffix}"
            if partner is None or not native_path.is_file():
                skipped.append((row.pdb_id, "missing partner or structure"))
                continue

            try:
                model = load_model(native_path, config.structure)
                native_designed = extract_chain(model, designed, config.structure)
                native_partner = extract_chain(model, partner, config.structure)
                threaded = native_designed.with_sequence(row.sequence)
            except StructureError as exc:
                skipped.append((row.pdb_id, str(exc)[:200]))
                continue

            record: dict = {"pdb_id": row.pdb_id, "designed_chain": designed, "beta": row.beta}
            for label, extract in (("native", native_designed), ("design", threaded)):
                for scope, cross in (("all", False), ("cross_chain", True)):
                    result = count_salt_bridges_threaded(
                        model,
                        extract,
                        native_partner,
                        config.salt_bridge,
                        config.structure,
                        cross_chain_only=cross,
                    )
                    record[f"{label}_{scope}_bridges"] = result.n_bridges
                    record[f"{label}_{scope}_per_opportunity"] = result.per_opportunity
                    record[f"{label}_{scope}_fraction_engaged"] = (
                        result.fraction_charged_residues_engaged
                    )
                    area = buried_area.get((row.pdb_id, designed))
                    if area is not None:
                        record[f"{label}_{scope}_per_1000_a2"] = result.per_1000_a2(area)
                composition = charged_composition(extract, config.salt_bridge.include_histidine)
                record[f"{label}_n_charged"] = composition["n_charged"]
                record[f"{label}_n_opportunities"] = composition["n_opportunities"]
            rows.append(record)

        if not rows:
            fail(f"no designs at beta={args.beta} could be scored")

        table = pd.DataFrame(rows)

        summary: dict = {"beta": args.beta, "n_complexes": len(table)}
        for scope in ("all", "cross_chain"):
            raw = paired_bootstrap_ci(
                table[f"design_{scope}_bridges"].tolist(),
                table[f"native_{scope}_bridges"].tolist(),
                seed=args.seed,
            )
            normalised = paired_bootstrap_ci(
                table[f"design_{scope}_per_opportunity"].tolist(),
                table[f"native_{scope}_per_opportunity"].tolist(),
                seed=args.seed,
            )
            summary[scope] = {
                "mean_design_bridges": float(table[f"design_{scope}_bridges"].mean()),
                "mean_native_bridges": float(table[f"native_{scope}_bridges"].mean()),
                "raw_difference": raw.estimate,
                "raw_ci": [raw.low, raw.high],
                "raw_excludes_zero": raw.excludes_zero,
                "normalised_difference": normalised.estimate,
                "normalised_ci": [normalised.low, normalised.high],
                "normalised_excludes_zero": normalised.excludes_zero,
            }

            density_column = f"design_{scope}_per_1000_a2"
            if density_column in table.columns:
                # Buried area is set by the native backbone and does not move
                # with the dial, unlike the opportunity denominator, which grows
                # quadratically with added charge. If the raw count rises and
                # density rises with it while per-opportunity stays flat, the
                # flat ratio is the denominator outrunning a real gain rather
                # than evidence of no gain.
                density = paired_bootstrap_ci(
                    table[density_column].tolist(),
                    table[f"native_{scope}_per_1000_a2"].tolist(),
                    seed=args.seed,
                )
                summary[scope]["density_difference_per_1000_a2"] = density.estimate
                summary[scope]["density_ci"] = [density.low, density.high]
                summary[scope]["density_excludes_zero"] = density.excludes_zero

        composition = paired_bootstrap_ci(
            table["design_n_charged"].tolist(), table["native_n_charged"].tolist(), seed=args.seed
        )
        summary["charge_composition"] = {
            "mean_design_charged_residues": float(table["design_n_charged"].mean()),
            "mean_native_charged_residues": float(table["native_n_charged"].mean()),
            "difference": composition.estimate,
            "ci": [composition.low, composition.high],
            "excludes_zero": composition.excludes_zero,
        }

        # The charge setting goes in the filename. Without it a run at one beta
        # silently destroys the run at another, since every other input is the
        # same. That is not hypothetical: a beta = 0 run overwrote the beta = 3
        # results here, and the loss was invisible because the replacement was a
        # perfectly valid table for a different question.
        table_path = results_dir / f"salt_bridges_beta{beta_tag(args.beta)}.csv"
        summary_path = results_dir / f"salt_bridges_beta{beta_tag(args.beta)}_summary.json"
        table.to_csv(table_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")

        manifest.add_output(table_path)
        manifest.add_output(summary_path)
        manifest.note("n_complexes", len(table))
        manifest.note("n_skipped", len(skipped))
        manifest_target = manifest_path_for(table_path)

    manifest.write(manifest_target)

    print(f"\n=== salt bridges at beta={args.beta}, {len(table)} complexes ===", file=sys.stderr)
    comp = summary["charge_composition"]
    print(
        f"\ncharged residues per chain: design {comp['mean_design_charged_residues']:.1f} "
        f"vs native {comp['mean_native_charged_residues']:.1f}, "
        f"difference {comp['difference']:+.2f} [{comp['ci'][0]:+.2f}, {comp['ci'][1]:+.2f}]",
        file=sys.stderr,
    )
    for scope in ("all", "cross_chain"):
        block = summary[scope]
        print(f"\n{scope}:", file=sys.stderr)
        print(
            f"  raw count        design {block['mean_design_bridges']:.2f} vs native "
            f"{block['mean_native_bridges']:.2f}, difference "
            f"{block['raw_difference']:+.2f} [{block['raw_ci'][0]:+.2f}, {block['raw_ci'][1]:+.2f}]"
            f"  {'RESOLVED' if block['raw_excludes_zero'] else 'includes zero'}",
            file=sys.stderr,
        )
        print(
            f"  per opportunity  difference {block['normalised_difference']:+.5f} "
            f"[{block['normalised_ci'][0]:+.5f}, {block['normalised_ci'][1]:+.5f}]"
            f"  {'RESOLVED' if block['normalised_excludes_zero'] else 'includes zero'}",
            file=sys.stderr,
        )
        if "density_difference_per_1000_a2" in block:
            print(
                f"  per 1000 A2      difference {block['density_difference_per_1000_a2']:+.4f} "
                f"[{block['density_ci'][0]:+.4f}, {block['density_ci'][1]:+.4f}]"
                f"  {'RESOLVED' if block['density_excludes_zero'] else 'includes zero'}",
                file=sys.stderr,
            )
    print(f"\nmanifest hash: {manifest.manifest_hash}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
