#!/usr/bin/env python
"""Electrostatic complementarity across the interface, for every design.

Each design's sequence is threaded onto its native backbone, and the summed
charge product over contacting cross-chain residue pairs is computed. A
negative sum means opposite charges face each other.

This is deliberately **not** a Poisson-Boltzmann calculation. See the module
docstring of ``interface_charge.complementarity`` for the argument: it is a
heavy dependency the claim does not need, and this project does not cite a tool
it has not run end to end.

What varies and what does not
-----------------------------
The contact set comes from the native complex and is held fixed across beta,
exactly as the interface definition is. Only the per-residue charges change,
because only the sequence changes. That isolates the quantity of interest: the
degree to which the imposed charge shift makes apposed residues repel each
other, rather than a confounded mixture of that and a moving contact set.

If the AlphaFold2-Multimer predictions are available, running this again with
``--structures predicted`` answers the different and complementary question of
whether the *predicted* geometry is complementary. Report the two separately
and never in the same column.

Outputs
-------
``results/complementarity.csv``
    One row per design.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from interface_charge.cli import add_common_arguments, banner, fail, resolve_config
from interface_charge.complementarity import electrostatic_complementarity
from interface_charge.config import NATIVE_DIR, RESULTS_DIR
from interface_charge.contracts import SchemaError, check_cross_table, load_designs, load_test_set
from interface_charge.provenance import manifest_path_for, run_manifest
from interface_charge.structures import StructureError, extract_chain, load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--designs", type=Path, default=Path("data/raw/designs.csv"))
    parser.add_argument("--test-set", type=Path, default=Path("data/raw/test_set.csv"))
    parser.add_argument("--native-dir", type=Path, default=None)
    parser.add_argument(
        "--charge-definition",
        choices=("simple", "ph", "both"),
        default="both",
        help=(
            "Which charge definition supplies the per-residue charges. "
            "'both' emits a column per definition, separately named."
        ),
    )
    return add_common_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)

    native_dir = args.native_dir or NATIVE_DIR
    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    banner("03_complementarity", args, {"designs": args.designs, "native_dir": native_dir})

    try:
        designs = load_designs(args.designs, allow_extra_columns=args.allow_extra_columns)
        test_set = load_test_set(args.test_set, allow_extra_columns=args.allow_extra_columns)
        check_cross_table(designs, test_set)
    except SchemaError as exc:
        fail(str(exc))
        return 2

    chains_by_id = dict(zip(test_set["pdb_id"], test_set["chains"], strict=True))
    definitions = (
        ("simple", "ph") if args.charge_definition == "both" else (args.charge_definition,)
    )
    suffix = config.structure.native_format

    inputs = [args.designs, args.test_set]
    inputs += [
        native_dir / f"{pdb_id}.{suffix}"
        for pdb_id in sorted(set(designs["pdb_id"]))
        if (native_dir / f"{pdb_id}.{suffix}").is_file()
    ]

    with run_manifest(
        script=Path(__file__),
        parameters={
            "complementarity": config.to_dict()["complementarity"],
            "charge": config.to_dict()["charge"],
            "structure": config.to_dict()["structure"],
            "charge_definitions_emitted": list(definitions),
        },
        seeds={"random_seed": config.random_seed},
        inputs=inputs,
    ) as manifest:
        rows: list[dict] = []
        problems: list[str] = []
        # Parsing a structure is the expensive step, so each native is loaded
        # once and reused across every design derived from it.
        cache: dict[str, tuple] = {}

        for line, row in enumerate(designs.itertuples(), start=2):
            if len(row.designed_chain) != 1:
                problems.append(
                    f"line {line}: designed_chain names {len(row.designed_chain)} chains; "
                    "see open question 1 in the README"
                )
                continue

            designed = row.designed_chain[0]
            pair = chains_by_id.get(row.pdb_id)
            if pair is None:
                problems.append(f"line {line}: {row.pdb_id} absent from test_set.csv")
                continue

            partner = next((c for c in pair if c != designed), None)
            if partner is None:
                problems.append(
                    f"line {line}: {row.pdb_id} lists chains {list(pair)}, which do not "
                    f"provide a partner for designed chain {designed}"
                )
                continue

            if row.pdb_id not in cache:
                native_path = native_dir / f"{row.pdb_id}.{suffix}"
                if not native_path.is_file():
                    problems.append(f"line {line}: native structure missing: {native_path}")
                    continue
                try:
                    model = load_model(native_path, config.structure)
                    cache[row.pdb_id] = (
                        model,
                        extract_chain(model, designed, config.structure),
                        extract_chain(model, partner, config.structure),
                    )
                except StructureError as exc:
                    problems.append(f"line {line}: {row.pdb_id}: {exc}")
                    continue

            model, native_designed, native_partner = cache[row.pdb_id]

            try:
                threaded = native_designed.with_sequence(row.sequence)
            except StructureError as exc:
                problems.append(f"line {line}: {exc}")
                continue

            record = {
                "pdb_id": row.pdb_id,
                "designed_chain": designed,
                "partner_chain": partner,
                "beta": row.beta,
                "replicate": row.replicate,
                "structure_source": "native",
            }
            for definition in definitions:
                params = type(config.complementarity)(
                    contact_cutoff_a=config.complementarity.contact_cutoff_a,
                    charge_definition=definition,
                    weight=config.complementarity.weight,
                    count_neutral_pairs=config.complementarity.count_neutral_pairs,
                )
                result = electrostatic_complementarity(
                    model,
                    threaded,
                    native_partner,
                    params,
                    config.charge,
                    config.structure,
                )
                record.update(result.to_row())
            rows.append(record)

        if problems:
            shown = problems[:25]
            more = f"\n  ... and {len(problems) - 25} more" if len(problems) > 25 else ""
            fail(
                f"{len(problems)} design row(s) could not be scored:\n  "
                + "\n  ".join(shown)
                + more
            )

        table = pd.DataFrame(rows)
        out_path = results_dir / "complementarity.csv"
        table.to_csv(out_path, index=False)

        manifest.add_output(out_path)
        manifest.note("n_designs", len(table))
        manifest.note("n_structures", len(cache))
        manifest_target = manifest_path_for(out_path)

    manifest.write(manifest_target)
    print(
        f"\nwrote {out_path} ({len(table)} designs over {len(cache)} structures)", file=sys.stderr
    )
    print(f"manifest hash: {manifest.manifest_hash}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
