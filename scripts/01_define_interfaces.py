#!/usr/bin/env python
"""Define the interface of every native complex, under both definitions.

For each complex in ``test_set.csv`` this computes:

* the delta-SASA interface (primary): residues losing more than the threshold
  area on complexation;
* the contact interface (secondary): residues with a heavy atom within the
  cutoff of the partner chain;
* their agreement, per complex, as a Jaccard index;
* the three-way partition of each chain into interface, non-interface surface
  and buried core.

Everything is computed on the **native** structure and frozen. Downstream
scripts consume these sets rather than recomputing, which is what keeps the
partition boundaries fixed as beta changes. A trend in interface charge is only
interpretable if "interface" means the same set of residues at every beta.

Outputs
-------
``results/interfaces.csv``
    One row per (pdb_id, chain), with partition counts, buried surface area and
    the agreement between the two definitions.
``results/interface_definitions.json``
    The residue sets themselves, keyed by pdb_id and chain, for scripts 02 and
    03. Residues are recorded as ``chain:seqid:icode`` so that author numbering
    with insertion codes round-trips exactly.
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
from interface_charge.contracts import SchemaError, load_test_set
from interface_charge.interface import Partition, define_interface, interface_positions
from interface_charge.provenance import manifest_path_for, run_manifest, sha256_file
from interface_charge.structures import StructureError, extract_chain, load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test-set", type=Path, default=Path("data/raw/test_set.csv"))
    parser.add_argument("--native-dir", type=Path, default=None)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to these pdb_ids. Useful for a quick check before a full run.",
    )
    return add_common_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)

    native_dir = args.native_dir or NATIVE_DIR
    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    banner("01_define_interfaces", args, {"test_set": args.test_set, "native_dir": native_dir})

    try:
        test_set = load_test_set(args.test_set, allow_extra_columns=args.allow_extra_columns)
    except SchemaError as exc:
        fail(str(exc))
        return 2

    if args.only:
        wanted = {p.upper() for p in args.only}
        test_set = test_set[test_set["pdb_id"].isin(wanted)]
        if test_set.empty:
            fail(f"none of {sorted(wanted)} are present in {args.test_set}")

    rows: list[dict] = []
    definitions: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    inputs: list[Path] = [args.test_set]

    suffix = config.structure.native_format
    for row in test_set.itertuples():
        native_path = native_dir / f"{row.pdb_id}.{suffix}"
        if native_path.is_file():
            inputs.append(native_path)

    with run_manifest(
        script=Path(__file__),
        parameters={
            "interface": config.to_dict()["interface"],
            "structure": config.to_dict()["structure"],
        },
        seeds={"random_seed": config.random_seed},
        inputs=inputs,
    ) as manifest:
        for index, row in enumerate(test_set.itertuples(), start=1):
            pdb_id = row.pdb_id
            chain_a, chain_b = row.chains
            native_path = native_dir / f"{pdb_id}.{suffix}"

            if not native_path.is_file():
                failures.append((pdb_id, f"native structure missing: {native_path}"))
                continue

            try:
                model = load_model(native_path, config.structure)
                definition = define_interface(
                    model,
                    pdb_id=pdb_id,
                    chain_a=chain_a,
                    chain_b=chain_b,
                    structure_sha256=sha256_file(native_path),
                    interface_params=config.interface,
                    structure_params=config.structure,
                    source="native",
                )
            except (StructureError, ValueError) as exc:
                failures.append((pdb_id, str(exc)))
                print(f"  [{index}] {pdb_id} FAILED: {exc}", file=sys.stderr)
                continue

            per_chain: dict[str, dict] = {}
            for chain_id in (chain_a, chain_b):
                extract = extract_chain(model, chain_id, config.structure)
                classification = definition.classification_for(chain_id)
                interface_idx, core_idx = interface_positions(extract, classification)
                counts = classification.counts()

                rows.append(
                    {
                        "pdb_id": pdb_id,
                        "chain": chain_id,
                        "partner_chain": chain_b if chain_id == chain_a else chain_a,
                        "split": row.split,
                        "mmseqs_cluster_id": row.mmseqs_cluster_id,
                        "n_residues": len(extract.sequence),
                        "n_interface": counts[Partition.INTERFACE.value],
                        "n_surface": counts[Partition.SURFACE.value],
                        "n_core": counts[Partition.CORE.value],
                        "buried_surface_area_a2": classification.buried_surface_area_a2(),
                        "n_modified_residues_mapped": len(extract.substitutions),
                        **definition.agreement.to_row(),
                        "interface_source": definition.source,
                        "native_sha256": definition.structure_sha256,
                    }
                )

                per_chain[chain_id] = {
                    "sequence": extract.sequence,
                    "interface_positions": sorted(interface_idx),
                    "core_positions": sorted(core_idx),
                    "interface_residue_ids": sorted(
                        f"{r.chain}:{r.seqid}:{r.icode.strip()}"
                        for r in classification.ids_in(Partition.INTERFACE)
                    ),
                    "core_residue_ids": sorted(
                        f"{r.chain}:{r.seqid}:{r.icode.strip()}"
                        for r in classification.ids_in(Partition.CORE)
                    ),
                    "modified_residues_mapped": [
                        {"residue": str(rid), "from": three, "to": one}
                        for rid, three, one in extract.substitutions
                    ],
                }

            definitions[pdb_id] = {
                "chain_a": chain_a,
                "chain_b": chain_b,
                "source": definition.source,
                "native_sha256": definition.structure_sha256,
                "chains": per_chain,
                "agreement": definition.agreement.to_row(),
            }
            print(
                f"  [{index}/{len(test_set)}] {pdb_id} {chain_a}/{chain_b}: "
                f"delta-SASA {len(definition.delta_sasa_set)}, "
                f"contact {len(definition.contact_set)}, "
                f"jaccard {definition.agreement.jaccard:.3f}",
                file=sys.stderr,
            )

        if not rows:
            fail(
                "no complexes were processed successfully. "
                + (f"First failure: {failures[0][1]}" if failures else "")
            )

        table = pd.DataFrame(rows)
        csv_path = results_dir / "interfaces.csv"
        json_path = results_dir / "interface_definitions.json"
        table.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(definitions, indent=2, sort_keys=True) + "\n")

        manifest.add_output(csv_path)
        manifest.add_output(json_path)
        manifest.note("n_complexes", len(definitions))
        manifest.note("n_failures", len(failures))
        manifest.note("failures", [{"pdb_id": p, "error": e[:500]} for p, e in failures])
        manifest.note(
            "median_jaccard",
            float(table["interface_definition_jaccard"].median()) if not table.empty else None,
        )

        manifest_target = manifest_path_for(csv_path)

    manifest.write(manifest_target)
    print(f"\nwrote {csv_path} and {json_path}", file=sys.stderr)
    print(f"manifest hash: {manifest.manifest_hash}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
