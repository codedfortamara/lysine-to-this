#!/usr/bin/env python
"""Partition the net charge of every design into interface, surface and core.

This is the script that answers question 1: when the charge dial turns, where
does the charge land?

For each design it takes the designed sequence, maps it onto the frozen native
interface and core index sets produced by ``01_define_interfaces.py``, and
reports the charge of each partition under **both** charge definitions, in
separately named columns that can never be averaged together by accident.

The fixed-backbone assumption
-----------------------------
ProteinMPNN redesigns a sequence onto a fixed backbone, so position *i* of a
designed sequence is the same structural position as residue *i* of the native
chain. Everything here rests on that. It is checked, not assumed: a design whose
sequence length differs from its native chain's is a hard error, because a
length mismatch means the residue index sets no longer refer to the residues
they were computed for, and every partition number would be quietly wrong.

Reconciliation against the collaborator's numbers
-------------------------------------------------
The ``net_charge_reported`` column is never used as an analysis input. It is
used once, here, as a check: the script computes the whole-chain charge under
every combination of definition, pKa set and terminus handling, and reports
which one reproduces the reported column. That answers open question 2 from the
README directly from the data rather than by asking. If nothing reproduces it
within tolerance, that is a finding worth knowing on day one rather than day
four.

Outputs
-------
``results/charge_partitions.csv``
    One row per design, with the partition counts and the charge of each
    partition under both definitions.
``results/charge_reconciliation.csv``
    One row per candidate charge definition, with the mean absolute error and
    the fraction of designs reproduced within tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from interface_charge.charge import net_charge_ph, net_charge_simple, partition_charge
from interface_charge.cli import add_common_arguments, banner, fail, resolve_config
from interface_charge.config import RESULTS_DIR
from interface_charge.contracts import SchemaError, check_cross_table, load_designs, load_test_set
from interface_charge.provenance import manifest_path_for, run_manifest

#: Candidate definitions tried during reconciliation. Each is
#: (label, callable taking a sequence and returning a charge).
RECONCILIATION_CANDIDATES = (
    ("simple_KR_minus_DE", lambda s: float(net_charge_simple(s))),
    ("ph7.4_emboss", lambda s: net_charge_ph(s, 7.4, "emboss")),
    ("ph7.4_bjellqvist", lambda s: net_charge_ph(s, 7.4, "bjellqvist")),
    ("ph7.4_emboss_no_termini", lambda s: net_charge_ph(s, 7.4, "emboss", include_termini=False)),
    ("ph7.4_emboss_no_cys_tyr", lambda s: net_charge_ph(s, 7.4, "emboss", include_cys_tyr=False)),
    ("ph7.0_emboss", lambda s: net_charge_ph(s, 7.0, "emboss")),
    ("ph7.0_bjellqvist", lambda s: net_charge_ph(s, 7.0, "bjellqvist")),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--designs", type=Path, default=Path("data/raw/designs.csv"))
    parser.add_argument("--test-set", type=Path, default=Path("data/raw/test_set.csv"))
    parser.add_argument(
        "--interface-definitions",
        type=Path,
        default=None,
        help="Output of 01_define_interfaces.py (default: results/interface_definitions.json)",
    )
    return add_common_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)

    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    definitions_path = args.interface_definitions or (results_dir / "interface_definitions.json")

    banner("02_partition_charge", args, {"designs": args.designs, "definitions": definitions_path})

    if not definitions_path.is_file():
        fail(
            f"interface definitions not found at {definitions_path}. "
            "Run scripts/01_define_interfaces.py first."
        )

    try:
        designs = load_designs(args.designs, allow_extra_columns=args.allow_extra_columns)
        test_set = load_test_set(args.test_set, allow_extra_columns=args.allow_extra_columns)
        check_cross_table(designs, test_set)
    except SchemaError as exc:
        fail(str(exc))
        return 2

    definitions = json.loads(definitions_path.read_text())

    with run_manifest(
        script=Path(__file__),
        parameters={
            "charge": config.to_dict()["charge"],
            "interface": config.to_dict()["interface"],
            "reconciliation_candidates": [name for name, _ in RECONCILIATION_CANDIDATES],
        },
        seeds={"random_seed": config.random_seed},
        inputs=[args.designs, args.test_set, definitions_path],
    ) as manifest:
        rows: list[dict] = []
        problems: list[str] = []

        for line, row in enumerate(designs.itertuples(), start=2):
            if len(row.designed_chain) != 1:
                problems.append(
                    f"line {line}: designed_chain names {len(row.designed_chain)} chains "
                    f"{list(row.designed_chain)} but a single sequence column. Which chain "
                    "the sequence belongs to is undetermined. Split into one row per "
                    "designed chain, or tell us the convention. This is open question 1 "
                    "in the README."
                )
                continue

            chain_id = row.designed_chain[0]
            entry = definitions.get(row.pdb_id)
            if entry is None:
                problems.append(
                    f"line {line}: no interface definition for {row.pdb_id}. "
                    "It may have failed in script 01."
                )
                continue

            chain_entry = entry["chains"].get(chain_id)
            if chain_entry is None:
                problems.append(
                    f"line {line}: {row.pdb_id} has no interface definition for chain "
                    f"{chain_id}. Defined chains: {sorted(entry['chains'])}"
                )
                continue

            native_sequence = chain_entry["sequence"]
            if len(row.sequence) != len(native_sequence):
                problems.append(
                    f"line {line}: {row.pdb_id} chain {chain_id} design is "
                    f"{len(row.sequence)} residues but the native chain is "
                    f"{len(native_sequence)}. ProteinMPNN is fixed-backbone, so these "
                    "must match. The residue index sets refer to native positions and "
                    "would otherwise point at the wrong residues."
                )
                continue

            partition = partition_charge(
                row.sequence,
                interface_idx=chain_entry["interface_positions"],
                core_idx=chain_entry["core_positions"],
                ph=config.charge.ph,
                pka_set=config.charge.pka_set,
                include_cys_tyr=config.charge.include_cys_tyr,
                include_termini=config.charge.include_termini,
            )
            native_partition = partition_charge(
                native_sequence,
                interface_idx=chain_entry["interface_positions"],
                core_idx=chain_entry["core_positions"],
                ph=config.charge.ph,
                pka_set=config.charge.pka_set,
                include_cys_tyr=config.charge.include_cys_tyr,
                include_termini=config.charge.include_termini,
            )
            densities = partition.charge_density_simple()

            record = {
                "pdb_id": row.pdb_id,
                "designed_chain": chain_id,
                "fixed_chain": ",".join(row.fixed_chain),
                "beta": row.beta,
                "replicate": row.replicate,
                "net_charge_reported": row.net_charge_reported,
                **partition.to_row(),
                # The charge *shift* relative to the native sequence of the same
                # chain is the fairer x-axis than beta itself, because beta is an
                # absolute logit shift whose charge effect is size dependent.
                # The paper makes this point in Limitations (iv).
                "charge_shift_interface_simple": partition.interface_simple
                - native_partition.interface_simple,
                "charge_shift_surface_simple": partition.surface_simple
                - native_partition.surface_simple,
                "charge_shift_core_simple": partition.core_simple - native_partition.core_simple,
                "charge_shift_total_simple": partition.total_simple - native_partition.total_simple,
                "charge_density_interface_simple": densities["interface"],
                "charge_density_surface_simple": densities["surface"],
                "charge_density_core_simple": densities["core"],
                "native_charge_total_simple": native_partition.total_simple,
                "n_mutations_from_native": sum(
                    1 for a, b in zip(row.sequence, native_sequence, strict=True) if a != b
                ),
            }
            rows.append(record)

        if problems:
            shown = problems[:25]
            more = f"\n  ... and {len(problems) - 25} more" if len(problems) > 25 else ""
            fail(
                f"{len(problems)} design row(s) could not be partitioned:\n  "
                + "\n  ".join(shown)
                + more
            )

        table = pd.DataFrame(rows)

        # -- reconciliation ------------------------------------------------
        reconciliation: list[dict] = []
        reported = designs["net_charge_reported"].to_numpy(dtype=float)
        tolerance = config.charge.reported_charge_tolerance
        for label, function in RECONCILIATION_CANDIDATES:
            computed = np.array([function(seq) for seq in designs["sequence"]], dtype=float)
            error = computed - reported
            reconciliation.append(
                {
                    "definition": label,
                    "mean_absolute_error": float(np.abs(error).mean()),
                    "median_absolute_error": float(np.median(np.abs(error))),
                    "max_absolute_error": float(np.abs(error).max()),
                    "mean_signed_error": float(error.mean()),
                    "fraction_within_tolerance": float((np.abs(error) <= tolerance).mean()),
                    "tolerance": tolerance,
                    "n_designs": len(error),
                }
            )

        reconciliation_table = pd.DataFrame(reconciliation).sort_values("mean_absolute_error")
        best = reconciliation_table.iloc[0]

        partitions_path = results_dir / "charge_partitions.csv"
        reconciliation_path = results_dir / "charge_reconciliation.csv"
        table.to_csv(partitions_path, index=False)
        reconciliation_table.to_csv(reconciliation_path, index=False)

        manifest.add_output(partitions_path)
        manifest.add_output(reconciliation_path)
        manifest.note("n_designs", len(table))
        manifest.note("best_matching_definition", str(best["definition"]))
        manifest.note("best_mean_absolute_error", float(best["mean_absolute_error"]))
        manifest.note(
            "reconciliation_conclusive",
            bool(best["fraction_within_tolerance"] > 0.95),
        )

        manifest_target = manifest_path_for(partitions_path)

    manifest.write(manifest_target)

    print(f"\nwrote {partitions_path} ({len(table)} designs)", file=sys.stderr)
    print(f"wrote {reconciliation_path}", file=sys.stderr)
    print("\nreconciliation against net_charge_reported:", file=sys.stderr)
    print(reconciliation_table.to_string(index=False), file=sys.stderr)
    if best["fraction_within_tolerance"] > 0.95:
        print(
            f"\n  -> net_charge_reported is reproduced by '{best['definition']}' "
            f"for {best['fraction_within_tolerance']:.1%} of designs.",
            file=sys.stderr,
        )
    else:
        print(
            "\n  -> WARNING: no candidate definition reproduces net_charge_reported "
            f"for more than 95 percent of designs (best is '{best['definition']}' at "
            f"{best['fraction_within_tolerance']:.1%}). Ask Mohammed how that column "
            "was computed before relying on any charge number.",
            file=sys.stderr,
        )
    print(f"\nmanifest hash: {manifest.manifest_hash}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
