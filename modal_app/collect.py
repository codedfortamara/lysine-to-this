#!/usr/bin/env python
"""Collect AlphaFold2-Multimer metrics from the Modal Volume into results/.

Reads every ``metrics.json`` written by ``af2_multimer.predict`` and flattens
them into ``results/af2_metrics.csv``, with a run manifest beside it as
everywhere else in this project.

Two things this does that a plain download would not:

**It reports what is missing.** The expected job list is rebuilt from
``designs.csv`` and compared against what actually landed. Missing jobs are
listed by key in the manifest and on stderr. Refolding failures are not random:
large complexes and heavily charged designs fail more often, so quietly
analysing whatever came back would bias every downstream average towards the
easy cases. The count of missing jobs belongs in the paper's supplement.

**It refuses to invent columns.** A metric that a job did not produce arrives
as an empty cell, not as a zero and not as an imputed value.

Usage::

    modal volume get interface-charge-af2-results / ./modal_output
    python modal_app/collect.py --from-dir modal_output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from interface_charge.config import RESULTS_DIR
from interface_charge.paired import paired_deltas, pairing_report
from interface_charge.provenance import manifest_path_for, run_manifest

#: Metrics reported as a within-complex difference against beta = 0. Absolute
#: values of these vary far more between complexes than across the charge grid
#: within one, so the paired form is the one that answers the question.
PAIRED_METRICS = [
    "interface_ptm",
    "ipsae_d0res",
    "ipsae_d0dom",
    "ipsae_d0chn",
    "interface_pae",
    "interface_rmsd_a",
    "complex_ptm",
    "mean_plddt",
    "designed_chain_plddt",
]

#: Columns emitted in a fixed order, so the CSV is diffable across runs.
COLUMN_ORDER = [
    "pdb_id",
    "designed_chain",
    "beta",
    "replicate",
    "key",
    "complex_ptm",
    "interface_ptm",
    "ipsae_d0res",
    "ipsae_d0dom",
    "ipsae_d0chn",
    "ipsae_n_interface_residues_0to1",
    "ipsae_n_interface_residues_1to0",
    "ipsae_pae_cutoff_a",
    "interface_pae",
    "interface_rmsd_a",
    "mean_plddt",
    "designed_chain_plddt",
    "partner_chain_plddt",
    "msa_paired",
    "n_interface_residues_compared",
    "receptor_superposition_rmsd_a",
    "wall_clock_s",
    "num_recycles",
    "num_models",
    "random_seed",
    "predicted_pdb",
    "interface_rmsd_note",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        required=True,
        help="Local directory holding the downloaded contents of the results volume",
    )
    parser.add_argument("--designs", type=Path, default=Path("data/raw/designs.csv"))
    parser.add_argument("--test-set", type=Path, default=Path("data/raw/test_set.csv"))
    parser.add_argument(
        "--definitions", type=Path, default=Path("results/interface_definitions.json")
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-completeness-check",
        action="store_true",
        help=(
            "Do not rebuild the expected job list. Only use this when the input "
            "tables are not to hand; it removes the check that tells you what "
            "failed to refold."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.from_dir.is_dir():
        print(
            f"ERROR: {args.from_dir} is not a directory.\n"
            "Download the volume first:\n"
            "  modal volume get interface-charge-af2-results / ./modal_output",
            file=sys.stderr,
        )
        return 2

    metrics_files = sorted(args.from_dir.glob("*/metrics.json"))
    if not metrics_files:
        print(
            f"ERROR: no */metrics.json under {args.from_dir}. Nothing to collect.",
            file=sys.stderr,
        )
        return 2

    records: list[dict] = []
    unreadable: list[str] = []
    for path in metrics_files:
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            unreadable.append(f"{path}: {exc!r}")

    table = pd.DataFrame(records)
    ordered = [c for c in COLUMN_ORDER if c in table.columns]
    remainder = sorted(c for c in table.columns if c not in ordered)
    table = table[ordered + remainder]
    if {"pdb_id", "beta", "replicate"}.issubset(table.columns):
        table = table.sort_values(["pdb_id", "beta", "replicate"]).reset_index(drop=True)

    missing_keys: list[str] = []
    n_expected: int | None = None
    if not args.skip_completeness_check:
        try:
            from af2_multimer import build_job_list

            sys.path.insert(0, str(Path(__file__).resolve().parent))
            expected = build_job_list(args.designs, args.definitions, args.test_set)
            n_expected = len(expected)
            present = set(table["key"]) if "key" in table.columns else set()
            missing_keys = sorted(job.key for job in expected if job.key not in present)
        except (SystemExit, ImportError, OSError) as exc:
            print(
                f"note: completeness check skipped ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )

    inputs = [p for p in (args.designs, args.test_set, args.definitions) if p.is_file()]

    with run_manifest(
        script=Path(__file__),
        parameters={"from_dir": str(args.from_dir), "n_metrics_files": len(metrics_files)},
        seeds={},
        inputs=inputs,
    ) as manifest:
        out_path = results_dir / "af2_metrics.csv"
        table.to_csv(out_path, index=False)
        manifest.add_output(out_path)

        pairing: dict | None = None
        available = [m for m in PAIRED_METRICS if m in table.columns]
        if available and {"pdb_id", "designed_chain", "replicate", "beta"}.issubset(table.columns):
            paired = paired_deltas(table, available)
            paired_path = results_dir / "af2_paired_deltas.csv"
            paired.to_csv(paired_path, index=False)
            manifest.add_output(paired_path)
            pairing = pairing_report(table)
            pairing["metrics_paired"] = available
            pairing["n_rows_without_a_beta_zero_reference"] = int((~paired["has_reference"]).sum())
        else:
            print(
                "note: paired deltas skipped, the collected table lacks the "
                "pairing keys or every paired metric",
                file=sys.stderr,
            )

        manifest.note("pairing", pairing)
        manifest.note("n_collected", len(table))
        manifest.note("n_expected", n_expected)
        manifest.note("n_missing", len(missing_keys))
        manifest.note("missing_keys", missing_keys[:200])
        manifest.note("unreadable_files", unreadable)
        manifest_target = manifest_path_for(out_path)

    manifest.write(manifest_target)

    print(f"collected {len(table)} result(s) into {out_path}", file=sys.stderr)
    if pairing is not None:
        print(
            f"paired against beta = 0: {pairing['n_lineages_complete_across_beta']} "
            f"of {pairing['n_lineages']} design lineage(s) came back at every beta",
            file=sys.stderr,
        )
        if pairing["n_rows_without_a_beta_zero_reference"]:
            print(
                f"  {pairing['n_rows_without_a_beta_zero_reference']} row(s) have no "
                "beta = 0 reference, so their deltas are empty rather than computed "
                "against another complex",
                file=sys.stderr,
            )
    if unreadable:
        print(f"{len(unreadable)} file(s) could not be read:", file=sys.stderr)
        for line in unreadable[:10]:
            print(f"  {line}", file=sys.stderr)
    if missing_keys:
        print(
            f"\nWARNING: {len(missing_keys)} of {n_expected} expected job(s) are missing.\n"
            "Refolding failures are not random, so analysing only what came back\n"
            "biases results towards the easy complexes. Re-run the launcher to pick\n"
            "these up (the run is resumable), or record the shortfall explicitly.\n"
            f"First few: {missing_keys[:5]}",
            file=sys.stderr,
        )
    print(f"manifest hash: {manifest.manifest_hash}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
