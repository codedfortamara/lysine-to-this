#!/usr/bin/env python
"""Interface backbone RMSD against the native complex, computed locally.

Why this is a separate script rather than part of the prediction
----------------------------------------------------------------
``predict`` has always tried to compute this inside the GPU container, reading
the native from ``/results/natives/{pdb_id}.pdb``. Nothing has ever written that
directory. So ``native_path.is_file()`` was false every single time, every row
carried ``interface_rmsd_a`` as null, and the accompanying note said the native
was unavailable. A column that is present and entirely empty reads as data.

Uploading 55 structures to a GPU container to fix that would be the wrong
repair. The calculation needs no GPU: it is a superposition and a distance. The
natives are already in ``data/native/`` and the predicted structures come down
with the results volume, so it belongs here, where it costs nothing and can be
rerun freely when the definition changes.

What it measures
----------------
Superpose the prediction onto the native **using the partner chain only**, then
take the backbone RMSD over the native interface residues of the designed chain.
Superposing on the partner rather than on the whole complex is what makes this a
measure of interface displacement rather than of global drift: if the designed
chain has swung away from its partner, that is precisely what should show up,
and a whole-complex superposition would average it away.

Residues are paired between native and prediction by sequence alignment, never
by residue number. The natives carry author numbering, which starts wherever the
depositor chose: 1FS2 chain A begins at 105, 1O9S at 117. ColabFold numbers its
output 1..N. Pairing on the number compares different residues and returns a
plausible answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))

import pandas as pd

from interface_charge.cli import add_common_arguments, banner, fail, resolve_config
from interface_charge.config import NATIVE_DIR, RESULTS_DIR
from interface_charge.provenance import manifest_path_for, run_manifest

#: Columns this script fills in. Written back into the metrics table so that the
#: analysis reads one file rather than joining two.
RMSD_COLUMNS = [
    "interface_rmsd_a",
    "interface_rmsd_note",
    "receptor_superposition_rmsd_a",
    "n_interface_residues_compared",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        required=True,
        help="Downloaded contents of the results volume, holding one folder per job",
    )
    parser.add_argument("--metrics", type=Path, default=Path("results/af2_metrics.csv"))
    parser.add_argument("--natives", type=Path, default=None, help="Default: data/native/")
    parser.add_argument("--designs", type=Path, default=Path("data/raw/designs.csv"))
    parser.add_argument("--test-set", type=Path, default=Path("data/raw/test_set.csv"))
    parser.add_argument(
        "--definitions", type=Path, default=Path("results/interface_definitions.json")
    )
    return add_common_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_config(args)
    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    natives_dir = args.natives or NATIVE_DIR

    if not args.metrics.is_file():
        fail(f"{args.metrics} not found. Run modal_app/collect.py first.")
    if not args.from_dir.is_dir():
        fail(
            f"{args.from_dir} is not a directory. Download the results volume first:\n"
            "  modal volume get --force interface-charge-af2-results / C:\\af2out"
        )
    if not natives_dir.is_dir():
        fail(f"{natives_dir} not found. The native structures are needed to superpose against.")

    from af2_multimer import build_job_list, interface_rmsd

    table = pd.read_csv(args.metrics)
    jobs = {job.key: job for job in build_job_list(args.designs, args.definitions, args.test_set)}

    banner("14_interface_rmsd", args, {"rows": len(table), "natives": natives_dir})

    computed: list[dict] = []
    reasons: dict[str, int] = {}

    def skip(reason: str) -> dict:
        reasons[reason] = reasons.get(reason, 0) + 1
        return {"interface_rmsd_a": None, "interface_rmsd_note": reason}

    for row in table.itertuples():
        key = str(row.key)
        job = jobs.get(key)
        if job is None:
            computed.append(skip("job key is not in the current design table"))
            continue

        native_path = natives_dir / f"{job.pdb_id}.pdb"
        if not native_path.is_file():
            computed.append(skip(f"no native structure at {native_path.name}"))
            continue

        predicted_name = getattr(row, "predicted_pdb", None)
        if not isinstance(predicted_name, str) or not predicted_name:
            computed.append(skip("this job recorded no predicted structure"))
            continue

        predicted_path = args.from_dir / key / predicted_name
        if not predicted_path.is_file():
            computed.append(skip(f"predicted structure missing from the download: {key}"))
            continue

        try:
            computed.append(
                interface_rmsd(
                    native_path=native_path,
                    predicted_path=predicted_path,
                    designed_chain=job.designed_chain,
                    # The same deterministic order predict used, so the A, B
                    # labels ColabFold assigned map back onto the right chains.
                    chain_order=sorted(job.chains),
                )
            )
        except Exception as exc:  # reported per row rather than aborting the batch
            computed.append(skip(f"{type(exc).__name__}: {exc}"))

    filled = pd.DataFrame(computed).reindex(columns=RMSD_COLUMNS)
    updated = table.drop(columns=[c for c in RMSD_COLUMNS if c in table.columns])
    updated = pd.concat([updated.reset_index(drop=True), filled], axis=1)

    n_computed = int(updated["interface_rmsd_a"].notna().sum())

    with run_manifest(
        script=Path(__file__),
        parameters={
            "superposition": "partner chain only, so the number reports interface displacement",
            "residue_pairing": "sequence alignment with an identity check, not residue number",
            "atoms": "backbone N, CA, C, O",
            "interface_definition_source": "the native complex, from script 01",
            "n_rows": len(updated),
            "n_computed": n_computed,
            "skip_reasons": reasons,
        },
        seeds={},
        inputs=[args.metrics, args.designs, args.test_set, args.definitions],
    ) as manifest:
        out_path = results_dir / "af2_metrics.csv"
        updated.to_csv(out_path, index=False)
        manifest.add_output(out_path)
        manifest.note("n_computed", n_computed)
        manifest.note("skip_reasons", reasons)
        manifest_target = manifest_path_for(out_path)

    manifest.write(manifest_target)

    print(f"\ninterface RMSD computed for {n_computed} of {len(updated)} row(s)", file=sys.stderr)
    if n_computed:
        values = updated["interface_rmsd_a"].dropna()
        print(
            f"  median {values.median():.2f} A, range {values.min():.2f} to {values.max():.2f}",
            file=sys.stderr,
        )
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        print(f"  {count:>3} skipped: {reason}", file=sys.stderr)
    if not n_computed:
        print(
            "\nNothing was computed. Every reason is printed above rather than "
            "left as an empty\ncolumn, because a column that is present and "
            "entirely null reads as data.",
            file=sys.stderr,
        )
    print(f"manifest hash: {manifest.manifest_hash}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
