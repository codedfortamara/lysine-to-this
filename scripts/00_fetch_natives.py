#!/usr/bin/env python
"""Fetch the native structures named in test_set.csv from the RCSB.

Reads ``test_set.csv``, downloads one structure per ``pdb_id`` into
``data/native/``, and writes a manifest recording the digest of every file
fetched. Existing files are reused unless ``--overwrite`` is given, so the
script is cheap to re-run and safe to interrupt.

Nothing here is committed: ``data/native/`` is gitignored, because these files
are re-fetchable from the identifier and there is no reason to carry tens of
megabytes of public structures in the repository.

Requires network access to ``files.rcsb.org``. If that host is blocked by a
network policy, this script cannot work and says so plainly rather than
retrying forever.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interface_charge.cli import add_common_arguments, banner, fail, resolve_config
from interface_charge.config import NATIVE_DIR, RESULTS_DIR
from interface_charge.contracts import SchemaError, load_test_set
from interface_charge.provenance import manifest_path_for, run_manifest
from interface_charge.structures import StructureError, fetch_native


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("data/raw/test_set.csv"),
        help="Path to test_set.csv (default: data/raw/test_set.csv)",
    )
    parser.add_argument(
        "--native-dir",
        type=Path,
        default=None,
        help="Destination for downloaded structures (default: data/native/)",
    )
    return add_common_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(args)

    native_dir = args.native_dir or NATIVE_DIR
    results_dir = args.results_dir or RESULTS_DIR

    banner(
        "00_fetch_natives",
        args,
        {
            "test_set": args.test_set,
            "native_dir": native_dir,
            "format": config.structure.native_format,
        },
    )

    try:
        test_set = load_test_set(args.test_set, allow_extra_columns=args.allow_extra_columns)
    except SchemaError as exc:
        fail(str(exc))
        return 2

    pdb_ids = sorted(set(test_set["pdb_id"]))
    print(f"{len(pdb_ids)} unique structure(s) to fetch", file=sys.stderr)

    with run_manifest(
        script=Path(__file__),
        parameters={"structure": config.to_dict()["structure"], "n_pdb_ids": len(pdb_ids)},
        seeds={"random_seed": config.random_seed},
        inputs=[args.test_set],
    ) as manifest:
        fetched: list[str] = []
        failures: list[tuple[str, str]] = []

        for index, pdb_id in enumerate(pdb_ids, start=1):
            try:
                path = fetch_native(pdb_id, native_dir, config.structure, overwrite=args.overwrite)
            except StructureError as exc:
                failures.append((pdb_id, str(exc)))
                print(f"  [{index}/{len(pdb_ids)}] {pdb_id} FAILED", file=sys.stderr)
                continue
            manifest.add_output(path)
            fetched.append(pdb_id)
            print(f"  [{index}/{len(pdb_ids)}] {pdb_id} -> {path.name}", file=sys.stderr)

        manifest.note("fetched", fetched)
        manifest.note("n_fetched", len(fetched))
        manifest.note("failures", [{"pdb_id": p, "error": e[:500]} for p, e in failures])

        results_dir.mkdir(parents=True, exist_ok=True)
        manifest_target = manifest_path_for(results_dir / "00_fetch_natives")

    manifest.write(manifest_target)
    print(f"\nmanifest: {manifest_target}", file=sys.stderr)
    print(f"manifest hash: {manifest.manifest_hash}", file=sys.stderr)

    if failures:
        print(
            f"\n{len(failures)} structure(s) could not be fetched. First error:\n"
            f"  {failures[0][0]}: {failures[0][1]}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
