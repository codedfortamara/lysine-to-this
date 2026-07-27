#!/usr/bin/env python
"""Compare two directories of structures and report whether they are the same.

The RCSB revises deposited entries. A structure downloaded today is not
guaranteed to be byte-identical to the same identifier downloaded a year ago,
and the differences that do occur (re-refined coordinates, corrected chain
assignments, added or removed alternate conformations) are exactly the kind that
move a solvent accessible surface area or shift an interface residue set by one
or two members.

That matters here because two people are computing numbers from "the same"
structures. If those numbers ever disagree, the first question is whether the
inputs were identical, and this answers it in seconds rather than by
archaeology.

Standalone: no imports from this project, so it runs anywhere.

    python scripts/compare_structure_sets.py --left data/pdb --right ../zetadial/pdb

Exit status is 0 when every shared identifier matches, 1 otherwise, so it can
gate a pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def index(directory: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    """Map upper-case stem to path, for every structure file found at any depth."""
    found: dict[str, Path] = {}
    for suffix in suffixes:
        for path in directory.rglob(f"*{suffix}"):
            if path.is_file():
                found.setdefault(path.stem.upper(), path)
    return found


def coordinate_lines(path: Path) -> list[str]:
    """ATOM and HETATM records only, so headers and remarks do not count as differences.

    Two files can differ in deposition metadata while describing identical
    coordinates. Only the second kind of difference can change a result, so the
    two are reported separately.
    """
    lines = []
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                lines.append(line.rstrip("\n").rstrip())
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument(
        "--suffixes", nargs="+", default=[".pdb", ".ent", ".cif"], help="File types to compare"
    )
    parser.add_argument(
        "--show", type=int, default=10, help="How many differing identifiers to list"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for label, directory in (("--left", args.left), ("--right", args.right)):
        if not directory.is_dir():
            print(f"ERROR: {label} is not a directory: {directory}", file=sys.stderr)
            return 2

    left = index(args.left, tuple(args.suffixes))
    right = index(args.right, tuple(args.suffixes))

    shared = sorted(set(left) & set(right))
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))

    print(f"left  {args.left}: {len(left)} structure(s)")
    print(f"right {args.right}: {len(right)} structure(s)")
    print(f"shared identifiers: {len(shared)}\n")

    identical: list[str] = []
    same_coordinates: list[str] = []
    different: list[str] = []

    for pdb_id in shared:
        if sha256(left[pdb_id]) == sha256(right[pdb_id]):
            identical.append(pdb_id)
        elif coordinate_lines(left[pdb_id]) == coordinate_lines(right[pdb_id]):
            same_coordinates.append(pdb_id)
        else:
            different.append(pdb_id)

    print(f"byte identical:                 {len(identical)}")
    print(f"same coordinates, other diffs:  {len(same_coordinates)}")
    print(f"DIFFERENT coordinates:          {len(different)}")

    if only_left:
        print(f"\nonly in left ({len(only_left)}): {only_left[: args.show]}")
    if only_right:
        print(f"only in right ({len(only_right)}): {only_right[: args.show]}")

    if same_coordinates:
        print(
            f"\n{len(same_coordinates)} file(s) differ only outside the coordinate records "
            f"(headers, remarks): {same_coordinates[: args.show]}"
            "\nThese cannot change a computed result. Safe to ignore."
        )

    if different:
        print(
            f"\nWARNING: {len(different)} structure(s) have DIFFERENT coordinates: "
            f"{different[: args.show]}"
            "\nThe deposited entries have been revised between the two downloads. Use one"
            "\nset for everything and say which in the methods, otherwise two people will"
            "\ncompute different interfaces from what they both call the same structure."
        )
        return 1

    if not shared:
        print("\nNothing to compare: the two directories share no identifiers.")
        return 1

    print("\nEvery shared structure matches. The two sets are interchangeable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
