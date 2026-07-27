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
    """Map upper-case stem to path, for every structure file found at any depth.

    Gzipped structures are matched too, since a downloaded archive often keeps
    them compressed, and their stem has the compression suffix stripped so that
    ``1ABC.pdb.gz`` and ``1ABC.pdb`` compare as the same identifier.
    """
    found: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        stem = path.stem if not name.lower().endswith(".gz") else Path(path.stem).stem
        if any(name.lower().endswith(s) or name.lower().endswith(s + ".gz") for s in suffixes):
            found.setdefault(stem.upper(), path)
    return found


def describe_directory(directory: Path, suffixes: tuple[str, ...]) -> str:
    """Explain what is actually in a directory that yielded no structures.

    An empty result has several quite different causes (wrong folder, an extra
    nesting level from an archive, files still showing as OneDrive placeholders,
    an unexpected extension) and they are indistinguishable from a bare count of
    zero. Listing what is really there names the cause immediately.
    """
    entries = sorted(directory.rglob("*"))
    files = [e for e in entries if e.is_file()]
    subdirs = [e for e in entries if e.is_dir()]

    lines = [
        f"  no files matching {list(suffixes)} (or their .gz forms) under {directory}",
        f"  found {len(files)} file(s) and {len(subdirs)} subdirector(y/ies) in total",
    ]
    if subdirs:
        lines.append(f"  subdirectories: {[d.name for d in subdirs[:8]]}")
    if files:
        lines.append(f"  example file names: {[f.name for f in files[:8]]}")
        extensions = sorted({f.suffix.lower() for f in files if f.suffix})
        lines.append(f"  extensions present: {extensions[:12]}")
    else:
        lines.append(
            "  the directory contains no files at all. If it is inside OneDrive, the "
            "contents may not be downloaded yet: right-click the folder and choose "
            "'Always keep on this device', then re-run."
        )
    return "\n".join(lines)


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

    args.left = Path(str(args.left).strip())
    args.right = Path(str(args.right).strip())

    for label, directory in (("--left", args.left), ("--right", args.right)):
        if not directory.is_dir():
            print(f"ERROR: {label} is not a directory: {directory}", file=sys.stderr)
            return 2

    suffixes = tuple(s.lower() for s in args.suffixes)
    left = index(args.left, suffixes)
    right = index(args.right, suffixes)

    for label, directory, found in (
        ("--left", args.left, left),
        ("--right", args.right, right),
    ):
        if not found:
            print(f"ERROR: {label} yielded no structures.", file=sys.stderr)
            print(describe_directory(directory, suffixes), file=sys.stderr)
            return 2

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
