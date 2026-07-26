"""Build a trimmed two-chain structural fixture for the test suite.

Why there is a script rather than just a file
---------------------------------------------
A committed binary-ish fixture with no recorded provenance is exactly the kind
of thing this project refuses to tolerate elsewhere, so the fixture is
reproducible: this script records where the structure came from and what was
stripped, and re-running it must produce a byte-identical file.

The intended fixture is 1BRS, barnase-barstar, the canonical highly charged
protein-protein interface. It could not be downloaded in the environment this
repository was scaffolded in, because ``files.rcsb.org`` was blocked by the
network egress policy. The committed fixture is therefore 7DDO instead, human
ACE2 chain A with SARS-CoV-2 RBD chain C, taken from the Biopython test corpus
distributed on PyPI. That is a genuine two-chain protein-protein complex, and
RBD is one of the paper's own three BindCraft targets, so it exercises the real
code path on a relevant structure.

To add 1BRS once the RCSB is reachable::

    python tests/fixtures/build_fixture.py --pdb-id 1BRS --chains A D

The test suite picks up any fixture matching ``*_fixture.pdb`` automatically and
will then run against both.

What is stripped and why
------------------------
Waters, ions and ligands are removed, because including them would make a
charge partition depend on crystallisation conditions. Hydrogens are removed,
because whether a deposited structure carries them is an artefact of the
experimental method. Alternate conformations other than the first are removed,
because a duplicated residue identity breaks the residue index map. All three
choices match what ``structures.submodel`` does at run time, so the fixture
exercises the same atom selection the pipeline uses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser, Select

FIXTURE_DIR = Path(__file__).resolve().parent
RCSB_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"


class TrimSelect(Select):
    """Keep the named chains, polymer residues only, heavy atoms, first altloc."""

    def __init__(self, chains: tuple[str, ...]) -> None:
        self.chains = chains

    def accept_chain(self, chain) -> bool:
        return chain.id in self.chains

    def accept_residue(self, residue) -> bool:
        return residue.id[0] == " "

    def accept_atom(self, atom) -> bool:
        if atom.element == "H":
            return False
        return atom.get_altloc() in (" ", "A")


def build(source: Path, out_path: Path, chains: tuple[str, ...]) -> dict[str, object]:
    """Trim ``source`` to ``chains`` and write ``out_path``. Returns a provenance dict."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(out_path.stem, source)

    present = {chain.id for chain in structure[0]}
    missing = [c for c in chains if c not in present]
    if missing:
        raise SystemExit(
            f"chain(s) {missing} not present in {source}. Chains found: {sorted(present)}"
        )

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), select=TrimSelect(chains))

    trimmed = parser.get_structure(out_path.stem, out_path)
    counts = {
        chain.id: {
            "residues": sum(1 for _ in chain),
            "atoms": sum(1 for _ in chain.get_atoms()),
        }
        for chain in trimmed[0]
    }

    return {
        "source": str(source),
        "output": out_path.name,
        "chains": list(chains),
        "counts": counts,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stripped": [
            "waters and heteroatoms",
            "hydrogens",
            "alternate conformations after the first",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source",
        type=Path,
        help="Local structure file to trim. If omitted, --pdb-id is downloaded from the RCSB.",
    )
    parser.add_argument("--pdb-id", help="RCSB identifier to download, for example 1BRS")
    parser.add_argument(
        "--chains",
        nargs=2,
        metavar=("CHAIN_A", "CHAIN_B"),
        required=True,
        help="The two author chain identifiers to keep",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output path. Defaults to tests/fixtures/<pdb_id lower>_fixture.pdb",
    )
    args = parser.parse_args(argv)

    if not args.source and not args.pdb_id:
        parser.error("one of --source or --pdb-id is required")

    source = args.source
    if source is None:
        import urllib.request

        url = RCSB_TEMPLATE.format(pdb_id=args.pdb_id.upper())
        source = FIXTURE_DIR / f"_{args.pdb_id.upper()}_download.pdb"
        print(f"downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                source.write_bytes(response.read())
        except OSError as exc:
            raise SystemExit(
                f"could not download {url}: {exc!r}\n"
                "If this is a network policy denial rather than a transient "
                "failure, run this script somewhere that can reach files.rcsb.org "
                "and commit the resulting fixture."
            ) from exc

    stem = (args.pdb_id or source.stem).lower()
    out_path = args.out or FIXTURE_DIR / f"{stem}_fixture.pdb"

    provenance = build(source, out_path, tuple(args.chains))

    manifest_path = out_path.with_suffix(".provenance.json")
    manifest_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    print(json.dumps(provenance, indent=2, sort_keys=True))
    print(f"\nwrote {out_path} and {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
