#!/usr/bin/env python
"""Convert the collaborator's ``designs_for_folding.json`` into the data contract.

The upstream export is self-contained by design: it stores a chain's native
sequence, its designed sequences across the beta grid, and its alpha-carbon
coordinates, with no chain identifier and no reference back to the deposited
structure. That is everything needed to fold one chain in isolation and score
self-consistency, and two things short of what an interface analysis needs.

**Which chain is it?** The export records only ``chain_len``. This script
recovers the author chain identifier by matching the stored ``native_seq``
against every chain of the deposited structure. The match must be unique and
above an identity threshold, otherwise it raises: guessing a chain would silently
attribute an interface to the wrong side of the complex, which inverts the
result rather than degrading it.

**What about the partner?** Its designed sequence was not kept, so the only
faithful reading of this file is the fixed-partner experiment: designed chain
against native partner. That is a legitimate and arguably more useful
experiment, since it matches binder design against an unmodified target, but it
is not what the upstream sampling did, where every chain was redesigned. Every
row emitted here is marked accordingly, and
``scripts/10_regenerate_designs.py`` produces the both-chains-designed variant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from interface_charge.charge import net_charge_simple
from interface_charge.cli import banner, fail
from interface_charge.config import DEFAULT_CONFIG, NATIVE_DIR, RAW_DIR
from interface_charge.interface import choose_contacting_partner
from interface_charge.provenance import manifest_path_for, run_manifest
from interface_charge.structures import (
    StructureError,
    chains_present,
    extract_chain,
    load_model,
)

#: Minimum fraction of positions that must agree for a chain to be accepted as
#: the one the export describes. Set high because the comparison is against the
#: chain's own native sequence, so a correct match should be near perfect. The
#: slack allows for modified residues that the two parsers map differently.
MIN_IDENTITY = 0.95


def identity(a: str, b: str) -> float:
    """Fraction of aligned positions that agree, for equal-length sequences."""
    if len(a) != len(b) or not a:
        return 0.0
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def match_chain(
    native_seq: str,
    model,
    structure_params,
    prefer_chain: str | None = None,
    homodimer_policy: str = "raise",
) -> tuple[str, float, str | None]:
    """Find which chain of ``model`` the exported native sequence came from.

    Returns ``(chain_id, identity, note)`` where ``note`` records any tie that
    had to be broken, so an arbitrary choice appears in the output rather than
    only in someone's memory.

    Ties happen with homodimers, where two chains carry the same sequence. The
    choice is then genuinely arbitrary, but arbitrary is not the same as
    unimportant: it must be recorded. ``homodimer_policy`` decides what happens.

    ``raise``
        Stop and make a person choose. Correct when a tie might indicate
        something unexpected about the structure.
    ``first``
        Take the alphabetically first chain and record that it was a tie. Sound
        for a genuine homodimer, where both chains were redesigned by the
        upstream sampling anyway and the interface is symmetric, so either
        choice gives the same interface.
    """
    scores: list[tuple[float, str]] = []
    for chain_id in chains_present(model):
        try:
            extract = extract_chain(model, chain_id, structure_params)
        except StructureError:
            continue
        if len(extract.sequence) != len(native_seq):
            continue
        scores.append((identity(extract.sequence, native_seq), chain_id))

    if not scores:
        # Report how close the nearest chain is, because the distance says what
        # kind of problem this is. A difference of one or two residues is a
        # parser disagreement: ProteinMPNN's parse_PDB requires a complete
        # N/CA/C/O backbone and drops residues missing one, while this project
        # keeps any residue with a recognised name. A large difference means a
        # genuinely different chain or a revised entry.
        available = {}
        for chain_id in chains_present(model):
            try:
                available[chain_id] = len(extract_chain(model, chain_id, structure_params).sequence)
            except StructureError:
                continue
        nearest = (
            min(available.items(), key=lambda kv: abs(kv[1] - len(native_seq)))
            if available
            else None
        )
        detail = ""
        if nearest is not None:
            chain_id, length = nearest
            gap = length - len(native_seq)
            detail = f" Nearest is chain {chain_id} at {length} residues, {gap:+d} from the export."
            if abs(gap) <= 3:
                detail += (
                    " A difference this small is a parser disagreement over residues "
                    "with incomplete backbones, not a different chain. It is left "
                    "unmatched deliberately: reconciling it would require aligning the "
                    "exported sequence onto the structure's residues, and a one-residue "
                    "offset there would silently shift every interface index rather "
                    "than fail visibly."
                )
        raise StructureError(
            f"no chain of length {len(native_seq)} found. Chain lengths present: "
            f"{available or chains_present(model)}.{detail}"
        )

    scores.sort(reverse=True)
    best_score, best_chain = scores[0]

    if best_score < MIN_IDENTITY:
        raise StructureError(
            f"best chain match is only {best_score:.1%} identical (chain {best_chain}). "
            "Refusing to guess: attributing the interface to the wrong chain would "
            "invert the result rather than merely degrade it."
        )

    ties = sorted(c for score, c in scores if score == best_score)
    if len(ties) == 1:
        return best_chain, best_score, None

    if prefer_chain is not None and prefer_chain in ties:
        return (
            prefer_chain,
            best_score,
            f"tie between {ties}, resolved to {prefer_chain} by --prefer-chain",
        )

    if homodimer_policy == "first":
        chosen = ties[0]
        return (
            chosen,
            best_score,
            f"tie between {ties}, took the alphabetically first ({chosen}); the chains "
            "are sequence-identical so the interface is the same either way",
        )

    raise StructureError(
        f"chains {ties} all match at {best_score:.1%}, so the designed chain is "
        "ambiguous. This is expected for a homodimer, where the choice is arbitrary "
        "but must be recorded rather than defaulted. Re-run with --homodimer first "
        "to take the alphabetically first chain and record the tie, or "
        f"--prefer-chain to name one of {ties}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--designs-json", type=Path, required=True, help="Upstream designs_for_folding.json"
    )
    parser.add_argument("--native-dir", type=Path, default=None, help="Default: data/native/")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: data/raw/")
    parser.add_argument(
        "--prefer-chain",
        default=None,
        help="Break ties by preferring this author chain identifier, when it is tied.",
    )
    parser.add_argument(
        "--homodimer",
        choices=("raise", "first"),
        default="raise",
        help=(
            "What to do when two chains match equally well. 'raise' stops and makes "
            "you choose. 'first' takes the alphabetically first and records the tie, "
            "which is sound for a genuine homodimer because the chains are "
            "sequence-identical and the interface is the same either way."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DEFAULT_CONFIG

    if not args.designs_json.is_file():
        fail(f"--designs-json not found: {args.designs_json}")

    native_dir = args.native_dir or NATIVE_DIR
    out_dir = args.out_dir or RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.designs_json.read_text())
    proteins = payload.get("proteins", [])
    if not proteins:
        fail(f"{args.designs_json} contains no 'proteins' entries")

    banner(
        "11_import_upstream_designs",
        args,
        {"proteins": len(proteins), "betas": payload.get("betas"), "native_dir": native_dir},
    )

    design_rows: list[dict] = []
    test_rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    tie_notes: list[dict[str, str]] = []
    partner_notes: list[dict] = []

    with run_manifest(
        script=Path(__file__),
        parameters={
            "min_identity": MIN_IDENTITY,
            "betas": payload.get("betas"),
            "partner_sequence": "native (upstream discarded the designed partner)",
            "partner_selection": "most heavy-atom contacts with the designed chain",
            "prefer_chain": args.prefer_chain,
            "homodimer_policy": args.homodimer,
        },
        seeds={"upstream_seed": "none recorded; upstream sampling was unseeded"},
        inputs=[args.designs_json],
    ) as manifest:
        for order, entry in enumerate(proteins, start=1):
            pdb_id = str(entry["pdb_id"]).upper()
            native_seq = entry["native_seq"]
            native_path = native_dir / f"{pdb_id}.pdb"
            if not native_path.is_file():
                native_path = native_dir / f"{pdb_id}.{config.structure.native_format}"
            if not native_path.is_file():
                skipped.append((pdb_id, f"native structure not found in {native_dir}"))
                continue

            try:
                model = load_model(native_path, config.structure)
                designed_chain, score, tie_note = match_chain(
                    native_seq,
                    model,
                    config.structure,
                    prefer_chain=args.prefer_chain,
                    homodimer_policy=args.homodimer,
                )
                if tie_note:
                    tie_notes.append({"pdb_id": pdb_id, "note": tie_note})

                # Choose the partner by contact rather than by size. A deposited
                # asymmetric unit often holds several copies of an assembly, and
                # the largest other chain is frequently a copy from a
                # neighbouring one that never touches the designed chain. That
                # produces an empty interface, which then propagates as a
                # legitimate-looking zero through every partition and average.
                partner_chain, contacts = choose_contacting_partner(
                    model, designed_chain, config.interface, config.structure
                )
                if partner_chain is None:
                    skipped.append(
                        (
                            pdb_id,
                            f"no chain contacts the designed chain {designed_chain}; this "
                            "entry has no protein-protein interface to analyse",
                        )
                    )
                    continue
                partner_sizes = {
                    partner_chain: len(
                        extract_chain(model, partner_chain, config.structure).sequence
                    )
                }
                if len(contacts) > 1:
                    partner_notes.append(
                        {
                            "pdb_id": pdb_id,
                            "designed_chain": designed_chain,
                            "chosen": partner_chain,
                            "contacts_by_chain": contacts,
                        }
                    )

            except StructureError as exc:
                skipped.append((pdb_id, str(exc)))
                print(f"  [{order}] {pdb_id} SKIPPED: {exc}", file=sys.stderr)
                continue

            test_rows.append(
                {
                    "pdb_id": pdb_id,
                    "chains": f"{designed_chain},{partner_chain}",
                    "mmseqs_cluster_id": f"cluster_{pdb_id}",
                    "split": "test",
                }
            )

            for replicate_index, design in enumerate(entry["designs"]):
                sequence = design["sequence"]
                design_rows.append(
                    {
                        "pdb_id": pdb_id,
                        "designed_chain": designed_chain,
                        "fixed_chain": partner_chain,
                        "beta": float(design["beta"]),
                        "replicate": 0,
                        "sequence": sequence,
                        # Upstream recorded chain_net_charge under the simple
                        # definition. Recomputed here so the reconciliation in
                        # script 02 has something independent to check against.
                        "net_charge_reported": float(
                            design.get("chain_net_charge", net_charge_simple(sequence))
                        ),
                    }
                )
                del replicate_index

            if not args.quiet:
                print(
                    f"  [{order}/{len(proteins)}] {pdb_id} -> chain {designed_chain} "
                    f"({score:.1%} identity, {len(native_seq)} res) vs partner "
                    f"{partner_chain} ({partner_sizes[partner_chain]} res)",
                    file=sys.stderr,
                )

        if not design_rows:
            fail(
                "no designs could be imported. First few skips:\n  "
                + "\n  ".join(f"{p}: {r}" for p, r in skipped[:10])
            )

        designs_path = out_dir / "designs.csv"
        test_path = out_dir / "test_set.csv"
        pd.DataFrame(design_rows).to_csv(designs_path, index=False)
        pd.DataFrame(test_rows).to_csv(test_path, index=False)

        manifest.add_output(designs_path)
        manifest.add_output(test_path)
        manifest.note("n_designs", len(design_rows))
        manifest.note("n_complexes", len(test_rows))
        manifest.note("n_skipped", len(skipped))
        manifest.note("n_ties_resolved", len(tie_notes))
        manifest.note("partner_chosen_by_contact", partner_notes)
        manifest.note("ties_resolved", tie_notes)
        manifest.note(
            "skip_reasons",
            {
                reason: sum(1 for _, r in skipped if r.startswith(reason))
                for reason in sorted({r.split(",")[0].split(".")[0] for _, r in skipped})
            },
        )
        manifest.note("skipped", [{"pdb_id": p, "reason": r[:300]} for p, r in skipped])
        manifest_target = manifest_path_for(designs_path)

    manifest.write(manifest_target)
    print(
        f"\nwrote {designs_path} ({len(design_rows)} designs over {len(test_rows)} complexes)"
        f"\nwrote {test_path}"
        f"\nskipped {len(skipped)}"
        f"\nmanifest hash: {manifest.manifest_hash}"
        "\n\nNOTE: the partner chain is NATIVE in this import, because the upstream"
        "\nexport discarded the designed partner sequence. That is the fixed-target"
        "\nexperiment. Use scripts/10_regenerate_designs.py for the variant where"
        "\nboth chains are redesigned, which is what the upstream sampling did.",
        file=sys.stderr,
    )
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
