#!/usr/bin/env python
"""Regenerate charge-controlled ProteinMPNN designs, seeded and keeping every chain.

This reproduces the collaborator's sampling primitive exactly, with three
changes that the complex-aware analysis requires.

**Every chain is kept.** The upstream export samples the whole complex and then
extracts only the largest chain's sequence, discarding the rest. That is
sufficient for folding one chain in isolation, and insufficient for anything
about the interface: the partner across that interface was also redesigned, and
also had its charge shifted, so refolding the complex needs its sequence too.
This script emits one row per chain per design.

**Sampling is seeded.** The upstream script calls ``torch.randn`` with no seed
set anywhere, so its designs cannot be regenerated. Here every design draws from
a seed derived deterministically from ``(pdb_id, chain, beta, replicate,
global seed)``, so any single design can be reproduced in isolation without
replaying the whole run, and the whole run is reproducible from the manifest.

**Nothing is silently coerced.** The upstream ``seq_str`` maps an unknown
residue code onto alanine. ``omit_AAs`` almost certainly prevents that ever
firing, but a latent silent substitution in a charge calculation is exactly the
failure this project exists to exclude, so here it raises instead.

The steering primitive itself is unchanged and must stay that way: the bias is
added to the K and R logits and subtracted from the D and E logits, indexed
against ProteinMPNN's own alphabet ``ACDEFGHIKLMNPQRSTVWYX``, at temperature
0.1. Indexing against a different alphabet ordering, AlphaFold's for instance,
biases the wrong residues and moves no charge, which is a silent failure the
source paper explicitly warns about.

Output is the project data contract: ``designs.csv`` and ``test_set.csv``, so
scripts 00 to 03 consume it directly.

A note on ``fixed_chain``
------------------------
The contract's ``fixed_chain`` column names the partner chain, but in this
pipeline the partner is **not** held native: ProteinMPNN is called with no
chain mask, so every chain is redesigned. Each chain therefore appears as
``designed_chain`` in its own row, and as ``fixed_chain`` in its partner's row.
Join on ``(pdb_id, beta, replicate)`` to recover both designed sequences of a
complex. The ``--native-partner`` flag emits the alternative experiment, in
which the partner keeps its native sequence, which is the setting that matches
binder design against a fixed target.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from interface_charge.charge import net_charge_simple
from interface_charge.cli import banner, fail
from interface_charge.config import RAW_DIR, STANDARD_AA
from interface_charge.provenance import manifest_path_for, run_manifest

#: ProteinMPNN's own alphabet. Index positions matter: biasing against a
#: different ordering silently shifts the wrong residues.
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

#: Sampling temperature. Softmax divides logits by this, so the effective
#: strength of a given beta scales inversely with it. Changing this without
#: changing beta changes the charge shift.
DEFAULT_TEMPERATURE = 0.1

#: The beta grid used upstream for the RCSB arm.
DEFAULT_BETAS = (-3.0, -1.5, 0.0, 1.5, 3.0)

#: Chain length window for the PRIMARY chain. This is an ESMFold constraint
#: inherited from upstream, not a property of the biology: the upstream export
#: folded one chain in isolation and needed it to fit. It is kept so that the
#: primary chain selected here matches the one upstream selected, which is what
#: makes the two sets of designs comparable.
CHAIN_MIN, CHAIN_MAX = 30, 500

#: The partner chain is deliberately NOT held to that window. Complex-aware
#: refolding has no per-chain limit, only a total-length one, and excluding
#: complexes because their partner is large would bias the set towards small
#: interfaces, which is precisely the population this study is about.
PARTNER_MIN = 30

#: Total residues across both chains. AlphaFold2-Multimer memory grows roughly
#: with the square of this, so it is the real constraint on what can be
#: refolded. Complexes above it are recorded as skipped rather than silently
#: dropped.
MAX_TOTAL_RESIDUES = 1400


def derive_seed(global_seed: int, pdb_id: str, beta: float, replicate: int) -> int:
    """A stable per-design seed.

    Derived by hashing rather than incrementing a counter, so that adding a beta
    value or a replicate to the grid does not shift the seed of every design
    that came after it. Any single design can then be regenerated on its own.
    """
    key = f"{global_seed}|{pdb_id}|{beta:.6f}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def sequence_from_indices(indices, positions: list[int]) -> str:
    """Decode sampled indices to a one-letter sequence, refusing unknown codes."""
    letters = []
    for position in positions:
        code = MPNN_ALPHABET[int(indices[position])]
        if code not in STANDARD_AA:
            raise ValueError(
                f"ProteinMPNN emitted residue code {code!r} at position {position}. "
                "The upstream export silently rewrote this to alanine, which would "
                "change the net charge with nothing to show for it. Investigate "
                "the omit_AAs setting rather than substituting."
            )
        letters.append(code)
    return "".join(letters)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pdb-dir", type=Path, required=True, help="Directory of input .pdb structures"
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Newline-separated PDB IDs. Defaults to every .pdb in --pdb-dir, sorted.",
    )
    parser.add_argument(
        "--mpnn-dir", type=Path, required=True, help="Path to a ProteinMPNN checkout"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Model checkpoint. Defaults to vanilla_model_weights/v_48_020.pt inside --mpnn-dir.",
    )
    parser.add_argument("--limit", type=int, default=60, help="How many structures to process")
    parser.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Independent samples per (structure, beta). Upstream used 1.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--chain-min", type=int, default=CHAIN_MIN)
    parser.add_argument("--chain-max", type=int, default=CHAIN_MAX)
    parser.add_argument(
        "--max-total-residues",
        type=int,
        default=MAX_TOTAL_RESIDUES,
        help="Skip complexes whose two chains together exceed this, for AF2 memory.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: data/raw/")
    parser.add_argument(
        "--native-partner",
        action="store_true",
        help=(
            "Emit the partner chain at its NATIVE sequence instead of its designed "
            "one. This is the fixed-target experiment, not what upstream ran."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.pdb_dir.is_dir():
        fail(f"--pdb-dir not found: {args.pdb_dir}")
    if not args.mpnn_dir.is_dir():
        fail(
            f"--mpnn-dir not found: {args.mpnn_dir}. "
            "Clone it with: git clone https://github.com/dauparas/ProteinMPNN"
        )

    weights = args.weights or (args.mpnn_dir / "vanilla_model_weights" / "v_48_020.pt")
    if not weights.is_file():
        fail(f"model checkpoint not found: {weights}")

    try:
        import torch
    except ImportError:
        fail("torch is required. Install it with: uv pip install torch")
        return 2

    sys.path.insert(0, str(args.mpnn_dir))
    from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

    out_dir = args.out_dir or RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split_file:
        ids = [line.strip() for line in args.split_file.read_text().split() if line.strip()]
    else:
        ids = sorted(p.stem for p in args.pdb_dir.glob("*.pdb"))
    ids = ids[: args.limit]

    banner(
        "10_regenerate_designs",
        args,
        {
            "structures": len(ids),
            "betas": args.betas,
            "replicates": args.replicates,
            "seed": args.seed,
            "partner": "native" if args.native_partner else "designed",
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    index_of = {aa: MPNN_ALPHABET.index(aa) for aa in "DEKR"}
    omit = np.array([aa in "X" for aa in MPNN_ALPHABET], dtype=np.float32)

    design_rows: list[dict] = []
    test_rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    with run_manifest(
        script=Path(__file__),
        parameters={
            "betas": args.betas,
            "replicates": args.replicates,
            "temperature": args.temperature,
            "limit": args.limit,
            "primary_chain_window": [args.chain_min, args.chain_max],
            "partner_min_residues": PARTNER_MIN,
            "max_total_residues": args.max_total_residues,
            "alphabet": MPNN_ALPHABET,
            "weights": str(weights),
            "native_partner": args.native_partner,
            "device": str(device),
        },
        seeds={"global_seed": args.seed, "derivation": "sha256(seed|pdb|beta|replicate)"},
        inputs=[weights] + ([args.split_file] if args.split_file else []),
    ) as manifest:
        for order, pdb_id in enumerate(ids, start=1):
            path = args.pdb_dir / f"{pdb_id}.pdb"
            if not path.is_file():
                skipped.append((pdb_id, "structure file missing"))
                continue

            try:
                batch = parse_PDB(str(path))
                entry = batch[0]

                # With chain_dict=None, ProteinMPNN sorts the author chain
                # letters alphabetically and assigns encoding 1..n in that
                # order, so encoding c corresponds to letters[c - 1]. Every
                # chain is masked, meaning every chain is redesigned.
                letters = sorted(k[-1:] for k in entry if k.startswith("seq_chain_"))

                featurised = tied_featurize(batch, device, None, None, None, None, None, None)
                X, S, mask, _lengths, chain_M, chain_encoding_all = featurised[:6]
                residue_idx, omit_AA_mask = featurised[12], featurised[11]
                pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all = featurised[15:19]

                valid = mask[0] > 0
                encoding = chain_encoding_all[0]
                present = [int(c) for c in encoding[valid].unique().tolist()]
                sizes = {c: int(((encoding == c) & valid).sum().item()) for c in present}

                # Primary chain: largest one inside the upstream window, which is
                # exactly the chain the upstream export selected and folded.
                candidates = [
                    c for c, size in sizes.items() if args.chain_min <= size <= args.chain_max
                ]
                if not candidates:
                    skipped.append(
                        (pdb_id, f"no chain in [{args.chain_min}, {args.chain_max}], sizes {sizes}")
                    )
                    continue
                primary = max(candidates, key=lambda c: sizes[c])

                # Partner: the largest other chain. Not held to the primary's
                # window, since that window exists only because ESMFold folded
                # one chain alone.
                others = [c for c in present if c != primary and sizes[c] >= PARTNER_MIN]
                if not others:
                    skipped.append(
                        (
                            pdb_id,
                            f"no partner chain of at least {PARTNER_MIN} residues, sizes {sizes}",
                        )
                    )
                    continue
                partner = max(others, key=lambda c: sizes[c])

                total = sizes[primary] + sizes[partner]
                if total > args.max_total_residues:
                    skipped.append(
                        (
                            pdb_id,
                            f"{total} residues across the pair exceeds the "
                            f"{args.max_total_residues} cap for complex-aware refolding",
                        )
                    )
                    continue
                chain_of = {
                    primary: letters[primary - 1],
                    partner: letters[partner - 1],
                }
                positions = {
                    c: ((encoding == c) & valid).nonzero(as_tuple=True)[0].tolist()
                    for c in (primary, partner)
                }
                native_seq = {
                    c: sequence_from_indices(S[0], positions[c]) for c in (primary, partner)
                }

                test_rows.append(
                    {
                        "pdb_id": pdb_id.upper(),
                        "chains": f"{chain_of[primary]},{chain_of[partner]}",
                        "mmseqs_cluster_id": f"cluster_{pdb_id.upper()}",
                        "split": "test",
                    }
                )

                for beta in args.betas:
                    bias = np.zeros(len(MPNN_ALPHABET), dtype=np.float32)
                    bias[index_of["K"]] = beta
                    bias[index_of["R"]] = beta
                    bias[index_of["D"]] = -beta
                    bias[index_of["E"]] = -beta

                    for replicate in range(args.replicates):
                        seed = derive_seed(args.seed, pdb_id, beta, replicate)
                        torch.manual_seed(seed)
                        np.random.seed(seed % (2**32))

                        randn = torch.randn(chain_M.shape, device=device)
                        with torch.no_grad():
                            sampled = model.sample(
                                X,
                                randn,
                                S,
                                chain_M,
                                chain_encoding_all,
                                residue_idx,
                                mask=mask,
                                temperature=args.temperature,
                                omit_AAs_np=omit,
                                bias_AAs_np=bias,
                                chain_M_pos=featurised[10],
                                omit_AA_mask=omit_AA_mask,
                                pssm_coef=pssm_coef,
                                pssm_bias=pssm_bias,
                                pssm_multi=0.0,
                                pssm_log_odds_flag=0,
                                pssm_log_odds_mask=(pssm_log_odds_all > 0.0).float(),
                                pssm_bias_flag=0,
                                bias_by_res=bias_by_res_all,
                            )["S"][0]

                        designed = {
                            c: sequence_from_indices(sampled, positions[c])
                            for c in (primary, partner)
                        }
                        if args.native_partner:
                            designed[partner] = native_seq[partner]

                        for this, other in ((primary, partner), (partner, primary)):
                            sequence = designed[this]
                            design_rows.append(
                                {
                                    "pdb_id": pdb_id.upper(),
                                    "designed_chain": chain_of[this],
                                    "fixed_chain": chain_of[other],
                                    "beta": beta,
                                    "replicate": replicate,
                                    "sequence": sequence,
                                    "net_charge_reported": net_charge_simple(sequence),
                                }
                            )

                if not args.quiet:
                    print(
                        f"  [{order}/{len(ids)}] {pdb_id} "
                        f"{chain_of[primary]}({sizes[primary]})/"
                        f"{chain_of[partner]}({sizes[partner]})",
                        file=sys.stderr,
                    )

            except Exception as exc:
                skipped.append((pdb_id, f"{type(exc).__name__}: {exc}"))
                print(f"  [{order}/{len(ids)}] {pdb_id} SKIPPED: {exc}", file=sys.stderr)

        if not design_rows:
            fail(
                "no designs were produced. First few skips:\n  "
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
        manifest.note("skipped", [{"pdb_id": p, "reason": r[:300]} for p, r in skipped])
        manifest_target = manifest_path_for(designs_path)

    manifest.write(manifest_target)
    print(
        f"\nwrote {designs_path} ({len(design_rows)} rows over {len(test_rows)} complexes)"
        f"\nwrote {test_path}"
        f"\nskipped {len(skipped)} structure(s)"
        f"\nmanifest hash: {manifest.manifest_hash}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
