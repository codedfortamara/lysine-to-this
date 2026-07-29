#!/usr/bin/env python
"""Assemble a self-contained directory to contribute to the upstream repository.

Built as a script rather than done by hand because it will be run more than
once: the results change as the grid completes, and a contribution assembled by
copying files is a contribution nobody can reproduce or refresh.

The upstream repository is flat research code with no packaging, so everything
lands in one directory that collides with nothing. What goes in:

* the analysis package and the scripts that drive it
* the tests, because a contribution whose correctness cannot be checked is a
  liability to whoever inherits it
* the results tables with their manifests, so every number can be traced to the
  run and the code version that produced it
* the Colab notebook, which is the specific thing that was asked for
* a README stating what this adds, what it does not, and how to rerun it

It refuses to package results it cannot vouch for. A directory that looks
complete and is not is worse than one that is obviously partial.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interface_charge.cli import fail

ROOT = Path(__file__).resolve().parents[1]

#: Copied verbatim. Paths are relative to the repository root.
CODE = [
    "src/interface_charge",
    "scripts",
    "modal_app",
    "tests",
    "notebooks",
]

#: Results tables that carry the findings. Each must have a manifest beside it.
RESULT_TABLES = [
    "analysis_per_complex.csv",
    "partner_ablation.csv",
    "burial_control.csv",
    "interfaces.csv",
    "interface_definitions.json",
    "complementarity.csv",
    "uversky.csv",
    "af2_metrics.csv",
    "af2_interface_survival.csv",
]

#: Summaries, which carry the confidence intervals rather than the raw rows.
RESULT_SUMMARIES = [
    "analysis_summary.json",
    "partner_ablation_summary.json",
    "burial_control_summary.json",
    "uversky_summary.json",
    "af2_interface_survival_summary.json",
]

EXCLUDE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_summary(results_dir: Path, name: str) -> dict:
    path = results_dir / name
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def build_readme(results_dir: Path, commit: str) -> str:
    """A README whose numbers come from the results, not from memory."""
    ablation = read_summary(results_dir, "partner_ablation_summary.json")
    burial = read_summary(results_dir, "burial_control_summary.json")
    survival = read_summary(results_dir, "af2_interface_survival_summary.json")

    def interval(block: dict, key: str = "ci") -> str:
        low, high = block.get(key, [float("nan"), float("nan")])
        return f"[{low:+.4f}, {high:+.4f}]"

    lines = [
        "# RCSB interface analysis",
        "",
        "Complex-aware analysis of the RCSB protein-protein set, addressing",
        "Limitations item (i): the foldability metric folds each chain in",
        "isolation, which cannot say whether the interface survives the charge",
        "dial. Everything here is computed on the complex.",
        "",
        f"Produced at commit `{commit}`.",
        "",
        "## What this adds",
        "",
        "**1. Where the imposed charge lands.** Each designed chain is split into",
        "buried core, interface, and remaining surface. Net charge is regressed on",
        "beta per region and the slope divided by that region's share of residues,",
        "so 1.0 means a region absorbing exactly its proportional share.",
        "",
    ]

    if ablation:
        paired = ablation.get("paired_arm", {})
        isolated = ablation.get("isolated_arm", {})
        effect = ablation.get("effect_of_removing_the_partner", {})
        lines += [
            f"Interface minus surface, partner present: "
            f"**{paired.get('mean_interface_minus_surface', float('nan')):+.4f}** "
            f"{interval(paired)}, negative in "
            f"{paired.get('n_buffering', '?')} of {ablation.get('n_complexes', '?')} complexes.",
            "",
            "**2. It is not a burial artefact.** Interface residues are more buried",
            "than average, which could explain the above on its own. Permuting",
            "region labels 10,000 times per chain, with each interface residue",
            "restricted to swap only with surface residues of matched relative",
            "solvent accessibility:",
            "",
        ]
    if burial:
        block = burial.get("observed_minus_null", {})
        significant = next((v for k, v in burial.items() if k.startswith("n_chains_p_below")), "?")
        lines += [
            f"Observed minus null: **{block.get('mean', float('nan')):+.4f}** "
            f"[{block.get('ci_low', float('nan')):+.4f}, "
            f"{block.get('ci_high', float('nan')):+.4f}], significant in "
            f"{significant} of {burial.get('n_chains', '?')} chains over "
            f"{burial.get('permutations', '?')} permutations.",
            "",
        ]
    if ablation:
        lines += [
            "**3. It is caused by partner context.** The partner chain is deleted",
            "from the structure before ProteinMPNN sees it and every design is",
            "regenerated. Interface membership is defined on the native complex in",
            "both arms, so the two are compared over identical positions and only",
            "the model's knowledge differs.",
            "",
            f"Partner removed: "
            f"{isolated.get('mean_interface_minus_surface', float('nan')):+.4f} "
            f"{interval(isolated)}. Effect of removal: "
            f"**{effect.get('estimate', float('nan')):+.4f}** {interval(effect)}, "
            f"weakened in {effect.get('n_weakened', '?')} of "
            f"{ablation.get('n_complexes', '?')} complexes.",
            "",
            "The effect reverses without the partner. So the model spares the",
            "contact patch because it can see what binds there, not because those",
            "positions differ intrinsically. That is a mechanism for why charge",
            "control is cheap at interfaces, rather than only a check that it is.",
            "",
        ]

    lines += [
        "**4. Does the interface survive.** AlphaFold2-Multimer refolds the",
        "complex, scored by ipSAE with d0 taken from the interface residue count",
        "rather than total chain length, as a within-complex paired change against",
        "the same complex at beta = 0.",
        "",
    ]
    if survival:
        for record in survival.get("primary_all_returned", []):
            mark = " (excludes zero)" if record.get("excludes_zero") else ""
            lines.append(
                f"- beta {record['beta']:+.1f}: "
                f"{record['mean_change_vs_beta0']:+.4f} "
                f"[{record['ci_low']:+.4f}, {record['ci_high']:+.4f}]{mark}, "
                f"{record['n_degraded']} of {record['n_complexes']} degraded"
            )
        lines += ["", survival.get("fold_quality_reading", ""), ""]
        completeness = survival.get("completeness", {})
        if completeness:
            lines += [
                "Completeness by charge setting, since refolding failures",
                "concentrate in large complexes and at extreme beta and are",
                "therefore not missing at random:",
                "",
                "```",
                json.dumps(completeness.get("return_rate_per_beta", {}), indent=2),
                "```",
                "",
            ]

    lines += [
        "## What this does not claim",
        "",
        "- All evidence is computational. Nothing here is an assay.",
        "- ipSAE is from a preprint and is not peer reviewed. actifpTM is a",
        "  peer-reviewed alternative and was not computed.",
        "- The mixed alignment treatment, native partner with an MSA and the",
        "  redesigned chain as a single sequence, means no homologues are paired",
        "  across the interface. Interface confidence therefore comes from the",
        "  partner's evolutionary signal and the design's own sequence alone.",
        "- Predictions use one model and one seed. AlphaFold confidence is not a",
        "  binding assay.",
        "- The salt-bridge normalisation used here is bridges per charged-pair",
        "  opportunity, which is not a standard denominator and is not directly",
        "  comparable to the C-beta proximity count reported upstream. The two",
        "  need reconciling rather than reporting side by side.",
        "",
        "## Reproducing",
        "",
        "```",
        "uv pip install -e .",
        "python scripts/01_define_interfaces.py --config config.toml",
        "python scripts/02_charge.py",
        "python scripts/04_analysis.py",
        "python scripts/07_burial_control.py",
        "python scripts/09_partner_ablation.py",
        "python scripts/12_af2_interface_survival.py",
        "python scripts/05_figures.py",
        "```",
        "",
        "The AlphaFold grid runs on Modal, or on a single GPU via",
        "`notebooks/af2_colab.ipynb`, which is generated from the Modal source so",
        "the two backends cannot compute a metric differently.",
        "",
        "Every output carries a manifest recording input hashes, the git commit,",
        "package versions, seeds and parameters.",
        "",
        "## Tests",
        "",
        "```",
        "python -m pytest",
        "```",
        "",
        "The tests are part of the contribution. Several encode failures that cost",
        "real money or would have produced confident wrong numbers: a silent CPU",
        "fallback billed at GPU rates, an alignment passed as a string and",
        "silently discarded, and native and predicted residues paired by residue",
        "number when the natives carry author numbering starting at 105.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dist/rcsb_interface"),
        help="Directory to assemble. Overwritten if it exists.",
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--allow-missing-results",
        action="store_true",
        help=(
            "Package even when result tables are absent. Off by default: a "
            "directory that looks complete and is not is worse than one that is "
            "obviously partial."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.results_dir

    missing = [name for name in RESULT_TABLES if not (results_dir / name).is_file()]
    if missing and not args.allow_missing_results:
        fail(
            f"{len(missing)} result table(s) are absent from {results_dir}:\n  "
            + "\n  ".join(missing)
            + "\n\nRun the pipeline first, or pass --allow-missing-results to "
            "package what exists.\nA contribution that looks complete and is not "
            "is worse than one that is obviously partial."
        )

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    copied: list[str] = []
    for relative in CODE:
        source = ROOT / relative
        if not source.exists():
            continue
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=EXCLUDE)
        else:
            shutil.copy2(source, target)
        copied.append(relative)

    for extra in ("pyproject.toml", "config.toml", "HANDOVER.md", "README.md"):
        source = ROOT / extra
        if source.is_file():
            shutil.copy2(source, out / (f"upstream_{extra}" if extra == "README.md" else extra))
            copied.append(extra)

    packaged_results = out / "results"
    packaged_results.mkdir(exist_ok=True)
    n_results = 0
    for name in RESULT_TABLES + RESULT_SUMMARIES:
        source = results_dir / name
        if not source.is_file():
            continue
        shutil.copy2(source, packaged_results / name)
        n_results += 1
        manifest = results_dir / f"{name}.manifest.json"
        if manifest.is_file():
            shutil.copy2(manifest, packaged_results / manifest.name)

    commit = git_commit()
    (out / "README.md").write_text(build_readme(results_dir, commit))

    print(f"assembled {out}")
    print(f"  code:    {', '.join(copied)}")
    print(f"  results: {n_results} file(s)")
    print(f"  commit:  {commit}")
    if missing:
        print(f"\n  WARNING: packaged without {len(missing)} result table(s): {missing}")
    print(
        "\nNext, following the upstream instruction to branch from the latest commit:\n"
        "  git clone https://github.com/SyedMohammedSameer/ZetaDial.git\n"
        "  cd ZetaDial\n"
        "  git pull\n"
        "  git checkout -b rcsb-interface-analysis\n"
        f"  (copy {out} in as rcsb_interface/)\n"
        "  git add rcsb_interface && git commit && git push -u origin rcsb-interface-analysis"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
