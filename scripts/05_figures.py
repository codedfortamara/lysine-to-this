#!/usr/bin/env python
"""Produce the paper figures for the RCSB interface arm.

Every figure goes through ``interface_charge.plotting.save_figure``, which
stamps the manifest hash into the file metadata and into a footer, so any figure
can be traced to the run that made it and to the table sharing its hash.

Each figure is built only if its input table exists. A missing input is reported
and skipped rather than faked, so this script is useful before the whole
pipeline has run and never invents a panel.

The figures
-----------

**Figure 1: where the charge lands.** The answer to question 1. Left panel, the
buffering ratio for each partition, one point per complex, against a reference
line at 1.0 which is what a region absorbing exactly its residue share would
score. Right panel, the interface minus surface difference per complex, sorted,
so the reader can count how many complexes buffer rather than trusting a mean.

**Figure 2: buffering is caused by the partner.** The ablation. One line per
complex joining its interface-minus-surface gap with the partner present to the
same gap with the partner removed from ProteinMPNN's input. The reversal is the
result, and a paired plot shows it without averaging it away.

**Figure 3: it is not a burial effect.** Observed buffering ratio against the
rSASA-matched permutation null, one point per chain, with the identity line.
Points below the line are chains where the interface absorbs less than
burial-matched surface predicts.

**Figure 4: the foldability boundary.** The designs on the Uversky
charge-hydropathy plane, coloured by beta, with the published separator drawn.
Reported as a stress indicator, not a foldability classifier: the boundary was
fitted to natural proteins and over-calls disorder here, which the caption must
say.

**Figure 5: electrostatic complementarity.** Complementarity against beta, one
line per complex plus a heavy median, showing the monotonic and asymmetric
response.

**Supplementary S1: interface definition agreement.** Jaccard index between the
delta-SASA and contact definitions per complex. Establishes that the choice of
definition does not drive the result.

Figure conventions
------------------
- Colours from ``plotting.PARTITION_COLOURS``, chosen to survive greyscale
  printing and the common colour vision deficiencies.
- PSB single column is 6.5 inches at full bleed. Figure size is set at creation
  rather than scaled afterwards, which would change the effective font size.
- Both PNG and PDF are written. PDF for the manuscript, PNG for slides and for
  the metadata that survives a screenshot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from interface_charge.cli import add_common_arguments, banner, fail, resolve_config
from interface_charge.config import FIGURES_DIR, RESULTS_DIR
from interface_charge.plotting import (
    PARTITION_COLOURS,
    new_figure,
    partition_colour,
    save_figure,
)
from interface_charge.provenance import manifest_path_for, run_manifest

#: Partitions in the order they are argued about, not alphabetical.
PARTITION_ORDER = ["interface", "surface", "core"]

#: Uversky, Gillespie and Fink (2000) separator, repeated here for the figure
#: only. The authoritative copy is in ``interface_charge.uversky``.
BOUNDARY_SLOPE, BOUNDARY_INTERCEPT = 2.785, -1.151


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--figures-dir", type=Path, default=None, help="Default: figures/")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Output formats. PDF for the manuscript, PNG for slides.",
    )
    return add_common_arguments(parser)


def read_if_present(path: Path, missing: list[str]) -> pd.DataFrame | None:
    """Load a results table, recording rather than raising when it is absent."""
    if not path.is_file():
        missing.append(str(path))
        return None
    return pd.read_csv(path)


def figure_where_charge_lands(table: pd.DataFrame, definition: str = "simple"):
    """Buffering ratio by partition, and the paired interface-surface gap."""
    data = table[table["charge_definition"] == definition]
    fig, (left, right) = new_figure(1, 2, width_in=6.5, height_in=3.0)

    rng = np.random.default_rng(0)
    for index, partition in enumerate(PARTITION_ORDER):
        values = data[data["partition"] == partition]["buffering_ratio"].dropna()
        # Jitter is cosmetic only, so it is seeded and never touches a statistic.
        jitter = rng.uniform(-0.16, 0.16, len(values))
        left.scatter(
            index + jitter, values, s=9, alpha=0.55, color=partition_colour(partition), zorder=3
        )
        left.hlines(values.median(), index - 0.3, index + 0.3, color="#111111", lw=1.8, zorder=4)

    left.axhline(1.0, color="#111111", ls="--", lw=1.0, zorder=2)
    left.text(
        2.42, 1.0, "fair share", fontsize=7, va="bottom", ha="right", color="#111111", alpha=0.8
    )
    left.set_xticks(range(len(PARTITION_ORDER)))
    left.set_xticklabels(PARTITION_ORDER)
    left.set_ylabel("buffering ratio\n(slope / residue share)")
    left.set_title("a. charge absorbed, by region", loc="left")

    wide = data.pivot_table(index="pdb_id", columns="partition", values="buffering_ratio")
    gap = (wide["interface"] - wide["surface"]).dropna().sort_values()
    colours = [
        PARTITION_COLOURS["interface"] if v < 0 else PARTITION_COLOURS["surface"] for v in gap
    ]
    right.bar(range(len(gap)), gap.to_numpy(), color=colours, width=1.0)
    right.axhline(0.0, color="#111111", lw=1.0)
    right.set_xlabel(f"complex, sorted ({int((gap < 0).sum())} of {len(gap)} buffer)")
    right.set_ylabel("interface minus surface")
    right.set_title("b. per complex", loc="left")
    fig.tight_layout()
    return fig


def figure_partner_ablation(table: pd.DataFrame):
    """Paired gap with the partner present and with it removed."""
    fig, ax = new_figure(width_in=4.2, height_in=3.4)
    for row in table.itertuples():
        weakened = row.gap_isolated > row.gap_paired
        ax.plot(
            [0, 1],
            [row.gap_paired, row.gap_isolated],
            color=PARTITION_COLOURS["interface"] if weakened else PARTITION_COLOURS["core"],
            alpha=0.35,
            lw=0.8,
        )
    for index, column in enumerate(("gap_paired", "gap_isolated")):
        ax.scatter([index] * len(table), table[column], s=12, color="#111111", zorder=3)
        ax.hlines(table[column].median(), index - 0.12, index + 0.12, color="#111111", lw=2.5)

    ax.axhline(0.0, color="#111111", ls="--", lw=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["partner\npresent", "partner\nremoved"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("interface minus surface buffering ratio")
    ax.set_title("buffering requires partner context", loc="left")
    fig.tight_layout()
    return fig


def figure_burial_control(table: pd.DataFrame):
    """Observed against the burial-matched null, one point per chain."""
    fig, ax = new_figure(width_in=4.2, height_in=3.6)
    null = table["observed_ratio"] - table["observed_minus_null"]
    significant = table["p_permutation"] < 0.05
    ax.scatter(
        null[~significant],
        table["observed_ratio"][~significant],
        s=14,
        color=PARTITION_COLOURS["core"],
        label="not significant",
    )
    ax.scatter(
        null[significant],
        table["observed_ratio"][significant],
        s=14,
        color=PARTITION_COLOURS["interface"],
        label="p < 0.05",
    )
    limits = [
        min(null.min(), table["observed_ratio"].min()) - 0.05,
        max(null.max(), table["observed_ratio"].max()) + 0.05,
    ]
    ax.plot(limits, limits, color="#111111", ls="--", lw=1.0)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("burial-matched null buffering ratio")
    ax.set_ylabel("observed interface buffering ratio")
    ax.set_title("below the line: not explained by burial", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def figure_uversky(table: pd.DataFrame):
    """The designs on the charge-hydropathy plane, with the published boundary."""
    fig, ax = new_figure(width_in=4.6, height_in=3.6)
    betas = sorted(table["beta"].unique())
    palette = ["#08306b", "#4292c6", "#6e6e6e", "#f16913", "#7f2704"]
    for beta, colour in zip(betas, palette[: len(betas)], strict=False):
        subset = table[table["beta"] == beta]
        ax.scatter(
            subset["mean_scaled_hydropathy"],
            subset["mean_net_charge"],
            s=12,
            alpha=0.75,
            color=colour,
            label=f"beta = {beta:+.1f}",
        )
    span = np.linspace(
        table["mean_scaled_hydropathy"].min() - 0.02,
        table["mean_scaled_hydropathy"].max() + 0.02,
        50,
    )
    ax.plot(span, BOUNDARY_SLOPE * span + BOUNDARY_INTERCEPT, color="#111111", lw=1.2)
    ax.set_xlabel("mean scaled hydropathy")
    ax.set_ylabel("mean absolute net charge")
    ax.set_title("distance from natural folded space", loc="left")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def figure_complementarity(table: pd.DataFrame, column: str):
    """Complementarity against beta, per complex, with a median line."""
    fig, ax = new_figure(width_in=4.2, height_in=3.2)
    for _, group in table.groupby("pdb_id"):
        ordered = group.sort_values("beta")
        ax.plot(
            ordered["beta"], ordered[column], color=PARTITION_COLOURS["core"], alpha=0.2, lw=0.7
        )
    median = table.groupby("beta")[column].median()
    ax.plot(
        median.index, median.to_numpy(), color=PARTITION_COLOURS["interface"], lw=2.2, marker="o"
    )
    ax.axhline(0.0, color="#111111", ls="--", lw=1.0)
    ax.set_xlabel("beta")
    ax.set_ylabel("electrostatic complementarity")
    ax.set_title("complementarity tracks the dial", loc="left")
    fig.tight_layout()
    return fig


def figure_definition_agreement(table: pd.DataFrame):
    """Jaccard between the two interface definitions, per complex."""
    fig, ax = new_figure(width_in=4.0, height_in=2.6)
    values = table["interface_definition_jaccard"].dropna()
    ax.hist(values, bins=20, color=PARTITION_COLOURS["interface"], alpha=0.85)
    ax.axvline(values.median(), color="#111111", lw=1.6)
    ax.set_xlabel("Jaccard, delta-SASA against contact definition")
    ax.set_ylabel("chains")
    ax.set_title(f"definitions agree, median {values.median():.3f}", loc="left")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_config(args)
    results_dir = args.results_dir or RESULTS_DIR
    figures_dir = args.figures_dir or FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    per_complex = read_if_present(results_dir / "analysis_per_complex.csv", missing)
    ablation = read_if_present(results_dir / "partner_ablation.csv", missing)
    burial = read_if_present(results_dir / "burial_control.csv", missing)
    uversky = read_if_present(results_dir / "uversky.csv", missing)
    complementarity = read_if_present(results_dir / "complementarity.csv", missing)
    interfaces = read_if_present(results_dir / "interfaces.csv", missing)

    inputs = [
        results_dir / name
        for name in (
            "analysis_per_complex.csv",
            "partner_ablation.csv",
            "burial_control.csv",
            "uversky.csv",
            "complementarity.csv",
            "interfaces.csv",
        )
        if (results_dir / name).is_file()
    ]
    if not inputs:
        fail(
            f"no results tables found under {results_dir}. Run scripts 01 to 04 first; "
            "this script draws what the analysis produced and invents nothing."
        )

    banner("05_figures", args, {"tables": len(inputs), "figures_dir": figures_dir})

    written: list[Path] = []
    with run_manifest(
        script=Path(__file__),
        parameters={"formats": list(args.formats), "charge_definition": "simple"},
        seeds={"jitter_seed": 0},
        inputs=inputs,
    ) as manifest:
        plan = [
            ("figure_1_where_charge_lands", per_complex, figure_where_charge_lands, {}),
            ("figure_2_partner_ablation", ablation, figure_partner_ablation, {}),
            ("figure_3_burial_control", burial, figure_burial_control, {}),
            ("figure_4_uversky", uversky, figure_uversky, {}),
            ("supp_s1_definition_agreement", interfaces, figure_definition_agreement, {}),
        ]
        for name, table, builder, kwargs in plan:
            if table is None or table.empty:
                print(f"  skipped {name}: input table absent or empty", file=sys.stderr)
                continue
            figure = builder(table, **kwargs)
            paths = save_figure(figure, figures_dir / name, manifest, formats=tuple(args.formats))
            written.extend(paths)
            print(f"  wrote {name} ({len(paths)} format(s))", file=sys.stderr)

        if complementarity is not None and not complementarity.empty:
            column = next(
                (c for c in complementarity.columns if c.startswith("ec_sum_product")), None
            )
            if column is None:
                print("  skipped figure_5: no ec_sum_product column", file=sys.stderr)
            else:
                figure = figure_complementarity(complementarity, column)
                paths = save_figure(
                    figure,
                    figures_dir / "figure_5_complementarity",
                    manifest,
                    formats=tuple(args.formats),
                )
                written.extend(paths)
                print(f"  wrote figure_5_complementarity ({len(paths)} format(s))", file=sys.stderr)

        for path in written:
            manifest.add_output(path)
        manifest.note("n_figures", len(written))
        manifest.note("skipped_inputs", missing)
        manifest_target = manifest_path_for(figures_dir / "figures")

    manifest.write(manifest_target)

    if missing:
        print(
            f"\n{len(missing)} input table(s) absent, their figures were skipped:", file=sys.stderr
        )
        for path in missing:
            print(f"  {path}", file=sys.stderr)
    print(f"\nwrote {len(written)} file(s) to {figures_dir}", file=sys.stderr)
    print(f"manifest hash: {manifest.manifest_hash}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
