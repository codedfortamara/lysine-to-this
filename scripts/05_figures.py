#!/usr/bin/env python
"""Produce the paper figures for the RCSB interface arm.

DELIBERATELY UNFINISHED. This is a documented skeleton, not working code.

It depends on ``results/analysis_summary.csv`` from script 04, which is itself
a skeleton. The figures are specified here so that the analysis is designed
towards a known target rather than the figures being reverse-engineered from
whatever the analysis happened to produce.

Every figure goes through ``interface_charge.plotting.save_figure``, which
stamps the manifest hash into the file metadata and into a footer, so that any
figure can be traced to the run that made it and to the table that shares its
hash.

Planned figures
---------------

**Figure A: where the charge lands.**
Three panels sharing a y axis, one per partition (interface, non-interface
surface, core). x axis: whole-chain charge shift from native, in charge units,
not beta, because beta is an absolute logit shift whose charge effect is size
dependent (the paper's own Limitation (iv)). y axis: that partition's charge
shift. One faint line per complex, plus a heavy median line. A dashed
grey reference line of slope equal to the partition's mean residue-count share,
which is the null model: that is what each partition would absorb if charge were
placed without regard to structural context. The claim is that the interface
line sits below its null and the surface line sits on or above its own.

This is the figure that carries question 1. If the interface line tracks its
null exactly, that is the honest result and the figure still says something
worth saying, namely that the controller is structurally indiscriminate.

**Figure B: does the interface survive?**
Two panels. Left: interface RMSD against charge shift. Right: interface pTM
against charge shift, with the beta = 0 value per complex marked. Points
coloured by whether the complex is in the train or test split. A shaded band
showing the usable charge window derived in script 04.

This is the direct answer to the paper's Limitation (i) and should be
cross-referenced to it in the text.

**Figure C: complementarity as an early-warning signal.**
Electrostatic complementarity sum on the x axis, interface pTM on the y axis,
points coloured by charge shift. If the relationship is monotonic and the
complementarity signal moves at a smaller charge shift than pTM does, the panel
makes the case that a cheap sequence-level statistic anticipates an expensive
structural one.

**Supplementary figure S1: interface definition agreement.**
Jaccard index between the delta-SASA and contact definitions, per complex, as a
strip plot. Establishes that the choice of definition does not drive the
result. If the median Jaccard is high this is a one-line reassurance; if it is
low, the main figures need repeating under the second definition and the paper
has to say which one it used.

**Supplementary figure S2: charge definition reconciliation.**
Computed charge against ``net_charge_reported``, one panel per candidate
definition, from ``results/charge_reconciliation.csv``. Documents which
definition the collaborator's pipeline used, which is otherwise an unresolved
ambiguity in the paper's own methods, since it reports one definition in the
text and targets another in the controller.

Figure conventions
------------------
- Colours from ``plotting.PARTITION_COLOURS``, chosen to survive greyscale
  printing and the common colour vision deficiencies.
- Every charge axis labelled with its definition via
  ``plotting.annotate_definition``. An unlabelled charge axis in a paper about
  charge is the first thing a referee will query.
- PSB single column is 6.5 inches at full bleed. Set the figure size at
  creation rather than scaling afterwards, which changes the effective font
  size.
- Both PNG and PDF are written. PDF for the manuscript, PNG for slides and for
  the metadata that survives a screenshot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interface_charge.cli import add_common_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--analysis-summary", type=Path, default=Path("results/analysis_summary.csv")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Output formats. PDF for the manuscript, PNG for slides.",
    )
    return add_common_arguments(parser)


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise NotImplementedError(
        "05_figures.py is a documented skeleton, not an implementation.\n"
        "\n"
        "It depends on results/analysis_summary.csv from script 04, which is\n"
        "itself a skeleton. The five planned figures are specified in this\n"
        "script's module docstring, including the null-model reference line\n"
        "that Figure A needs in order to mean anything.\n"
        "\n"
        "The plotting house style, the manifest stamping and the partition\n"
        "colours are implemented and tested in\n"
        "src/interface_charge/plotting.py."
    )


if __name__ == "__main__":
    sys.exit(main())
