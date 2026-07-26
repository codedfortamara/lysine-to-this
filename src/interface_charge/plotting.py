"""Figure house style, and the rule that every figure carries its provenance.

:func:`save_figure` is the only sanctioned way to write a figure in this
project. It stamps the manifest hash into the file metadata and, by default,
into a small footer on the figure itself. A figure that has been pasted into a
slide deck and emailed twice can then still be traced back to the run that
produced it, which is the point.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .provenance import Manifest, figure_metadata

__all__ = [
    "PARTITION_COLOURS",
    "apply_house_style",
    "new_figure",
    "partition_colour",
    "save_figure",
]

#: Colours for the three partitions, chosen to stay distinguishable in
#: greyscale and under the common forms of colour vision deficiency. Interface
#: is the emphasis colour because it carries the argument.
PARTITION_COLOURS: Final[dict[str, str]] = {
    "interface": "#1b6ca8",
    "surface": "#e08214",
    "core": "#6e6e6e",
}

#: Colour for the native reference in any before-and-after comparison.
NATIVE_COLOUR: Final[str] = "#111111"

_RC: Final[dict[str, Any]] = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
}


def apply_house_style() -> None:
    """Apply the project's matplotlib defaults."""
    plt.rcParams.update(_RC)


def partition_colour(partition: str) -> str:
    """Colour for a named partition, raising on an unknown name."""
    try:
        return PARTITION_COLOURS[partition]
    except KeyError as exc:
        raise KeyError(
            f"no colour defined for partition {partition!r}; known: {sorted(PARTITION_COLOURS)}"
        ) from exc


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    width_in: float = 6.5,
    height_in: float = 4.0,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a figure with the house style applied.

    The default width is 6.5 inches, which is a single column of the PSB
    two-column layout at full bleed. Set it explicitly for anything else rather
    than scaling afterwards, which changes the effective font size.
    """
    apply_house_style()
    return plt.subplots(nrows, ncols, figsize=(width_in, height_in), **kwargs)


def save_figure(
    fig: Figure,
    path: Path,
    manifest: Manifest,
    title: str | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    stamp_footer: bool = True,
) -> list[Path]:
    """Write a figure with its provenance attached, in every requested format.

    The manifest hash goes into the file metadata via
    :func:`interface_charge.provenance.figure_metadata`, and unless
    ``stamp_footer`` is disabled a truncated hash is also drawn in the bottom
    left corner. The visible stamp is what survives a screenshot; the metadata
    is what survives a format conversion. Both are cheap.

    Returns the paths written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if stamp_footer:
        fig.text(
            0.005,
            0.005,
            f"manifest {manifest.short_hash}"
            + ("  [DIRTY TREE]" if manifest.git.get("dirty") else ""),
            fontsize=5,
            color="#999999",
            ha="left",
            va="bottom",
        )

    metadata = figure_metadata(manifest, title=title)
    written: list[Path] = []
    for fmt in formats:
        target = path.with_suffix(f".{fmt}")
        # Matplotlib only accepts the standard document keys for PDF and SVG.
        # PNG takes arbitrary text chunks, so the full set goes there.
        if fmt == "png":
            fig.savefig(target, format=fmt, metadata=metadata)
        else:
            fig.savefig(
                target,
                format=fmt,
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key in {"Title", "Author", "Subject", "Keywords", "Creator"}
                },
            )
        written.append(target)
        manifest.add_output(target)

    plt.close(fig)
    return written


def annotate_definition(ax: Axes, definition: str, ph: float | None = None) -> None:
    """Label an axis with the charge definition it uses.

    Used on every charge axis. The two definitions differ by several charge
    units, so an unlabelled charge axis is ambiguous, and an ambiguous axis in
    a paper about charge is a referee's first question.
    """
    if definition == "simple":
        label = "net charge, (K+R) - (D+E)"
    elif definition == "ph":
        label = f"net charge at pH {ph:g}" if ph is not None else "net charge, pH-based"
    else:
        raise ValueError(f"unknown charge definition {definition!r}")
    ax.set_xlabel(label)
