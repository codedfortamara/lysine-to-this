"""Charge-hydropathy phase space: does the dial push designs out of foldable space?

Uversky, Gillespie and Fink (2000) showed that natively unfolded proteins occupy
a distinct region of the plane spanned by mean scaled hydropathy and mean
absolute net charge, separated from folded proteins by a straight boundary. A
sequence above the line has too much charge for its hydrophobicity to keep it
globular.

This is the mechanism the charge dial is missing an explanation for. The
foldability collapse at large bias is currently reported as an empirical
observation; placing the designs on this diagram says *why* it happens, and
predicts where it will happen for a sequence that has not been folded yet.

Two things to be careful about
------------------------------
The boundary is a **population-level separator fitted to natural proteins**, not
a hard physical threshold, and it was fitted long before de novo design existed.
Designs crossing it are *predicted* to be disorder-prone, which is a hypothesis
about them rather than a measurement of them. The honest use is to ask whether
crossing the boundary coincides with the observed loss of foldability, which is
a testable correspondence rather than an assumption.

The diagram also uses **mean absolute net charge**, so it is blind to the sign
asymmetry that shows up elsewhere in this project. Positive and negative designs
at the same distance from neutrality land in the same place. That is a real
limitation of the diagram and it is worth reporting alongside, because it means
any asymmetry observed in the data is something the diagram cannot explain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .charge import net_charge_simple, validate_sequence

__all__ = [
    "UverskyPoint",
    "boundary_charge",
    "mean_scaled_hydropathy",
    "place_on_diagram",
]

#: Kyte and Doolittle (1982) hydropathy indices.
KYTE_DOOLITTLE: Final[dict[str, float]] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

#: Uversky normalises the Kyte-Doolittle scale onto [0, 1] by shifting by the
#: minimum and dividing by the range, which for this scale is -4.5 and 9.0.
_KD_MIN: Final[float] = -4.5
_KD_RANGE: Final[float] = 9.0

#: Boundary from Uversky, Gillespie and Fink (2000), Proteins 41:415-427:
#: a sequence is predicted disorder-prone when
#:     mean absolute net charge > 2.785 * mean scaled hydropathy - 1.151
BOUNDARY_SLOPE: Final[float] = 2.785
BOUNDARY_INTERCEPT: Final[float] = -1.151


def mean_scaled_hydropathy(sequence: str) -> float:
    """Mean Kyte-Doolittle hydropathy, normalised onto [0, 1]."""
    validate_sequence(sequence)
    values = [(KYTE_DOOLITTLE[aa] - _KD_MIN) / _KD_RANGE for aa in sequence]
    return float(np.mean(values))


def mean_net_charge(sequence: str) -> float:
    """Mean absolute net charge per residue, the diagram's vertical axis."""
    validate_sequence(sequence)
    return abs(net_charge_simple(sequence)) / len(sequence)


def boundary_charge(hydropathy: float) -> float:
    """Mean net charge at which a sequence of this hydropathy crosses the boundary."""
    return BOUNDARY_SLOPE * hydropathy + BOUNDARY_INTERCEPT


@dataclass(frozen=True, slots=True)
class UverskyPoint:
    """One sequence placed on the charge-hydropathy diagram."""

    mean_scaled_hydropathy: float
    mean_net_charge: float
    signed_net_charge: int
    n_residues: int

    @property
    def boundary(self) -> float:
        """The charge threshold at this sequence's hydropathy."""
        return boundary_charge(self.mean_scaled_hydropathy)

    @property
    def distance_from_boundary(self) -> float:
        """Positive means predicted disorder-prone, negative means predicted globular.

        Reported as a signed distance rather than a boolean so that a design
        sitting a hair over the line is distinguishable from one far above it,
        which a threshold would flatten.
        """
        return self.mean_net_charge - self.boundary

    @property
    def predicted_disordered(self) -> bool:
        return self.distance_from_boundary > 0.0

    def to_row(self) -> dict[str, float | int | bool]:
        return {
            "mean_scaled_hydropathy": self.mean_scaled_hydropathy,
            "mean_net_charge": self.mean_net_charge,
            "signed_net_charge": self.signed_net_charge,
            "n_residues": self.n_residues,
            "uversky_boundary": self.boundary,
            "distance_from_boundary": self.distance_from_boundary,
            "predicted_disordered": self.predicted_disordered,
        }


def place_on_diagram(sequence: str) -> UverskyPoint:
    """Place one sequence on the Uversky charge-hydropathy diagram."""
    validate_sequence(sequence)
    return UverskyPoint(
        mean_scaled_hydropathy=mean_scaled_hydropathy(sequence),
        mean_net_charge=mean_net_charge(sequence),
        signed_net_charge=net_charge_simple(sequence),
        n_residues=len(sequence),
    )
