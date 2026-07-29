"""Native and predicted residues are paired by sequence, never by residue number.

Found by adversarial review, and confirmed against the real structures in this
set. The native complexes come from the PDB with author numbering: 1FS2 chain A
starts at residue 105, 1O9S at 117, 1I7Q at 4, and several chains have numbering
gaps. ColabFold numbers its output 1..N by sequence position.

Matching on the number therefore compared native position 1 against predicted
position 105 and returned a plausible RMSD, or found no overlap at all and
returned nothing. Neither outcome announces itself, which is what makes it worth
a test file: an interface RMSD of 4.2 angstroms computed from the wrong residues
looks exactly like one computed from the right ones.

Insertion codes are the sharper form of the same bug. A dictionary keyed on the
integer alone collapses 52, 52A and 52B onto a single entry.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))

from af2_multimer import pair_residues_by_sequence

THREE_LETTER = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


@dataclass
class FakeResidue:
    """The two things pair_residues_by_sequence reads off a Bio.PDB residue."""

    letter: str
    seqid: int
    icode: str = " "
    id: tuple = field(init=False)

    def __post_init__(self) -> None:
        self.id = (" ", self.seqid, self.icode)

    def get_resname(self) -> str:
        return THREE_LETTER[self.letter]


def chain(sequence: str, start: int = 1) -> list[FakeResidue]:
    return [FakeResidue(letter, start + i) for i, letter in enumerate(sequence)]


SEQUENCE = "MKVLTAWGDEKRSNPFYQCMKVLTAWGDE"


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------


def test_an_author_numbering_offset_does_not_break_the_pairing() -> None:
    """1FS2 chain A starts at 105; the prediction starts at 1. Same residues."""
    native = chain(SEQUENCE, start=105)
    predicted = chain(SEQUENCE, start=1)

    pairs = pair_residues_by_sequence(native, predicted)

    assert len(pairs) == len(SEQUENCE)
    for native_residue, predicted_residue in pairs:
        assert native_residue.letter == predicted_residue.letter


def test_pairing_by_number_would_have_been_wrong_here() -> None:
    """States the defect explicitly, so the test names what it is preventing.

    With the native at 105.. and the prediction at 1.., the overlapping numbers
    are 105 to 133. Pairing on them compares native position 1 with predicted
    position 105, which is a different residue, and produces a number rather
    than an error.
    """
    native = chain(SEQUENCE, start=105)
    predicted = chain(SEQUENCE, start=1)

    by_number = set(r.id[1] for r in native) & set(r.id[1] for r in predicted)
    assert not by_number, "no numbers overlap at all here, so the old code got nothing"

    assert len(pair_residues_by_sequence(native, predicted)) == len(SEQUENCE)


def test_the_pairing_is_positional_not_coincidental() -> None:
    """Each native residue must map to the prediction of the same position."""
    native = chain(SEQUENCE, start=117)
    predicted = chain(SEQUENCE, start=1)

    for index, (native_residue, predicted_residue) in enumerate(
        pair_residues_by_sequence(native, predicted)
    ):
        assert native_residue.seqid == 117 + index
        assert predicted_residue.seqid == 1 + index


# ---------------------------------------------------------------------------
# Gaps and insertion codes, both present in this set
# ---------------------------------------------------------------------------


def test_a_gap_in_the_native_numbering_is_handled() -> None:
    """1FS2, 1HL6, 1I7Q and 1I7S all have unmodelled stretches."""
    native = chain(SEQUENCE[:10], start=8) + chain(SEQUENCE[10:], start=200)
    predicted = chain(SEQUENCE, start=1)

    pairs = pair_residues_by_sequence(native, predicted)

    assert len(pairs) == len(SEQUENCE)
    assert [n.letter for n, _ in pairs] == list(SEQUENCE)


def test_missing_native_residues_shrink_the_pairing_rather_than_misalign_it() -> None:
    """An unmodelled loop must not shift everything after it by its length."""
    observed = SEQUENCE[:10] + SEQUENCE[15:]
    native = chain(observed, start=1)
    predicted = chain(SEQUENCE, start=1)

    pairs = pair_residues_by_sequence(native, predicted)

    assert len(pairs) == len(observed)
    # The residues after the gap must map past it, not five positions early.
    tail_native, tail_predicted = pairs[-1]
    assert tail_native.letter == tail_predicted.letter == SEQUENCE[-1]
    assert tail_predicted.seqid == len(SEQUENCE)


def test_insertion_codes_are_distinct_residues() -> None:
    """52, 52A and 52B are three residues, and an integer key collapses them."""
    native = chain(SEQUENCE[:5], start=50)
    native.append(FakeResidue(SEQUENCE[5], seqid=52, icode="A"))
    native.append(FakeResidue(SEQUENCE[6], seqid=52, icode="B"))
    native.extend(chain(SEQUENCE[7:], start=60))
    predicted = chain(SEQUENCE, start=1)

    pairs = pair_residues_by_sequence(native, predicted)

    assert len(pairs) == len(SEQUENCE)
    inserted = [n for n, _ in pairs if n.icode.strip()]
    assert len(inserted) == 2, "both insertion-coded residues must survive as separate pairs"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_residues_whose_identity_disagrees_are_dropped() -> None:
    """An alignment can place a gap plausibly and still be wrong.

    Two residues that are not the same amino acid are not the same residue, so
    they are excluded rather than measured.
    """
    native = chain(SEQUENCE, start=1)
    predicted = chain(SEQUENCE, start=1)
    predicted[3] = FakeResidue("W", seqid=4)

    pairs = pair_residues_by_sequence(native, predicted)

    assert all(n.letter == p.letter for n, p in pairs)
    assert len(pairs) == len(SEQUENCE) - 1


def test_an_empty_chain_pairs_with_nothing() -> None:
    assert pair_residues_by_sequence([], chain(SEQUENCE)) == []
    assert pair_residues_by_sequence(chain(SEQUENCE), []) == []


def test_completely_unrelated_sequences_yield_almost_no_pairs() -> None:
    """A wrong chain mapping must not quietly produce a full set of pairs.

    Not zero: the native contains a single glycine, which legitimately matches
    one of the prediction's. One pair out of nineteen is the identity check
    doing its job. Nineteen would mean it was not running.
    """
    native = chain("MKVLTAWGDEKRSNPFYQC", start=1)
    predicted = chain("GGGGGGGGGGGGGGGGGGG", start=1)
    pairs = pair_residues_by_sequence(native, predicted)
    assert len(pairs) <= 2, f"{len(pairs)} of 19 paired between unrelated chains"
    assert all(n.letter == p.letter for n, p in pairs)
