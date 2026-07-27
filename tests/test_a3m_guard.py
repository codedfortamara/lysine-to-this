"""The alignment handed to ColabFold must be one ColabFold will actually use.

ColabFold's ``unserialize_msa`` validates the a3m header, and when it fails the
check it does **not** raise. It falls through to a single-sequence branch and
returns an alignment of depth one. Every number downstream then looks entirely
normal while the MSA has been thrown away.

On this grid that would mean paying for 275 GPU jobs and reporting interface
confidences computed against no alignment at all, with nothing in the output to
show it. The guard exists because the failure is silent, not because it is
likely.

``colabfold_accepts_as_complex`` below reproduces ColabFold 1.5.5's own check,
transcribed from ``colabfold/batch.py``. The tests assert that the guard rejects
everything that check would quietly reinterpret.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))

from af2_multimer import require_parseable_complex_a3m

LENGTHS = [194, 597]


def good_a3m(lengths: list[int] = LENGTHS) -> str:
    """The shape ``msa_to_str`` produces: header, query name, query, then hits."""
    total = sum(lengths)
    header = "#" + ",".join(str(n) for n in lengths) + "\t" + ",".join("1" for _ in lengths)
    return f"{header}\n>101\t102\n{'A' * total}\n>hit1\n{'A' * total}\n"


def colabfold_accepts_as_complex(a3m_lines: list[str]) -> bool:
    """ColabFold 1.5.5's check, transcribed. False means the MSA is discarded."""
    lines = a3m_lines[0].replace("\x00", "").splitlines()
    return lines[0].startswith("#") and len(lines[0][1:].split("\t")) == 2


# ---------------------------------------------------------------------------
# The bug this was written for
# ---------------------------------------------------------------------------


def test_a_bare_string_is_what_colabfold_silently_rejects() -> None:
    """Passing the a3m as a string rather than a list discards the alignment.

    ColabFold indexes ``a3m_lines[0]``. Given a list that is the whole
    alignment; given a string it is the single character "#". The check then
    fails and the single-sequence fallback runs, without raising.
    """
    a3m = good_a3m()
    assert colabfold_accepts_as_complex([a3m]), "a one-element list must be accepted"
    assert not colabfold_accepts_as_complex(a3m), (
        "a bare string must NOT parse as a complex a3m; if this ever passes, the "
        "silent-fallback hazard has gone away and the wrapping is no longer load bearing"
    )


def test_the_guard_accepts_a_well_formed_alignment() -> None:
    require_parseable_complex_a3m(good_a3m(), LENGTHS)


# ---------------------------------------------------------------------------
# Everything the guard must refuse
# ---------------------------------------------------------------------------


def test_a_header_without_the_hash_is_refused() -> None:
    a3m = good_a3m().replace("#", "", 1)
    assert not colabfold_accepts_as_complex([a3m])
    with pytest.raises(ValueError, match="start"):
        require_parseable_complex_a3m(a3m, LENGTHS)


def test_a_header_without_the_tab_is_refused() -> None:
    """A space instead of a tab reads as one field, and the MSA is dropped."""
    a3m = good_a3m().replace("\t", " ", 1)
    assert not colabfold_accepts_as_complex([a3m])
    with pytest.raises(ValueError, match="tab-separated"):
        require_parseable_complex_a3m(a3m, LENGTHS)


def test_an_alignment_too_short_to_parse_is_refused() -> None:
    with pytest.raises(ValueError, match="at least three"):
        require_parseable_complex_a3m("#194,597\t1,1\n>101\n", LENGTHS)


def test_lengths_disagreeing_with_the_job_are_refused() -> None:
    """This one ColabFold would accept, and slice against the wrong residues.

    The header parses, so no fallback fires, but the declared lengths decide
    where one chain ends and the next begins. Wrong lengths mean every hit is
    split at the wrong column.
    """
    a3m = good_a3m([100, 200])
    assert colabfold_accepts_as_complex([a3m]), "ColabFold would accept this happily"
    with pytest.raises(ValueError, match="declares chain lengths"):
        require_parseable_complex_a3m(a3m, LENGTHS)


def test_a_chain_count_mismatch_between_the_header_fields_is_refused() -> None:
    a3m = "#194,597\t1\n>101\t102\n" + "A" * 791 + "\n"
    with pytest.raises(ValueError, match="cardinality"):
        require_parseable_complex_a3m(a3m, LENGTHS)


def test_the_guard_is_strict_wherever_colabfold_is_silent() -> None:
    """Every malformed case is refused loudly rather than reinterpreted."""
    broken = [
        good_a3m().replace("#", "", 1),
        good_a3m().replace("\t", " ", 1),
        good_a3m([100, 200]),
        "#194,597\t1,1\n>101\n",
    ]
    for a3m in broken:
        with pytest.raises(ValueError):
            require_parseable_complex_a3m(a3m, LENGTHS)


def test_a_single_chain_alignment_is_accepted_when_it_matches() -> None:
    require_parseable_complex_a3m(good_a3m([250]), [250])
