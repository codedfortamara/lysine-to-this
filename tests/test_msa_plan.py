"""The mixed single-sequence and MSA treatment.

Only the decision logic is tested here. The alignment assembly itself calls
ColabFold inside the Modal container and is deliberately not reimplemented, so
there is nothing local to exercise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))

from af2_multimer import Job, msa_plan, pairing_is_meaningful


def job(msa_mode: dict[str, str] | None = None, designed: str = "A") -> Job:
    return Job(
        pdb_id="1ABC",
        beta=1.5,
        replicate=0,
        designed_chain=designed,
        chains={"A": "PEPTIDEK", "B": "PARTNERR"},
        msa_mode=msa_mode if msa_mode is not None else {"A": "single_sequence", "B": "msa"},
    )


def test_the_intended_arrangement_passes() -> None:
    modes = msa_plan(job())
    assert modes == {"A": "single_sequence", "B": "msa"}


def test_giving_the_designed_chain_an_msa_raises() -> None:
    """The failure this guard exists for, and it would not otherwise raise.

    An MSA for a design returns homologues of the native it came from, so
    AlphaFold would predict the native's fold and a design that ought to fail
    could be propped up into looking fine. That produces a number, not an error,
    which is exactly why it is checked rather than trusted.
    """
    with pytest.raises(ValueError, match="no evolutionary history"):
        msa_plan(job({"A": "msa", "B": "msa"}))


def test_a_chain_with_no_recorded_mode_raises() -> None:
    with pytest.raises(ValueError, match="no MSA mode"):
        msa_plan(job({"A": "single_sequence"}))


def test_an_unrecognised_mode_raises() -> None:
    with pytest.raises(ValueError, match="unrecognised"):
        msa_plan(job({"A": "single_sequence", "B": "profile"}))


def test_the_partner_may_also_be_held_at_single_sequence() -> None:
    """Permitted, since it never overstates what the design knows."""
    assert msa_plan(job({"A": "single_sequence", "B": "single_sequence"}))


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_pairing_is_never_meaningful_when_a_chain_is_designed() -> None:
    """A design belongs to no organism, so nothing can be paired to it.

    Every job in this grid has one designed chain, so this is uniformly false
    and the limitation applies to the whole run. It is recorded per job rather
    than assumed, so the results table can state it.
    """
    assert not pairing_is_meaningful(msa_plan(job()))


def test_pairing_is_meaningful_only_when_every_chain_has_an_msa() -> None:
    assert pairing_is_meaningful({"A": "msa", "B": "msa"})
    assert not pairing_is_meaningful({"A": "msa", "B": "single_sequence"})
