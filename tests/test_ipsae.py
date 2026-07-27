"""ipSAE, and the property it exists to provide.

The implementation in ``modal_app/af2_multimer.py`` follows the reference at
github.com/DunbrackLab/IPSAE. These tests pin the behaviour that matters for
this project rather than a set of numbers from a previous run: chiefly that the
score does not move when the chains around a fixed interface change length,
which is precisely what ipTM fails to do and the reason ipTM cannot be compared
across a set of complexes of differing size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))

from af2_multimer import _d0_array, _d0_scalar, ipsae

CUTOFF = 10.0


def patch_pae(len_a: int, len_b: int, patch: int, cross_pae: float = 4.0) -> np.ndarray:
    """A complex where only a patch of each chain is confidently placed across.

    Within-chain blocks are confident, the cross-chain block is confident only
    over ``patch`` residues at the chain junction, and everything else is far
    above the cutoff. This is the shape of a real two-domain interface.
    """
    total = len_a + len_b
    pae = np.full((total, total), 25.0)
    pae[:len_a, :len_a] = 3.0
    pae[len_a:, len_a:] = 3.0
    pae[len_a - patch : len_a, len_a : len_a + patch] = cross_pae
    pae[len_a : len_a + patch, len_a - patch : len_a] = cross_pae
    np.fill_diagonal(pae, 0.3)
    return pae


# ---------------------------------------------------------------------------
# The length normalisation
# ---------------------------------------------------------------------------


def test_scalar_d0_matches_yang_and_skolnick_above_the_cut() -> None:
    assert _d0_scalar(200) == pytest.approx(1.24 * (200 - 15) ** (1 / 3) - 1.8)


def test_scalar_d0_is_flat_at_and_below_27_residues() -> None:
    """The reference returns exactly 1.0 there rather than evaluating the root."""
    assert _d0_scalar(27) == 1.0
    assert _d0_scalar(5) == 1.0
    assert _d0_scalar(28) > 1.0


def test_array_d0_floors_the_length_not_the_result() -> None:
    """The array form differs from the scalar form below 28 residues.

    Keeping the difference is deliberate: the reference applies each in a
    specific place, and quietly unifying them would change the score.
    """
    assert _d0_array(10) == pytest.approx(_d0_array(26))
    assert _d0_array(27) != pytest.approx(_d0_scalar(27))


def test_d0_grows_with_length() -> None:
    assert _d0_scalar(500) > _d0_scalar(200) > _d0_scalar(50)


# ---------------------------------------------------------------------------
# The property the metric is being adopted for
# ---------------------------------------------------------------------------


def test_ipsae_is_invariant_to_chain_length_around_a_fixed_interface() -> None:
    """The whole reason ipSAE is in this project.

    Three complexes with an identical confident interface patch, differing only
    in how much unrelated chain surrounds it. ipSAE must not move. The d0chn
    variant, whose normalisation comes from the chain lengths, must move, and it
    is kept precisely so this contrast can be shown.
    """
    scores = [
        ipsae(patch_pae(a, b, 12), [a, b], CUTOFF) for a, b in [(80, 40), (200, 60), (400, 200)]
    ]

    fixed = {round(s["ipsae_d0res"], 10) for s in scores}
    assert len(fixed) == 1, f"ipSAE moved with chain length: {fixed}"

    varies = {round(s["ipsae_d0chn"], 6) for s in scores}
    assert len(varies) == 3, f"d0chn should track chain length, got {varies}"


def test_the_chain_normalised_variant_rises_with_irrelevant_chain() -> None:
    """Adding chain that has nothing to do with binding inflates the d0chn score.

    This is the confound in concrete form: a bigger complex looks like a better
    interface under a chain-length normalisation, with no change at the
    interface at all.
    """
    small = ipsae(patch_pae(80, 40, 12), [80, 40], CUTOFF)
    large = ipsae(patch_pae(400, 200, 12), [400, 200], CUTOFF)
    assert large["ipsae_d0chn"] > small["ipsae_d0chn"]
    assert large["ipsae_d0res"] == pytest.approx(small["ipsae_d0res"])


def test_interface_residue_count_tracks_the_patch_not_the_chains() -> None:
    for patch in (8, 12, 20):
        scores = ipsae(patch_pae(200, 60, patch), [200, 60], CUTOFF)
        assert scores["ipsae_n_interface_residues_0to1"] == 2 * patch


# ---------------------------------------------------------------------------
# Ordering and bounds
# ---------------------------------------------------------------------------


def test_a_confident_interface_scores_above_a_poor_one() -> None:
    good = ipsae(patch_pae(150, 90, 20, cross_pae=1.0), [150, 90], CUTOFF)
    poor = ipsae(patch_pae(150, 90, 20, cross_pae=9.0), [150, 90], CUTOFF)
    assert good["ipsae_d0res"] > poor["ipsae_d0res"]


def test_no_confident_cross_chain_pair_scores_zero() -> None:
    """A complex with nothing placed across the interface must score zero, not fail."""
    total = 120
    pae = np.full((total, total), 25.0)
    np.fill_diagonal(pae, 0.3)
    scores = ipsae(pae, [70, 50], CUTOFF)
    assert scores["ipsae_d0res"] == 0.0
    assert scores["ipsae_n_interface_residues_0to1"] == 0


def test_scores_are_bounded() -> None:
    scores = ipsae(patch_pae(150, 90, 20, cross_pae=0.1), [150, 90], CUTOFF)
    for key in ("ipsae_d0res", "ipsae_d0dom", "ipsae_d0chn"):
        assert 0.0 <= scores[key] <= 1.0


def test_the_reported_score_is_the_better_of_the_two_directions() -> None:
    scores = ipsae(patch_pae(200, 60, 12), [200, 60], CUTOFF)
    assert scores["ipsae_d0res"] == max(scores["ipsae_d0res_0to1"], scores["ipsae_d0res_1to0"])


def test_a_wider_cutoff_admits_more_pairs() -> None:
    tight = ipsae(patch_pae(150, 90, 20, cross_pae=12.0), [150, 90], 10.0)
    loose = ipsae(patch_pae(150, 90, 20, cross_pae=12.0), [150, 90], 15.0)
    assert tight["ipsae_n_interface_residues_0to1"] == 0
    assert loose["ipsae_n_interface_residues_0to1"] > 0


def test_the_cutoff_is_recorded_with_the_score() -> None:
    """The score is not scale-free in the cutoff, so it travels with it."""
    assert ipsae(patch_pae(150, 90, 20), [150, 90], 12.5)["ipsae_pae_cutoff_a"] == 12.5


# ---------------------------------------------------------------------------
# Refusing to score the wrong thing
# ---------------------------------------------------------------------------


def test_a_pae_matrix_that_does_not_match_the_chains_raises() -> None:
    """Slicing a mismatched matrix would score the wrong residue pairs silently."""
    pae = np.full((100, 100), 5.0)
    with pytest.raises(ValueError, match="expected"):
        ipsae(pae, [70, 50], CUTOFF)


def test_more_than_two_chains_raises() -> None:
    pae = np.full((150, 150), 5.0)
    with pytest.raises(ValueError, match="two-chain"):
        ipsae(pae, [50, 50, 50], CUTOFF)
