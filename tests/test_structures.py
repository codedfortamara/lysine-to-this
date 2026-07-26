"""Structure parsing, chain extraction and the residue index map.

The index map is where an off-by-one turns into a wrong charge partition with
no visible symptom, so most of these tests are about that mapping being exact
and about it refusing to guess.
"""

from __future__ import annotations

import pytest

from interface_charge.config import StructureParams
from interface_charge.structures import (
    CHARGE_ALTERING_MODIFICATIONS,
    MODIFIED_RESIDUE_MAP,
    ResidueId,
    StructureError,
    _normalise_pdb_id,
    chains_present,
    extract_chain,
    heavy_atoms,
    load_model,
    submodel,
)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_fixture_loads_and_has_two_chains(model, structure_case) -> None:
    _, chain_a, chain_b = structure_case
    assert set(chains_present(model)) == {chain_a, chain_b}


def test_missing_file_raises_with_a_pointer_to_script_00(tmp_path, config) -> None:
    with pytest.raises(StructureError, match="00_fetch_natives"):
        load_model(tmp_path / "nope.cif", config.structure)


def test_unknown_extension_raises(tmp_path, config) -> None:
    path = tmp_path / "structure.xyz"
    path.write_text("nonsense")
    with pytest.raises(StructureError, match="cannot infer a structure format"):
        load_model(path, config.structure)


def test_model_index_beyond_the_file_raises(structure_case, config) -> None:
    path, _, _ = structure_case
    params = StructureParams(model_index=99)
    with pytest.raises(StructureError, match="model_index"):
        load_model(path, params)


# ---------------------------------------------------------------------------
# Chain extraction
# ---------------------------------------------------------------------------


def test_sequence_length_matches_residue_count(extracts) -> None:
    for extract in extracts:
        assert len(extract.sequence) == len(extract.residues)
        assert len(extract.index_of) == len(extract.residues)


def test_index_map_is_a_bijection_onto_positions(extracts) -> None:
    for extract in extracts:
        assert sorted(extract.index_of.values()) == list(range(len(extract.sequence)))


def test_residue_ids_are_in_sequence_order(extracts) -> None:
    for extract in extracts:
        ids = extract.residue_ids()
        assert len(ids) == len(extract.sequence)
        for position, rid in enumerate(ids):
            assert extract.index_of[rid] == position


def test_sequence_contains_only_standard_residues(extracts) -> None:
    from interface_charge.config import STANDARD_AA

    for extract in extracts:
        assert set(extract.sequence) <= set(STANDARD_AA)


def test_missing_chain_raises_and_lists_what_is_there(model, config) -> None:
    with pytest.raises(StructureError) as info:
        extract_chain(model, "Z", config.structure)
    message = str(info.value)
    assert "not found" in message
    assert "Chains present" in message


def test_positions_of_maps_residue_ids_back(extracts) -> None:
    extract = extracts[0]
    ids = extract.residue_ids()[:5]
    assert extract.positions_of(ids) == frozenset(range(5))


def test_positions_of_refuses_to_drop_an_unknown_residue(extracts) -> None:
    """A dropped interface residue would shrink the interface partition silently."""
    extract = extracts[0]
    with pytest.raises(StructureError, match="not present in chain"):
        extract.positions_of([ResidueId("Q", 99999, " ")])


# ---------------------------------------------------------------------------
# Threading a design onto a native backbone
# ---------------------------------------------------------------------------


def test_with_sequence_preserves_the_index_map(extracts) -> None:
    extract = extracts[0]
    threaded = extract.with_sequence("A" * len(extract.sequence))
    assert threaded.index_of == extract.index_of
    assert threaded.residues == extract.residues
    assert threaded.sequence != extract.sequence


def test_with_sequence_rejects_a_length_mismatch(extracts) -> None:
    """The fixed-backbone assumption, enforced.

    A design of the wrong length would shift every residue index by the
    difference and produce a partition that looks perfectly reasonable.
    """
    extract = extracts[0]
    with pytest.raises(StructureError, match="fixed-backbone"):
        extract.with_sequence("A" * (len(extract.sequence) - 1))


# ---------------------------------------------------------------------------
# Atom selection
# ---------------------------------------------------------------------------


def test_heavy_atoms_excludes_hydrogens(extracts, config) -> None:
    extract = extracts[0]
    for residue in extract.residues[:50]:
        assert all(atom.element != "H" for atom in heavy_atoms(residue, config.structure))


def test_submodel_keeps_only_the_named_chain(model, structure_case, config) -> None:
    _, chain_a, _ = structure_case
    clone = submodel(model, (chain_a,), config.structure)
    assert [chain.id for chain in clone] == [chain_a]


def test_submodel_does_not_mutate_the_original(model, structure_case, config) -> None:
    """A shallow copy here would destroy the complex on the first monomer call."""
    _, chain_a, chain_b = structure_case
    before = set(chains_present(model))
    submodel(model, (chain_a,), config.structure)
    submodel(model, (chain_b,), config.structure)
    assert set(chains_present(model)) == before


def test_submodel_with_an_absent_chain_raises(model, config) -> None:
    with pytest.raises(StructureError, match="not in model"):
        submodel(model, ("Z",), config.structure)


# ---------------------------------------------------------------------------
# Identifier handling
# ---------------------------------------------------------------------------


def test_pdb_id_is_upper_cased() -> None:
    assert _normalise_pdb_id("1brs") == "1BRS"


@pytest.mark.parametrize("bad", ["ABCD", "1BR", "1BRSS", "", "1-RS"])
def test_bad_pdb_id_raises(bad: str) -> None:
    with pytest.raises(StructureError):
        _normalise_pdb_id(bad)


# ---------------------------------------------------------------------------
# Modified residues
# ---------------------------------------------------------------------------


def test_charge_altering_modifications_are_a_subset_of_the_map() -> None:
    """Every code flagged as charge altering must be one we know how to map."""
    assert set(MODIFIED_RESIDUE_MAP) >= CHARGE_ALTERING_MODIFICATIONS


def test_phosphorylated_residues_are_flagged_as_charge_altering() -> None:
    """Mapping a phosphoserine onto serine would lose about two negative charges."""
    for code in ("SEP", "TPO", "PTR"):
        assert code in CHARGE_ALTERING_MODIFICATIONS


def test_selenomethionine_is_not_flagged() -> None:
    """MSE is ubiquitous and carries the same charge as methionine, which is none."""
    assert "MSE" in MODIFIED_RESIDUE_MAP
    assert "MSE" not in CHARGE_ALTERING_MODIFICATIONS
    assert MODIFIED_RESIDUE_MAP["MSE"] == "M"


def test_every_mapped_residue_maps_to_a_standard_letter() -> None:
    from interface_charge.config import STANDARD_AA

    assert set(MODIFIED_RESIDUE_MAP.values()) <= set(STANDARD_AA)
