"""Schema validation: the contract must reject malformed input, loudly.

The rule under test throughout is that nothing is coerced, filled or dropped.
Every case here is a way a real delivered CSV goes wrong, and every one of them
must raise rather than produce a plausible-looking table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from interface_charge.contracts import (
    DESIGNS_SPEC,
    TEST_SET_SPEC,
    SchemaError,
    check_cross_table,
    describe_contract,
    load_designs,
    load_test_set,
    parse_chain_spec,
    parse_float,
    parse_int,
    parse_pdb_id,
    parse_sequence,
    parse_split,
)

DESIGNS_HEADER = "pdb_id,designed_chain,fixed_chain,beta,replicate,sequence,net_charge_reported"
TEST_SET_HEADER = "pdb_id,chains,mmseqs_cluster_id,split"


def write(path: Path, header: str, *rows: str) -> Path:
    path.write_text(header + "\n" + "\n".join(rows) + ("\n" if rows else ""))
    return path


@pytest.fixture
def designs(tmp_path):
    def _write(*rows: str) -> Path:
        return write(tmp_path / "designs.csv", DESIGNS_HEADER, *rows)

    return _write


@pytest.fixture
def test_set(tmp_path):
    def _write(*rows: str) -> Path:
        return write(tmp_path / "test_set.csv", TEST_SET_HEADER, *rows)

    return _write


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_valid_designs_load(designs) -> None:
    frame = load_designs(
        designs(
            "1BRS,A,D,0.0,0,MKKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGCDITIILS,-2.0"
        )
    )
    assert len(frame) == 1
    assert frame.loc[0, "pdb_id"] == "1BRS"
    assert frame.loc[0, "designed_chain"] == ("A",)
    assert frame.loc[0, "beta"] == 0.0
    assert frame.loc[0, "replicate"] == 0


def test_valid_test_set_loads(test_set) -> None:
    frame = load_test_set(test_set('1BRS,"A,D",cluster_1,test'))
    assert frame.loc[0, "chains"] == ("A", "D")
    assert frame.loc[0, "split"] == "test"


def test_pdb_id_is_upper_cased(designs) -> None:
    """The one documented normalisation."""
    frame = load_designs(designs("1brs,A,D,0.0,0,MKKAV,-2.0"))
    assert frame.loc[0, "pdb_id"] == "1BRS"


# ---------------------------------------------------------------------------
# Missing and extra columns
# ---------------------------------------------------------------------------


def test_missing_file_raises_with_the_contract_in_the_message(tmp_path) -> None:
    with pytest.raises(SchemaError) as info:
        load_designs(tmp_path / "nope.csv")
    message = str(info.value)
    assert "not found" in message
    for column in DESIGNS_SPEC.column_names:
        assert column in message


def test_missing_column_raises_and_names_it(tmp_path) -> None:
    path = write(
        tmp_path / "designs.csv",
        "pdb_id,designed_chain,fixed_chain,beta,replicate,sequence",
        "1BRS,A,D,0.0,0,MKKAV",
    )
    with pytest.raises(SchemaError, match="net_charge_reported"):
        load_designs(path)


def test_extra_column_raises_by_default(tmp_path) -> None:
    path = write(
        tmp_path / "designs.csv",
        DESIGNS_HEADER + ",surprise",
        "1BRS,A,D,0.0,0,MKKAV,-2.0,hello",
    )
    with pytest.raises(SchemaError, match="unexpected column"):
        load_designs(path)


def test_extra_column_allowed_when_requested(tmp_path) -> None:
    path = write(
        tmp_path / "designs.csv",
        DESIGNS_HEADER + ",surprise",
        "1BRS,A,D,0.0,0,MKKAV,-2.0,hello",
    )
    frame = load_designs(path, allow_extra_columns=True)
    assert frame.loc[0, "surprise"] == "hello"


def test_duplicate_column_names_raise(tmp_path) -> None:
    path = write(
        tmp_path / "designs.csv",
        DESIGNS_HEADER + ",beta",
        "1BRS,A,D,0.0,0,MKKAV,-2.0,9.9",
    )
    with pytest.raises(SchemaError, match="duplicate column"):
        load_designs(path)


def test_empty_file_raises(tmp_path) -> None:
    path = tmp_path / "designs.csv"
    path.write_text("")
    with pytest.raises(SchemaError, match="empty"):
        load_designs(path)


# ---------------------------------------------------------------------------
# Malformed cells. Nothing may be coerced, filled or dropped.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "row",
    [
        "XXXX,A,D,0.0,0,MKKAV,-2.0",  # pdb_id does not start with a digit
        "1BR,A,D,0.0,0,MKKAV,-2.0",  # too short
        "1BRSX,A,D,0.0,0,MKKAV,-2.0",  # too long
        ",A,D,0.0,0,MKKAV,-2.0",  # empty
        "1BRS,A,D,notanumber,0,MKKAV,-2.0",  # beta not numeric
        "1BRS,A,D,,0,MKKAV,-2.0",  # beta empty
        "1BRS,A,D,nan,0,MKKAV,-2.0",  # beta NaN
        "1BRS,A,D,inf,0,MKKAV,-2.0",  # beta infinite
        "1BRS,A,D,0.0,-1,MKKAV,-2.0",  # negative replicate
        "1BRS,A,D,0.0,1.5,MKKAV,-2.0",  # non-integer replicate
        "1BRS,A,D,0.0,3.0,MKKAV,-2.0",  # integral-looking float replicate
        "1BRS,A,D,0.0,0,MKKAVX,-2.0",  # non-standard residue
        "1BRS,A,D,0.0,0,,-2.0",  # empty sequence
        "1BRS,A,D,0.0,0,MKKAV,",  # empty reported charge
        "1BRS,A,D,0.0,0,MKKAV,N/A",  # placeholder reported charge
        "1BRS,,D,0.0,0,MKKAV,-2.0",  # empty designed chain
        "1BRS,A,A,0.0,0,MKKAV,-2.0",  # designed and fixed chain identical
    ],
)
def test_malformed_row_raises(designs, row: str) -> None:
    with pytest.raises(SchemaError):
        load_designs(designs(row))


def test_placeholder_values_are_not_treated_as_missing(designs) -> None:
    """'NA' must raise rather than becoming NaN, which pandas would do by default."""
    with pytest.raises(SchemaError, match="placeholder"):
        load_designs(designs("1BRS,A,D,NA,0,MKKAV,-2.0"))


def test_all_errors_are_reported_at_once(designs) -> None:
    """Fixing a delivered dataset one exception at a time is miserable."""
    path = designs(
        "1BRS,A,D,bad,0,MKKAV,-2.0",
        "1BRS,A,D,0.0,bad,MKKAV,-2.0",
        "1BRS,A,D,0.0,0,MKKAVX,-2.0",
    )
    with pytest.raises(SchemaError) as info:
        load_designs(path)
    message = str(info.value)
    assert "3 problem(s)" in message
    # Line numbers must match what a person sees in a spreadsheet, where the
    # header is line 1.
    assert "line 2" in message
    assert "line 3" in message
    assert "line 4" in message


def test_row_count_is_never_reduced_silently(designs) -> None:
    """A malformed row must raise, never be dropped, leaving a shorter table."""
    good = "1BRS,A,D,0.0,0,MKKAV,-2.0"
    frame = load_designs(designs(good, "1BRS,A,D,1.0,0,MKKAV,-2.0"))
    assert len(frame) == 2


# ---------------------------------------------------------------------------
# Table-level invariants
# ---------------------------------------------------------------------------


def test_duplicate_design_key_raises(designs) -> None:
    row = "1BRS,A,D,0.0,0,MKKAV,-2.0"
    with pytest.raises(SchemaError, match="unique"):
        load_designs(designs(row, row))


def test_inconsistent_sequence_length_raises(designs) -> None:
    """ProteinMPNN is fixed-backbone, so lengths cannot vary within a chain."""
    with pytest.raises(SchemaError, match="fixed-backbone"):
        load_designs(
            designs(
                "1BRS,A,D,0.0,0,MKKAV,-2.0",
                "1BRS,A,D,1.0,0,MKKAVGG,-2.0",
            )
        )


def test_multi_chain_designed_column_parses(designs) -> None:
    frame = load_designs(designs('1BRS,"A,B",D,0.0,0,MKKAV,-2.0'))
    assert frame.loc[0, "designed_chain"] == ("A", "B")


def test_overlapping_designed_and_fixed_chains_raise(designs) -> None:
    with pytest.raises(SchemaError, match="both"):
        load_designs(designs('1BRS,"A,B","B,C",0.0,0,MKKAV,-2.0'))


def test_test_set_requires_exactly_two_chains(test_set) -> None:
    with pytest.raises(SchemaError, match="exactly two chains"):
        load_test_set(test_set('1BRS,"A,B,C",cluster_1,test'))


def test_duplicate_pdb_id_in_test_set_raises(test_set) -> None:
    with pytest.raises(SchemaError, match="duplicate"):
        load_test_set(test_set('1BRS,"A,D",c1,test', '1BRS,"A,D",c1,test'))


def test_cluster_straddling_the_split_raises(test_set) -> None:
    """Sequence leakage invalidates the held-out claim, so it is an error."""
    with pytest.raises(SchemaError, match="leakage"):
        load_test_set(test_set('1BRS,"A,D",shared,test', '2BRS,"A,D",shared,train'))


def test_invalid_split_value_raises(test_set) -> None:
    with pytest.raises(SchemaError, match="train"):
        load_test_set(test_set('1BRS,"A,D",c1,validation'))


# ---------------------------------------------------------------------------
# Cross-table consistency
# ---------------------------------------------------------------------------


def test_cross_table_accepts_consistent_tables(designs, test_set) -> None:
    frame_designs = load_designs(designs("1BRS,A,D,0.0,0,MKKAV,-2.0"))
    frame_test = load_test_set(test_set('1BRS,"A,D",c1,test'))
    check_cross_table(frame_designs, frame_test)  # must not raise


def test_design_for_unknown_complex_raises(designs, test_set) -> None:
    frame_designs = load_designs(designs("9XYZ,A,D,0.0,0,MKKAV,-2.0"))
    frame_test = load_test_set(test_set('1BRS,"A,D",c1,test'))
    with pytest.raises(SchemaError, match="not in test_set"):
        check_cross_table(frame_designs, frame_test)


def test_design_naming_an_unlisted_chain_raises(designs, test_set) -> None:
    frame_designs = load_designs(designs("1BRS,A,Z,0.0,0,MKKAV,-2.0"))
    frame_test = load_test_set(test_set('1BRS,"A,D",c1,test'))
    with pytest.raises(SchemaError, match="does not list"):
        check_cross_table(frame_designs, frame_test)


# ---------------------------------------------------------------------------
# Individual parsers
# ---------------------------------------------------------------------------


def test_parse_pdb_id_accepts_canonical_ids() -> None:
    assert parse_pdb_id("1brs") == "1BRS"
    assert parse_pdb_id(" 7DDO ") == "7DDO"


@pytest.mark.parametrize("bad", ["ABCD", "1BR", "1BRSS", "", "  ", "1-RS"])
def test_parse_pdb_id_rejects_bad_ids(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_pdb_id(bad)


def test_parse_chain_spec_rejects_repeats() -> None:
    with pytest.raises(ValueError, match="repeats"):
        parse_chain_spec("A,A", "designed_chain")


def test_parse_chain_spec_rejects_empty_element() -> None:
    with pytest.raises(ValueError, match="empty chain"):
        parse_chain_spec("A,,B", "designed_chain")


def test_parse_int_enforces_minimum() -> None:
    with pytest.raises(ValueError, match="minimum"):
        parse_int("-1", "replicate", minimum=0)


def test_parse_float_rejects_infinities() -> None:
    for value in ("inf", "-inf", "Infinity"):
        with pytest.raises(ValueError, match="finite"):
            parse_float(value, "beta")


def test_parse_float_rejects_nan_as_a_placeholder() -> None:
    """'nan' is caught by the placeholder check, which gives the better message.

    Either rejection would be correct. The placeholder message is the more
    useful one because a literal 'nan' in a delivered CSV almost always means an
    upstream calculation produced a missing value, not that a genuine
    not-a-number was intended.
    """
    with pytest.raises(ValueError, match="placeholder"):
        parse_float("nan", "beta")


def test_parse_sequence_strips_spaces_and_upper_cases() -> None:
    assert parse_sequence("mk kav") == "MKKAV"


def test_parse_split_normalises_case() -> None:
    assert parse_split("TEST") == "test"


# ---------------------------------------------------------------------------
# Documentation stays in step with the code
# ---------------------------------------------------------------------------


def test_data_readme_documents_every_contracted_column() -> None:
    """The contract in prose and the contract in code must not drift apart.

    data/README.md is what the collaborator reads. If a column is added here
    without being documented there, this fails.
    """
    readme = (Path(__file__).parents[1] / "data" / "README.md").read_text()
    for spec in (DESIGNS_SPEC, TEST_SET_SPEC):
        assert spec.name in readme
        for column in spec.column_names:
            assert column in readme, f"{column} of {spec.name} is not documented in data/README.md"


def test_describe_contract_mentions_every_column() -> None:
    text = describe_contract(DESIGNS_SPEC)
    for column in DESIGNS_SPEC.column_names:
        assert column in text
