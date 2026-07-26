"""The data contract, defined once and enforced on load.

The input tables do not exist yet. They arrive from the collaborator. Rather
than guess at their shape and discover the mismatch halfway through an
analysis, the expected shape is written down here, documented in
``data/README.md``, and checked on every load.

Three rules govern everything in this module:

1. **Nothing is coerced silently.** The CSV is read entirely as text with
   pandas' own type inference and NA handling switched off, then every column
   is parsed explicitly by a named function that raises on anything it does not
   recognise. Pandas would happily turn a stray ``"n/a"`` into ``NaN`` and an
   empty ``beta`` cell into a float, and either would change a result without
   changing anything visible.
2. **Nothing is filled or dropped.** A missing value is an error, not something
   to impute. A malformed row is an error, not something to skip.
3. **Every problem is reported at once.** Validation accumulates errors across
   the whole table and raises a single exception listing them, because fixing a
   delivered dataset one exception at a time is miserable.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from .config import STANDARD_AA

__all__ = [
    "DESIGNS_SPEC",
    "TEST_SET_SPEC",
    "ColumnSpec",
    "SchemaError",
    "TableSpec",
    "check_cross_table",
    "describe_contract",
    "load_designs",
    "load_test_set",
]

_MAX_REPORTED_ERRORS: Final[int] = 40
_PDB_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
_CHAIN_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9]{1,4}$")
_STANDARD_SET: Final[frozenset[str]] = frozenset(STANDARD_AA)


class SchemaError(ValueError):
    """Raised when an input table does not satisfy the contract."""


# ---------------------------------------------------------------------------
# Cell parsers
# ---------------------------------------------------------------------------


def _require_nonempty(raw: str, column: str) -> str:
    text = raw.strip()
    if not text:
        raise ValueError(f"{column} is empty. Missing values are not permitted.")
    if text.lower() in {"na", "n/a", "nan", "none", "null", "-"}:
        raise ValueError(
            f"{column} holds the placeholder {raw!r}. Supply a real value or "
            "remove the row. Placeholders are not silently treated as missing."
        )
    return text


def parse_pdb_id(raw: str, column: str = "pdb_id") -> str:
    """Four-character RCSB identifier, upper-cased.

    Case normalisation is the single transformation this project applies
    silently, because the RCSB itself is case insensitive here and every
    downstream file name would otherwise depend on how the collaborator typed
    it. It is documented in ``data/README.md``.
    """
    text = _require_nonempty(raw, column)
    if not _PDB_ID_RE.match(text):
        raise ValueError(
            f"{column}={raw!r} is not a four-character PDB identifier "
            "(a digit followed by three alphanumerics), for example 1BRS"
        )
    return text.upper()


def parse_chain_spec(raw: str, column: str) -> tuple[str, ...]:
    """One or more author chain identifiers, comma separated.

    A single identifier such as ``A`` is the common case. A comma separated
    list such as ``A,B`` is accepted because it is not yet settled whether the
    designed side of every complex is a single chain. That is the open question
    flagged at the top of ``data/README.md``. Whitespace around identifiers is
    stripped; nothing else is altered.
    """
    text = _require_nonempty(raw, column)
    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"{column}={raw!r} contains an empty chain identifier")
    for part in parts:
        if not _CHAIN_RE.match(part):
            raise ValueError(
                f"{column}={raw!r} contains {part!r}, which is not a valid author "
                "chain identifier (one to four alphanumeric characters)"
            )
    if len(set(parts)) != len(parts):
        raise ValueError(f"{column}={raw!r} repeats a chain identifier")
    return tuple(parts)


def parse_float(raw: str, column: str) -> float:
    """A finite float. Rejects blanks, infinities and NaN."""
    text = _require_nonempty(raw, column)
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{column}={raw!r} is not a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{column}={raw!r} is not finite")
    return value


def parse_int(raw: str, column: str, minimum: int | None = None) -> int:
    """An integer. Rejects floats that merely look integral, such as ``3.0``."""
    text = _require_nonempty(raw, column)
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(
            f"{column}={raw!r} is not an integer. If the source wrote it as a "
            "float such as '3.0', fix it upstream rather than rounding here."
        ) from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{column}={value} is below the minimum of {minimum}")
    return value


def parse_sequence(raw: str, column: str = "sequence") -> str:
    """A protein sequence over the standard twenty, upper case.

    Non-standard codes are rejected. An ``X`` treated as neutral would shift the
    net charge of a design without any visible sign, which is precisely the
    failure this project is built to exclude.
    """
    text = _require_nonempty(raw, column).upper().replace(" ", "")
    bad = sorted({ch for ch in text if ch not in _STANDARD_SET})
    if bad:
        raise ValueError(
            f"{column} contains non-standard residue code(s) {bad}. "
            f"Only the twenty standard amino acids ({STANDARD_AA}) are accepted."
        )
    return text


def parse_split(raw: str, column: str = "split") -> str:
    """Either ``train`` or ``test``, lower cased."""
    text = _require_nonempty(raw, column).lower()
    if text not in {"train", "test"}:
        raise ValueError(f"{column}={raw!r} must be either 'train' or 'test'")
    return text


def parse_identifier(raw: str, column: str) -> str:
    """A free-form non-empty identifier, whitespace stripped."""
    return _require_nonempty(raw, column)


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One column of an input table."""

    name: str
    parser: Callable[[str, str], Any]
    description: str
    dtype_note: str


@dataclass(frozen=True, slots=True)
class TableSpec:
    """One input table."""

    name: str
    columns: tuple[ColumnSpec, ...]
    description: str

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def spec_for(self, name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)


DESIGNS_SPEC: Final[TableSpec] = TableSpec(
    name="designs.csv",
    description=(
        "One row per design: a single ProteinMPNN sample of one chain of one "
        "complex at one value of the charge bias beta."
    ),
    columns=(
        ColumnSpec(
            "pdb_id",
            parse_pdb_id,
            "RCSB identifier of the native complex this design was made from.",
            "four-character string, upper-cased on load",
        ),
        ColumnSpec(
            "designed_chain",
            lambda raw, col: parse_chain_spec(raw, col),
            "Author chain identifier(s) that ProteinMPNN was allowed to redesign.",
            "chain id, or comma separated chain ids",
        ),
        ColumnSpec(
            "fixed_chain",
            lambda raw, col: parse_chain_spec(raw, col),
            "Author chain identifier(s) held at the native sequence.",
            "chain id, or comma separated chain ids",
        ),
        ColumnSpec(
            "beta",
            parse_float,
            (
                "Signed scalar logit bias. Added to the K and R logits and "
                "subtracted from the D and E logits at every position. "
                "beta = 0 recovers vanilla ProteinMPNN."
            ),
            "finite float",
        ),
        ColumnSpec(
            "replicate",
            lambda raw, col: parse_int(raw, col, minimum=0),
            "Index distinguishing repeated samples at the same (pdb_id, chain, beta).",
            "non-negative integer",
        ),
        ColumnSpec(
            "sequence",
            parse_sequence,
            (
                "The designed sequence of designed_chain, in the same residue "
                "order as the native chain. Must be the same length as the "
                "native chain, since ProteinMPNN is fixed-backbone."
            ),
            "string over the standard twenty amino acids",
        ),
        ColumnSpec(
            "net_charge_reported",
            parse_float,
            (
                "Net charge as computed by the collaborator's pipeline. Used "
                "only for reconciliation, never as an analysis input. Script 02 "
                "reports which charge definition reproduces this column."
            ),
            "finite float",
        ),
    ),
)


TEST_SET_SPEC: Final[TableSpec] = TableSpec(
    name="test_set.csv",
    description=(
        "One row per native complex, with the MMseqs2 clustering used to hold "
        "the test split out at 30 percent sequence identity."
    ),
    columns=(
        ColumnSpec(
            "pdb_id",
            parse_pdb_id,
            "RCSB identifier of the native complex.",
            "four-character string, upper-cased on load",
        ),
        ColumnSpec(
            "chains",
            lambda raw, col: parse_chain_spec(raw, col),
            (
                "The author chain identifiers making up the complex to analyse, "
                "comma separated. Exactly two chains are required by the "
                "interface analysis."
            ),
            "comma separated chain ids",
        ),
        ColumnSpec(
            "mmseqs_cluster_id",
            parse_identifier,
            "Cluster assigned by MMseqs2 at 30 percent sequence identity.",
            "non-empty string",
        ),
        ColumnSpec(
            "split",
            parse_split,
            "Which side of the leakage-free split this complex falls on.",
            "'train' or 'test'",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_raw(path: Path, spec: TableSpec, allow_extra_columns: bool) -> pd.DataFrame:
    """Read the CSV as pure text and check the header."""
    if not path.is_file():
        raise SchemaError(
            f"{spec.name} not found at {path}.\n"
            f"{spec.description}\n"
            "The expected columns are:\n"
            + describe_contract(spec, indent="  ")
            + "\nSee data/README.md for the full contract. This file is not in "
            "the repository because it is the collaborator's data; it has to be "
            "placed under data/raw/ before any script will run."
        )

    if path.stat().st_size == 0:
        raise SchemaError(f"{path} is empty")

    # Read the header ourselves before pandas sees it. pandas silently renames a
    # repeated column to "beta.1", which would turn a genuine duplicate into
    # something that merely looks like an unexpected extra column, and the
    # resulting error message would send someone hunting for the wrong problem.
    with path.open(newline="") as handle:
        try:
            raw_header = next(csv.reader(handle))
        except StopIteration:
            raise SchemaError(f"{path} has no header row") from None

    header = [name.strip() for name in raw_header]
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise SchemaError(
            f"{path} has duplicate column name(s): {duplicates}. "
            "Two columns with the same name means one of them is being ignored, "
            "and which one is an accident of ordering."
        )

    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        skip_blank_lines=False,
    )

    actual = list(frame.columns)

    expected = set(spec.column_names)
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)

    if missing:
        raise SchemaError(
            f"{path} is missing required column(s): {missing}.\n"
            f"Found: {actual}.\n"
            "Expected columns:\n" + describe_contract(spec, indent="  ")
        )

    if extra and not allow_extra_columns:
        raise SchemaError(
            f"{path} has unexpected column(s): {extra}.\n"
            "The contract is strict by default so that a renamed or duplicated "
            "column cannot pass unnoticed. If these columns are genuinely new "
            "and expected, pass allow_extra_columns=True (the scripts expose "
            "--allow-extra-columns) and record the change in data/README.md."
        )

    return frame


def _parse_table(frame: pd.DataFrame, spec: TableSpec, path: Path) -> pd.DataFrame:
    """Parse every cell through its column parser, accumulating all errors."""
    errors: list[str] = []
    parsed: dict[str, list[Any]] = {name: [] for name in spec.column_names}

    for row_index, row in enumerate(frame.itertuples(index=False), start=2):
        # start=2 because row 1 of the file is the header, which is what a
        # person looking at the CSV in a spreadsheet will see.
        record = dict(zip(frame.columns, row, strict=True))
        for column in spec.columns:
            raw = record[column.name]
            try:
                parsed[column.name].append(column.parser(str(raw), column.name))
            except ValueError as exc:
                parsed[column.name].append(None)
                errors.append(f"  line {row_index}: {exc}")

    if errors:
        shown = errors[:_MAX_REPORTED_ERRORS]
        suffix = (
            f"\n  ... and {len(errors) - _MAX_REPORTED_ERRORS} more"
            if len(errors) > _MAX_REPORTED_ERRORS
            else ""
        )
        raise SchemaError(
            f"{path} failed validation with {len(errors)} problem(s):\n" + "\n".join(shown) + suffix
        )

    out = pd.DataFrame(parsed)
    for name in frame.columns:
        if name not in out.columns:
            out[name] = frame[name].tolist()
    return out


def load_designs(path: Path, allow_extra_columns: bool = False) -> pd.DataFrame:
    """Load and validate ``designs.csv``.

    Beyond the per-cell contract, three table-level invariants are checked:

    * ``(pdb_id, designed_chain, beta, replicate)`` is unique. A duplicate would
      double-weight one design in every average.
    * ``designed_chain`` and ``fixed_chain`` do not overlap. A chain cannot be
      both redesigned and held native.
    * All designs of the same ``(pdb_id, designed_chain)`` have the same
      sequence length. ProteinMPNN is fixed-backbone, so a length change means
      rows have been mixed up or a chain was mislabelled.
    """
    raw = _read_raw(path, DESIGNS_SPEC, allow_extra_columns)
    frame = _parse_table(raw, DESIGNS_SPEC, path)

    problems: list[str] = []

    key_columns = ["pdb_id", "designed_chain", "beta", "replicate"]
    keys = frame[key_columns].astype(str).agg("|".join, axis=1)
    duplicated = keys[keys.duplicated(keep=False)]
    if not duplicated.empty:
        examples = sorted(set(duplicated))[:5]
        problems.append(
            f"{len(duplicated)} row(s) share a (pdb_id, designed_chain, beta, "
            f"replicate) key, for example {examples}. Keys must be unique."
        )

    for index, row in frame.iterrows():
        overlap = set(row["designed_chain"]) & set(row["fixed_chain"])
        if overlap:
            problems.append(
                f"line {int(index) + 2}: chain(s) {sorted(overlap)} appear in both "
                "designed_chain and fixed_chain"
            )

    lengths = (
        frame.assign(_len=frame["sequence"].str.len())
        .groupby(["pdb_id", frame["designed_chain"].map(",".join)])["_len"]
        .nunique()
    )
    inconsistent = lengths[lengths > 1]
    if not inconsistent.empty:
        problems.append(
            "sequence length varies within a (pdb_id, designed_chain) group for "
            f"{list(inconsistent.index)[:5]}. ProteinMPNN is fixed-backbone, so "
            "every design of a given chain must have the native chain's length."
        )

    if problems:
        raise SchemaError(f"{path} failed table-level validation:\n  " + "\n  ".join(problems))

    return frame


def load_test_set(path: Path, allow_extra_columns: bool = False) -> pd.DataFrame:
    """Load and validate ``test_set.csv``.

    Table-level invariants: ``pdb_id`` is unique, every complex names exactly
    two chains, and no MMseqs2 cluster straddles the train and test splits,
    which would break the leakage-free guarantee the split exists to provide.
    """
    raw = _read_raw(path, TEST_SET_SPEC, allow_extra_columns)
    frame = _parse_table(raw, TEST_SET_SPEC, path)

    problems: list[str] = []

    duplicated = frame["pdb_id"][frame["pdb_id"].duplicated(keep=False)]
    if not duplicated.empty:
        problems.append(f"duplicate pdb_id(s): {sorted(set(duplicated))[:5]}")

    wrong_arity = frame[frame["chains"].map(len) != 2]
    if not wrong_arity.empty:
        examples = [
            f"{row.pdb_id}={','.join(row.chains)}" for row in wrong_arity.head(5).itertuples()
        ]
        problems.append(
            f"{len(wrong_arity)} complex(es) do not name exactly two chains, for "
            f"example {examples}. The interface analysis is defined between a "
            "pair of chains; a complex with more has to be split into pairs "
            "explicitly, with the choice recorded."
        )

    straddling = frame.groupby("mmseqs_cluster_id")["split"].nunique()
    leaked = straddling[straddling > 1]
    if not leaked.empty:
        problems.append(
            f"MMseqs2 cluster(s) {list(leaked.index)[:5]} appear in both the train "
            "and test splits. That is sequence leakage and it invalidates the "
            "held-out claim."
        )

    if problems:
        raise SchemaError(f"{path} failed table-level validation:\n  " + "\n  ".join(problems))

    return frame


def check_cross_table(designs: pd.DataFrame, test_set: pd.DataFrame) -> None:
    """Check the two tables agree with each other.

    Every ``pdb_id`` in ``designs.csv`` must appear in ``test_set.csv``, and
    every chain named in ``designs.csv`` must be one of the chains that
    ``test_set.csv`` lists for that complex. Without this the analysis could
    silently run over a complex whose split membership is unknown.
    """
    problems: list[str] = []

    known = set(test_set["pdb_id"])
    unknown = sorted(set(designs["pdb_id"]) - known)
    if unknown:
        problems.append(
            f"pdb_id(s) {unknown[:10]} appear in designs.csv but not in "
            "test_set.csv, so their train/test membership is unknown"
        )

    chains_by_id = dict(zip(test_set["pdb_id"], test_set["chains"], strict=True))
    for row in designs.itertuples():
        expected = chains_by_id.get(row.pdb_id)
        if expected is None:
            continue
        named = set(row.designed_chain) | set(row.fixed_chain)
        stray = sorted(named - set(expected))
        if stray:
            problems.append(
                f"{row.pdb_id}: designs.csv names chain(s) {stray} that test_set.csv "
                f"does not list for this complex (it lists {list(expected)})"
            )

    if problems:
        unique = list(dict.fromkeys(problems))
        raise SchemaError(
            "designs.csv and test_set.csv are inconsistent:\n  " + "\n  ".join(unique[:20])
        )


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


def describe_contract(spec: TableSpec, indent: str = "") -> str:
    """Render a table specification as text, for error messages and the README."""
    lines = []
    for column in spec.columns:
        lines.append(f"{indent}{column.name} ({column.dtype_note})")
        lines.append(f"{indent}    {column.description}")
    return "\n".join(lines)


def all_specs() -> tuple[TableSpec, ...]:
    """Every table in the contract."""
    return (DESIGNS_SPEC, TEST_SET_SPEC)


def required_columns(spec: TableSpec) -> Iterable[str]:
    """Column names required by a specification."""
    return spec.column_names
