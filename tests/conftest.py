"""Shared test fixtures.

The structural fixtures are session scoped because a SASA calculation over a
six thousand atom complex takes about a second and the interface definition
runs it three times. Computing it once per session keeps the suite fast enough
to run on every save.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from interface_charge.config import DEFAULT_CONFIG, Config
from interface_charge.interface import define_interface
from interface_charge.structures import extract_chain, load_model

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _available_structures() -> list[tuple[str, str, str]]:
    """Every committed structural fixture, as (stem, chain_a, chain_b).

    1BRS is the intended primary fixture, being the canonical highly charged
    protein-protein interface. It could not be downloaded in the environment
    this repository was scaffolded in, because files.rcsb.org was blocked by
    the network egress policy. Once it is present the suite picks it up here
    with no other change. Build it with::

        python tests/fixtures/build_fixture.py --pdb-id 1BRS --chains A D
    """
    known = {
        "7ddo_fixture": ("A", "C"),  # human ACE2 with SARS-CoV-2 RBD
        "1brs_fixture": ("A", "D"),  # barnase with barstar, if available
    }
    out = []
    for stem, (chain_a, chain_b) in known.items():
        if (FIXTURE_DIR / f"{stem}.pdb").is_file():
            out.append((stem, chain_a, chain_b))
    return out


STRUCTURES = _available_structures()


@pytest.fixture(scope="session")
def config() -> Config:
    return DEFAULT_CONFIG


@pytest.fixture(scope="session", params=[s[0] for s in STRUCTURES], ids=[s[0] for s in STRUCTURES])
def structure_case(request) -> tuple[Path, str, str]:
    """A committed two-chain complex, as (path, chain_a, chain_b)."""
    stem = request.param
    chain_a, chain_b = next((a, b) for s, a, b in STRUCTURES if s == stem)
    return FIXTURE_DIR / f"{stem}.pdb", chain_a, chain_b


@pytest.fixture(scope="session")
def model(structure_case, config):
    path, _, _ = structure_case
    return load_model(path, config.structure)


@pytest.fixture(scope="session")
def extracts(model, structure_case, config):
    """The two chains of the fixture complex, as (extract_a, extract_b)."""
    _, chain_a, chain_b = structure_case
    return (
        extract_chain(model, chain_a, config.structure),
        extract_chain(model, chain_b, config.structure),
    )


@pytest.fixture(scope="session")
def interface_definition(model, structure_case, config):
    """The full interface definition of the fixture complex. Computed once."""
    path, chain_a, chain_b = structure_case
    return define_interface(
        model,
        pdb_id=path.stem.upper()[:4],
        chain_a=chain_a,
        chain_b=chain_b,
        structure_sha256="test-fixture",
        interface_params=config.interface,
        structure_params=config.structure,
        source="native",
    )


@pytest.fixture
def valid_tables(tmp_path, extracts, structure_case) -> tuple[Path, Path]:
    """A minimal but genuinely consistent designs.csv and test_set.csv pair.

    Built from the real fixture chain so that sequence lengths agree with the
    native structure, which is the invariant the loaders and script 02 check.
    The "designs" are the native sequence with a small number of deliberate
    substitutions, which is enough to exercise the loaders without pretending
    to be design output.
    """
    _, chain_a, chain_b = structure_case
    _, extract_b = extracts
    pdb_id = "1ABC"
    native = extract_b.sequence

    def mutate(sequence: str, count: int, residue: str) -> str:
        chars = list(sequence)
        for position in range(0, min(count * 3, len(chars)), 3):
            chars[position] = residue
        return "".join(chars)

    rows = [
        (pdb_id, chain_b, chain_a, "0.0", "0", native, "0.0"),
        (pdb_id, chain_b, chain_a, "1.5", "0", mutate(native, 5, "K"), "0.0"),
        (pdb_id, chain_b, chain_a, "-1.5", "0", mutate(native, 5, "E"), "0.0"),
    ]

    designs = tmp_path / "designs.csv"
    designs.write_text(
        "pdb_id,designed_chain,fixed_chain,beta,replicate,sequence,net_charge_reported\n"
        + "\n".join(",".join(row) for row in rows)
        + "\n"
    )

    test_set = tmp_path / "test_set.csv"
    test_set.write_text(
        f'pdb_id,chains,mmseqs_cluster_id,split\n{pdb_id},"{chain_a},{chain_b}",cluster_0,test\n'
    )
    return designs, test_set
