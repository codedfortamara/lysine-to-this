"""Scripts and the Modal app: clean failure without data, working dry run.

The requirement being tested is that every script runs end to end on the
fixture and fails cleanly with a useful message when the real data is absent.
A traceback is not a clean failure, and neither is a script that produces an
empty output file and exits zero.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )


# ---------------------------------------------------------------------------
# Every script is at least importable and offers help
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "00_fetch_natives.py",
        "01_define_interfaces.py",
        "02_partition_charge.py",
        "03_complementarity.py",
        "04_join_and_analyse.py",
        "05_figures.py",
    ],
)
def test_script_help_works(name: str) -> None:
    result = run_script(name, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


# ---------------------------------------------------------------------------
# Clean failure when the real data is absent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "00_fetch_natives.py",
        "01_define_interfaces.py",
        "02_partition_charge.py",
        "03_complementarity.py",
    ],
)
def test_script_fails_cleanly_without_data(name: str, tmp_path: Path) -> None:
    """No traceback, a non-zero exit, and a message naming what is missing."""
    result = run_script(
        name,
        "--test-set",
        str(tmp_path / "absent_test_set.csv"),
        *(
            ["--designs", str(tmp_path / "absent_designs.csv")]
            if name in ("02_partition_charge.py", "03_complementarity.py")
            else []
        ),
        "--results-dir",
        str(tmp_path / "results"),
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr, "a missing input must not produce a traceback"
    assert "ERROR" in result.stderr
    assert "not found" in result.stderr


def test_missing_data_error_quotes_the_contract(tmp_path: Path) -> None:
    """The message tells the reader what the file should have contained."""
    result = run_script(
        "00_fetch_natives.py",
        "--test-set",
        str(tmp_path / "absent.csv"),
        "--results-dir",
        str(tmp_path / "results"),
    )
    for column in ("pdb_id", "chains", "mmseqs_cluster_id", "split"):
        assert column in result.stderr
    assert "data/README.md" in result.stderr


# ---------------------------------------------------------------------------
# The skeletons say so
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["04_join_and_analyse.py", "05_figures.py"])
def test_skeleton_scripts_refuse_to_pretend(name: str) -> None:
    """A stub that silently produced an empty table would be worse than a stub."""
    result = run_script(name)
    assert result.returncode != 0
    assert "NotImplementedError" in result.stderr
    assert "skeleton" in result.stderr


# ---------------------------------------------------------------------------
# Scripts 01 to 03 run end to end on the fixture
# ---------------------------------------------------------------------------


def test_pipeline_runs_end_to_end_on_the_fixture(tmp_path, structure_case, extracts) -> None:
    """Scripts 01, 02 and 03 in sequence over the committed complex.

    This is the closest thing to a real run that can be done without the
    collaborator's data: the structure is real, the interface is real, and only
    the design table is constructed. It exercises the actual code path from
    structure through interface to partitioned charge and complementarity.
    """
    path, chain_a, chain_b = structure_case
    _, extract_b = extracts
    pdb_id = "1ABC"

    native_dir = tmp_path / "native"
    native_dir.mkdir()
    (native_dir / f"{pdb_id}.pdb").write_bytes(path.read_bytes())

    raw = tmp_path / "raw"
    raw.mkdir()
    test_set = raw / "test_set.csv"
    test_set.write_text(
        f'pdb_id,chains,mmseqs_cluster_id,split\n{pdb_id},"{chain_a},{chain_b}",cluster_0,test\n'
    )

    native = extract_b.sequence
    supercharged = "".join(
        "K" if i % 7 == 0 and aa not in "CP" else aa for i, aa in enumerate(native)
    )
    designs = raw / "designs.csv"
    designs.write_text(
        "pdb_id,designed_chain,fixed_chain,beta,replicate,sequence,net_charge_reported\n"
        f"{pdb_id},{chain_b},{chain_a},0.0,0,{native},0.0\n"
        f"{pdb_id},{chain_b},{chain_a},1.5,0,{supercharged},0.0\n"
    )

    results = tmp_path / "results"
    # The committed fixture is legacy PDB while the default native format is
    # mmCIF, so this also exercises the TOML override machinery.
    override = tmp_path / "config.toml"
    override.write_text('[structure]\nnative_format = "pdb"\n')
    base = ["--results-dir", str(results), "--config", str(override), "--quiet"]
    structural = [*base, "--native-dir", str(native_dir)]

    step1 = run_script("01_define_interfaces.py", "--test-set", str(test_set), *structural)
    assert step1.returncode == 0, step1.stderr
    assert (results / "interfaces.csv").is_file()
    assert (results / "interface_definitions.json").is_file()
    assert (results / "interfaces.csv.manifest.json").is_file()

    step2 = run_script(
        "02_partition_charge.py", "--designs", str(designs), "--test-set", str(test_set), *base
    )
    assert step2.returncode == 0, step2.stderr
    assert (results / "charge_partitions.csv").is_file()
    assert (results / "charge_reconciliation.csv").is_file()

    step3 = run_script(
        "03_complementarity.py", "--designs", str(designs), "--test-set", str(test_set), *structural
    )
    assert step3.returncode == 0, step3.stderr
    assert (results / "complementarity.csv").is_file()

    # Every output carries a manifest, and the manifests verify.
    from interface_charge.provenance import verify_manifest

    for output in ("interfaces.csv", "charge_partitions.csv", "complementarity.csv"):
        manifest_path = results / f"{output}.manifest.json"
        assert manifest_path.is_file(), f"no manifest beside {output}"
        report = verify_manifest(manifest_path, repo_root=REPO_ROOT)
        assert report["status"] == "completed"
        assert report["changed"] == [], report

    # The partition table must carry both charge definitions in separate columns.
    import pandas as pd

    table = pd.read_csv(results / "charge_partitions.csv")
    assert "charge_interface_simple" in table.columns
    assert any(c.startswith("charge_interface_ph") for c in table.columns)
    assert len(table) == 2

    # Making the chain more positive must raise its total simple charge.
    by_beta = table.set_index("beta")
    assert by_beta.loc[1.5, "charge_total_simple"] > by_beta.loc[0.0, "charge_total_simple"]


def test_length_mismatch_is_rejected_by_script_02(tmp_path, structure_case, extracts) -> None:
    """The fixed-backbone check, at the script level."""
    path, chain_a, chain_b = structure_case
    _, extract_b = extracts
    pdb_id = "1ABC"

    native_dir = tmp_path / "native"
    native_dir.mkdir()
    (native_dir / f"{pdb_id}.pdb").write_bytes(path.read_bytes())
    raw = tmp_path / "raw"
    raw.mkdir()
    results = tmp_path / "results"

    test_set = raw / "test_set.csv"
    test_set.write_text(
        f'pdb_id,chains,mmseqs_cluster_id,split\n{pdb_id},"{chain_a},{chain_b}",cluster_0,test\n'
    )
    designs = raw / "designs.csv"
    designs.write_text(
        "pdb_id,designed_chain,fixed_chain,beta,replicate,sequence,net_charge_reported\n"
        f"{pdb_id},{chain_b},{chain_a},0.0,0,{extract_b.sequence[:-3]},0.0\n"
    )

    override = tmp_path / "config.toml"
    override.write_text('[structure]\nnative_format = "pdb"\n')
    base = ["--results-dir", str(results), "--config", str(override), "--quiet"]

    step1 = run_script(
        "01_define_interfaces.py",
        "--test-set",
        str(test_set),
        *base,
        "--native-dir",
        str(native_dir),
    )
    assert step1.returncode == 0, step1.stderr

    result = run_script(
        "02_partition_charge.py", "--designs", str(designs), "--test-set", str(test_set), *base
    )
    assert result.returncode != 0
    assert "fixed-backbone" in result.stderr
    assert not (results / "charge_partitions.csv").is_file()


# ---------------------------------------------------------------------------
# The Modal app
# ---------------------------------------------------------------------------


def test_modal_dry_run_works_without_modal_installed(tmp_path, structure_case, extracts) -> None:
    """The dry run is pure Python and must never need a GPU or a Modal login."""
    _, chain_a, chain_b = structure_case
    _, extract_b = extracts
    pdb_id = "1ABC"

    raw = tmp_path / "raw"
    raw.mkdir()
    test_set = raw / "test_set.csv"
    test_set.write_text(
        f'pdb_id,chains,mmseqs_cluster_id,split\n{pdb_id},"{chain_a},{chain_b}",cluster_0,test\n'
    )
    designs = raw / "designs.csv"
    rows = "\n".join(
        f"{pdb_id},{chain_b},{chain_a},{beta},{rep},{extract_b.sequence},0.0"
        for beta in ("0.0", "1.5", "-1.5")
        for rep in (0, 1)
    )
    designs.write_text(
        "pdb_id,designed_chain,fixed_chain,beta,replicate,sequence,net_charge_reported\n"
        + rows
        + "\n"
    )

    definitions = tmp_path / "definitions.json"
    definitions.write_text(
        json.dumps(
            {
                pdb_id: {
                    "chain_a": chain_a,
                    "chain_b": chain_b,
                    "source": "native",
                    "chains": {
                        chain_a: {
                            "sequence": "MKV" * 10,
                            "interface_positions": [],
                            "core_positions": [],
                        },
                        chain_b: {
                            "sequence": extract_b.sequence,
                            "interface_positions": [],
                            "core_positions": [],
                        },
                    },
                }
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "modal_app" / "af2_multimer.py"),
            "--dry-run",
            "--designs",
            str(designs),
            "--test-set",
            str(test_set),
            "--definitions",
            str(definitions),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "jobs enumerated:      6" in result.stdout
    assert "estimated_gpu_hours" in result.stdout
    assert "estimated_usd" in result.stdout
    # The estimate must not be presented as a measurement.
    assert "planning assumption" in result.stdout


def test_job_key_is_stable_and_filename_safe() -> None:
    sys.path.insert(0, str(REPO_ROOT / "modal_app"))
    from af2_multimer import Job

    job = Job(
        pdb_id="1BRS",
        beta=-1.5,
        replicate=2,
        designed_chain="A",
        chains={"A": "MKV", "D": "MKV"},
        msa_mode={"A": "single_sequence", "D": "msa"},
    )
    key = job.key
    assert "/" not in key and " " not in key and "." not in key
    assert "1BRS" in key and "rep2" in key
    # Positive and negative betas must not collide.
    positive = Job(
        pdb_id="1BRS",
        beta=1.5,
        replicate=2,
        designed_chain="A",
        chains=job.chains,
        msa_mode=job.msa_mode,
    )
    assert positive.key != key


def test_cost_estimate_scales_linearly() -> None:
    sys.path.insert(0, str(REPO_ROOT / "modal_app"))
    from af2_multimer import Job, estimate_cost

    def make(n: int) -> list[Job]:
        return [
            Job(
                pdb_id="1BRS",
                beta=0.0,
                replicate=i,
                designed_chain="A",
                chains={"A": "MKV", "D": "MKV"},
                msa_mode={"A": "single_sequence", "D": "msa"},
            )
            for i in range(n)
        ]

    one = estimate_cost(make(10))
    two = estimate_cost(make(20))
    assert two["estimated_gpu_hours"] == pytest.approx(2 * one["estimated_gpu_hours"])
    assert two["estimated_usd"] == pytest.approx(2 * one["estimated_usd"])


def test_timeout_is_a_single_named_constant() -> None:
    sys.path.insert(0, str(REPO_ROOT / "modal_app"))
    import af2_multimer
    from interface_charge.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.af2.timeout_s == af2_multimer.TIMEOUT_S
    assert af2_multimer.TIMEOUT_S >= 3600


def test_redesigned_chain_uses_single_sequence_mode(tmp_path, structure_case, extracts) -> None:
    """A design has no evolutionary history, so an MSA for it is misleading."""
    sys.path.insert(0, str(REPO_ROOT / "modal_app"))
    from af2_multimer import build_job_list

    _, chain_a, chain_b = structure_case
    _, extract_b = extracts
    pdb_id = "1ABC"
    raw = tmp_path / "raw"
    raw.mkdir()

    test_set = raw / "test_set.csv"
    test_set.write_text(
        f'pdb_id,chains,mmseqs_cluster_id,split\n{pdb_id},"{chain_a},{chain_b}",c0,test\n'
    )
    designs = raw / "designs.csv"
    designs.write_text(
        "pdb_id,designed_chain,fixed_chain,beta,replicate,sequence,net_charge_reported\n"
        f"{pdb_id},{chain_b},{chain_a},0.0,0,{extract_b.sequence},0.0\n"
    )
    definitions = tmp_path / "definitions.json"
    definitions.write_text(
        json.dumps(
            {
                pdb_id: {
                    "chain_a": chain_a,
                    "chain_b": chain_b,
                    "chains": {
                        chain_a: {"sequence": "MKV" * 10},
                        chain_b: {"sequence": extract_b.sequence},
                    },
                }
            }
        )
    )

    jobs = build_job_list(designs, definitions, test_set)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.msa_mode[chain_b] == "single_sequence"
    assert job.msa_mode[chain_a] == "msa"
