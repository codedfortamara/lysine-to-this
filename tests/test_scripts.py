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
# No skeletons remain
# ---------------------------------------------------------------------------


def test_no_script_is_still_a_stub() -> None:
    """Both former skeletons, 04 and 05, are implemented.

    Kept as a test rather than deleted because a stub reintroduced later would
    otherwise pass silently, and a script that raises NotImplementedError is not
    something the pipeline should ever contain again.
    """
    import pathlib

    for path in sorted(pathlib.Path("scripts").glob("*.py")):
        source = path.read_text()
        assert "NotImplementedError" not in source, (
            f"{path.name} raises NotImplementedError; every script is expected to "
            "either do its work or fail cleanly on missing input"
        )


def test_figures_script_fails_cleanly_with_no_results(tmp_path: Path) -> None:
    """05 draws what the analysis produced. With nothing to draw it must say so."""
    result = run_script("05_figures.py", "--results-dir", str(tmp_path))
    assert result.returncode != 0
    assert "no results tables found" in result.stderr
    assert "invents nothing" in result.stderr


def test_analysis_script_fails_cleanly_without_its_input(tmp_path: Path) -> None:
    """04 is implemented, so it must fail like a real script, not like a stub."""
    result = run_script(
        "04_join_and_analyse.py",
        "--charge-partitions",
        str(tmp_path / "absent.csv"),
        "--results-dir",
        str(tmp_path / "results"),
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "02_partition_charge" in result.stderr


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


def test_cost_estimate_accounts_for_cold_starts_and_failures() -> None:
    """Compute scales with jobs; cold starts scale with containers, then stop.

    The distinction matters for real money. A ten-job pilot is dominated by
    cold starts, so costing it as jobs times minutes understates it badly. A
    four-hundred-job grid is dominated by compute, and once the container cap
    is reached the cold-start term stops growing entirely.
    """
    sys.path.insert(0, str(REPO_ROOT / "modal_app"))
    from af2_multimer import PARAMS, estimate_cost

    small = estimate_cost(10)
    large = estimate_cost(20)

    # Compute hours are exactly linear in the job count.
    assert large["compute_gpu_hours"] == pytest.approx(2 * small["compute_gpu_hours"])

    # The total exceeds bare compute, because cold starts and retries are real.
    assert small["estimated_gpu_hours"] > small["compute_gpu_hours"]
    assert small["cold_start_gpu_hours"] > 0

    # Beyond the container cap the cold-start term is constant, so cost becomes
    # linear in jobs and the pilot's per-job overhead disappears.
    capped_a = estimate_cost(PARAMS.max_containers * 10)
    capped_b = estimate_cost(PARAMS.max_containers * 20)
    assert capped_a["cold_start_gpu_hours"] == pytest.approx(capped_b["cold_start_gpu_hours"])

    # A pilot carries proportionally far more overhead than the full grid.
    pilot_overhead = small["cold_start_gpu_hours"] / small["compute_gpu_hours"]
    grid_overhead = capped_b["cold_start_gpu_hours"] / capped_b["compute_gpu_hours"]
    assert pilot_overhead > 10 * grid_overhead


def test_cost_estimate_accepts_a_bare_job_count() -> None:
    """A grid must be costable before designs.csv exists, which is most of the time."""
    sys.path.insert(0, str(REPO_ROOT / "modal_app"))
    from af2_multimer import estimate_cost

    estimate = estimate_cost(26 * 5 * 3)
    assert estimate["n_jobs"] == 390
    assert estimate["estimated_usd"] > 0
    assert "UNVERIFIED" in estimate["estimate_basis"]
    # No job list, so no residue statistics may be invented.
    assert "median_total_residues" not in estimate


def test_cheaper_gpu_costs_less_for_the_same_grid() -> None:
    sys.path.insert(0, str(REPO_ROOT / "modal_app"))
    from af2_multimer import estimate_cost

    a100 = estimate_cost(390, gpu_type="A100-40GB")
    l4 = estimate_cost(390, gpu_type="L4")
    assert l4["estimated_usd"] < a100["estimated_usd"]
    # Same GPU-hours: the model holds minutes-per-job constant across cards,
    # which is exactly why the CLI prints a caveat saying so.
    assert l4["estimated_gpu_hours"] == pytest.approx(a100["estimated_gpu_hours"])


def test_unknown_gpu_raises_rather_than_guessing_a_rate() -> None:
    sys.path.insert(0, str(REPO_ROOT / "modal_app"))
    from af2_multimer import estimate_cost

    with pytest.raises(ValueError, match="no published rate"):
        estimate_cost(100, gpu_type="RTX4090")


def test_hypothetical_dry_run_needs_no_input_tables() -> None:
    """The planning path must work with no designs.csv anywhere on disk."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "modal_app" / "af2_multimer.py"),
            "--dry-run",
            "--assume-complexes",
            "26",
            "--assume-betas",
            "5",
            "--assume-replicates",
            "3",
            "--compare-gpus",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "jobs:                 390" in result.stdout
    assert "no input tables were read" in result.stdout
    assert "A100-40GB" in result.stdout and "L4" in result.stdout
    assert "CAVEATS" in result.stdout


def test_partial_hypothetical_grid_is_rejected() -> None:
    """Two of the three counts is an ambiguous request, not a default."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "modal_app" / "af2_multimer.py"),
            "--dry-run",
            "--assume-complexes",
            "26",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "must be given together" in result.stderr


def test_timeout_is_a_single_named_constant() -> None:
    sys.path.insert(0, str(REPO_ROOT / "modal_app"))
    import af2_multimer
    from interface_charge.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.af2.timeout_s == af2_multimer.TIMEOUT_S
    # Generous enough for the largest complex still in scope, tight enough that a
    # stuck job is killed rather than billed for hours. It was 7200, and every
    # job that went wrong ran the full two hours before anything noticed.
    assert 600 <= af2_multimer.TIMEOUT_S <= 3600


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


# ---------------------------------------------------------------------------
# Design regeneration (the pure parts; sampling needs torch and ProteinMPNN)
# ---------------------------------------------------------------------------


def _regen_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("regen", SCRIPTS / "10_regenerate_designs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_derived_seed_is_stable_and_distinct() -> None:
    """Same inputs give the same seed; any change gives a different one."""
    regen = _regen_module()
    base = regen.derive_seed(0, "1BRS", 1.5, 0)
    assert base == regen.derive_seed(0, "1BRS", 1.5, 0)
    assert base != regen.derive_seed(1, "1BRS", 1.5, 0)
    assert base != regen.derive_seed(0, "1BRT", 1.5, 0)
    assert base != regen.derive_seed(0, "1BRS", -1.5, 0)
    assert base != regen.derive_seed(0, "1BRS", 1.5, 1)


def test_derived_seed_does_not_shift_when_the_grid_grows() -> None:
    """Hashing rather than counting: adding a beta must not move other seeds.

    With a running counter, inserting a value into the grid would renumber
    every design after it and silently invalidate an existing run.
    """
    regen = _regen_module()
    before = {b: regen.derive_seed(0, "1BRS", b, 0) for b in (-1.5, 0.0, 1.5)}
    after = {b: regen.derive_seed(0, "1BRS", b, 0) for b in (-3.0, -1.5, 0.0, 1.5, 3.0)}
    for beta, seed in before.items():
        assert after[beta] == seed


def test_seed_is_in_range_for_torch() -> None:
    regen = _regen_module()
    for replicate in range(20):
        assert 0 <= regen.derive_seed(0, "1BRS", 1.5, replicate) < 2**32


def test_alphabet_matches_proteinmpnn_exactly() -> None:
    """Indexing the bias against the wrong alphabet biases the wrong residues.

    The source paper flags this as a silent failure that moves no charge, so
    the ordering is pinned here rather than trusted.
    """
    regen = _regen_module()
    assert regen.MPNN_ALPHABET == "ACDEFGHIKLMNPQRSTVWYX"
    assert regen.MPNN_ALPHABET.index("D") == 2
    assert regen.MPNN_ALPHABET.index("E") == 3
    assert regen.MPNN_ALPHABET.index("K") == 8
    assert regen.MPNN_ALPHABET.index("R") == 14


def test_unknown_residue_code_raises_rather_than_becoming_alanine() -> None:
    """The upstream export rewrites X to A. That would shift a net charge."""
    regen = _regen_module()
    x_index = regen.MPNN_ALPHABET.index("X")
    with pytest.raises(ValueError, match="alanine"):
        regen.sequence_from_indices([x_index], [0])


def test_known_residue_codes_decode() -> None:
    regen = _regen_module()
    indices = [regen.MPNN_ALPHABET.index(aa) for aa in "MKVDE"]
    assert regen.sequence_from_indices(indices, list(range(5))) == "MKVDE"


# ---------------------------------------------------------------------------
# Wave planning, canary ordering and the budget guard
# ---------------------------------------------------------------------------


def _af2():
    sys.path.insert(0, str(REPO_ROOT / "modal_app"))
    import af2_multimer

    return af2_multimer


def _function_source(af2, name: str) -> str:
    """Source of a function by name, including ones nested inside the Modal block.

    Read rather than imported because the Modal-only functions need colabfold
    and a GPU to import, and these checks are about what the source does.
    """
    import ast

    text = (REPO_ROOT / "modal_app" / "af2_multimer.py").read_text()
    lines = text.splitlines()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{name} not found in af2_multimer.py")


def _job(af2, pdb_id="1BRS", beta=0.0, replicate=0, primary_len=100, partner_len=100):
    return af2.Job(
        pdb_id=pdb_id,
        beta=beta,
        replicate=replicate,
        designed_chain="A",
        chains={"A": "M" * primary_len, "D": "M" * partner_len},
        msa_mode={"A": "single_sequence", "D": "msa"},
    )


def test_canary_puts_the_largest_complexes_first() -> None:
    """Out-of-memory and timeout track total length, so the biggest go first."""
    af2 = _af2()
    jobs = [_job(af2, pdb_id=f"1BR{i}", primary_len=50 * i) for i in range(1, 7)]
    ordered = af2.order_canary_first(jobs, n_canary=3)
    sizes = [j.total_residues() for j in ordered[:3]]
    assert sizes == sorted(sizes, reverse=True)
    assert min(sizes) > max(j.total_residues() for j in ordered[3:])


def test_canary_preserves_every_job() -> None:
    af2 = _af2()
    jobs = [_job(af2, pdb_id=f"1BR{i}", primary_len=50 * i) for i in range(1, 7)]
    ordered = af2.order_canary_first(jobs, n_canary=2)
    assert sorted(j.key for j in ordered) == sorted(j.key for j in jobs)


def test_canary_ordering_is_deterministic() -> None:
    """A resumed run must process outstanding work in the same order."""
    af2 = _af2()
    jobs = [_job(af2, pdb_id=f"1BR{i}", primary_len=100) for i in range(1, 8)]
    first = [j.key for j in af2.order_canary_first(jobs, 3)]
    second = [j.key for j in af2.order_canary_first(list(reversed(jobs)), 3)]
    assert first == second


def test_zero_canary_is_allowed() -> None:
    af2 = _af2()
    jobs = [_job(af2, pdb_id=f"1BR{i}") for i in range(1, 4)]
    assert len(af2.order_canary_first(jobs, 0)) == 3


def test_waves_are_cut_by_beta_so_each_stop_is_a_complete_grid() -> None:
    """Stopping after wave one must leave every complex covered at those betas."""
    af2 = _af2()
    jobs = [
        _job(af2, pdb_id=f"1BR{c}", beta=b)
        for c in range(1, 5)
        for b in (0.0, -1.5, 1.5, -3.0, 3.0)
    ]
    waves = af2.split_into_waves(jobs, af2.WAVE_BETAS)
    assert len(waves) == 2
    assert {j.beta for j in waves[0]} == {0.0, -1.5, 1.5}
    assert {j.beta for j in waves[1]} == {-3.0, 3.0}
    # Every complex appears in wave one: that is what makes it analysable alone.
    assert {j.pdb_id for j in waves[0]} == {f"1BR{c}" for c in range(1, 5)}


def test_no_job_is_dropped_by_a_mis_specified_wave_plan() -> None:
    af2 = _af2()
    jobs = [_job(af2, beta=b, replicate=i) for i, b in enumerate((0.0, 7.7, -3.0))]
    waves = af2.split_into_waves(jobs, [[0.0]])
    assert sum(len(w) for w in waves) == 3
    assert any(j.beta == 7.7 for w in waves for j in w)


def test_budget_guard_permits_work_within_the_ceiling() -> None:
    af2 = _af2()
    guard = af2.BudgetGuard(ceiling_usd=100.0, usd_per_gpu_hour=2.10)
    assert guard.may_start(10, minutes_per_job=8.0)
    assert guard.spent_usd == 0.0


def test_budget_guard_refuses_to_start_beyond_the_ceiling() -> None:
    af2 = _af2()
    guard = af2.BudgetGuard(ceiling_usd=1.0, usd_per_gpu_hour=2.10)
    assert not guard.may_start(100, minutes_per_job=8.0)


def test_budget_guard_uses_measured_time_once_it_has_any() -> None:
    """The planning assumption is only used until real numbers arrive."""
    af2 = _af2()
    guard = af2.BudgetGuard(ceiling_usd=100.0, usd_per_gpu_hour=2.10)
    assert guard.observed_minutes_per_job() is None
    for _ in range(4):
        guard.record(120.0)  # two minutes each, far under the 8 minute assumption
    assert guard.observed_minutes_per_job() == pytest.approx(2.0)
    # At two minutes a job, a batch that the assumption would have blocked fits.
    assert guard.may_start(200, minutes_per_job=8.0)


def test_budget_guard_accumulates_spend() -> None:
    af2 = _af2()
    guard = af2.BudgetGuard(ceiling_usd=100.0, usd_per_gpu_hour=3600.0, billing_overhead=1.0)
    guard.record(1.0)
    assert guard.spent_usd == pytest.approx(1.0)
    guard.record(2.0)
    assert guard.spent_usd == pytest.approx(3.0)
    assert guard.remaining_usd == pytest.approx(97.0)


def test_budget_guard_reports_billed_spend_not_job_time() -> None:
    """The defect this exists for: a guard that under-reports by half.

    It summed each job's own wall clock. Modal bills container time, which also
    covers image pulls, loading the model parameters, and containers held idle
    waiting for the slowest job in a chunk. On the first real batch that read
    $12.41 against roughly $26.00 billed, so a ceiling set at $54 would not have
    stopped anything until well past $100. A safeguard being wrong in this
    direction is the only direction that matters.
    """
    af2 = _af2()
    honest = af2.BudgetGuard(ceiling_usd=100.0, usd_per_gpu_hour=3600.0, billing_overhead=2.1)
    naive = af2.BudgetGuard(ceiling_usd=100.0, usd_per_gpu_hour=3600.0, billing_overhead=1.0)
    for guard in (honest, naive):
        guard.record(10.0)
    assert honest.spent_usd == pytest.approx(2.1 * naive.spent_usd)
    assert honest.remaining_usd < naive.remaining_usd


def test_budget_guard_projects_batches_at_the_billed_rate_too() -> None:
    """Reporting honestly but gating naively would leave the hole open."""
    af2 = _af2()
    honest = af2.BudgetGuard(ceiling_usd=10.0, usd_per_gpu_hour=3600.0, billing_overhead=2.1)
    naive = af2.BudgetGuard(ceiling_usd=10.0, usd_per_gpu_hour=3600.0, billing_overhead=1.0)
    # A batch costing 6 units of job time is 12.6 billed, over the ceiling.
    assert naive.may_start(6, minutes_per_job=1.0 / 60.0)
    assert not honest.may_start(6, minutes_per_job=1.0 / 60.0)


def test_a_billing_overhead_below_one_is_refused() -> None:
    """It would claim Modal bills less than the jobs themselves consume."""
    af2 = _af2()
    with pytest.raises(ValueError, match=r"at least 1\.0"):
        af2.BudgetGuard(ceiling_usd=10.0, usd_per_gpu_hour=2.10, billing_overhead=0.5)


def test_budget_guard_cannot_report_negative_remaining() -> None:
    af2 = _af2()
    guard = af2.BudgetGuard(ceiling_usd=1.0, usd_per_gpu_hour=3600.0)
    guard.record(10.0)
    assert guard.remaining_usd == 0.0


def test_budget_guard_has_no_way_to_cancel_running_work() -> None:
    """The guard gates dispatch only, by design.

    Killing in-flight work wastes what has already been paid for in that batch
    and leaves a partial grid, which is worse than a bounded overshoot.
    """
    af2 = _af2()
    guard = af2.BudgetGuard(ceiling_usd=1.0, usd_per_gpu_hour=2.10)
    for forbidden in ("cancel", "kill", "abort", "terminate", "stop"):
        assert not hasattr(guard, forbidden)


def test_invalid_ceiling_raises() -> None:
    af2 = _af2()
    for bad in (0.0, -5.0):
        with pytest.raises(ValueError, match="ceiling_usd"):
            af2.BudgetGuard(ceiling_usd=bad, usd_per_gpu_hour=2.10)


def test_chunking_covers_everything_exactly_once() -> None:
    af2 = _af2()
    items = list(range(47))
    chunks = af2.chunked(items, 20)
    assert [len(c) for c in chunks] == [20, 20, 7]
    assert [x for c in chunks for x in c] == items


def test_chunk_size_must_be_positive() -> None:
    af2 = _af2()
    with pytest.raises(ValueError, match="chunk size"):
        af2.chunked([1, 2, 3], 0)


def test_wave_report_counts_failures_without_raising() -> None:
    """A partly failed wave is still informative; the decision is a person's."""
    af2 = _af2()
    guard = af2.BudgetGuard(100.0, 2.10)
    results = [
        {"interface_ptm": 0.80, "wall_clock_s": 300.0},
        {"interface_ptm": 0.60, "wall_clock_s": 360.0},
        RuntimeError("CUDA out of memory"),
    ]
    report = af2.summarise_wave(1, results, guard)
    assert report.n_jobs == 3
    assert report.n_succeeded == 2
    assert report.n_failed == 1
    assert report.success_rate == pytest.approx(2 / 3)
    assert report.median_interface_ptm == pytest.approx(0.70)
    assert report.median_wall_clock_minutes == pytest.approx(5.5)
    assert "CUDA out of memory" in report.failures[0]


def test_wave_report_survives_missing_metrics() -> None:
    af2 = _af2()
    report = af2.summarise_wave(1, [{"key": "x"}], af2.BudgetGuard(100.0, 2.10))
    assert report.median_interface_ptm is None
    assert "unavailable" in report.render()


def test_wave_report_of_all_failures_reports_zero_not_an_error() -> None:
    af2 = _af2()
    report = af2.summarise_wave(1, [RuntimeError("boom")] * 4, af2.BudgetGuard(100.0, 2.10))
    assert report.success_rate == 0.0
    assert report.n_succeeded == 0


def test_wave_one_covers_the_band_where_the_science_is() -> None:
    """Beta +/-3 is deferred: upstream shows only 10-17 percent fold there at all."""
    af2 = _af2()
    assert af2.WAVE_BETAS[0] == [0.0, -1.5, 1.5]
    assert af2.WAVE_BETAS[1] == [-3.0, 3.0]


# ---------------------------------------------------------------------------
# Findings from the adversarial review of e1937c5
# ---------------------------------------------------------------------------


def test_failed_jobs_are_charged_to_the_budget() -> None:
    """The most expensive failure mode was invisible to the guard.

    A job that hits the timeout or dies out of memory consumed GPU time and
    produced nothing, and its exception carries no duration. Charging only
    successes meant three attempts at the largest complex burned six GPU-hours
    and moved the counter by zero, after which the ceiling happily let more
    batches start.
    """
    af2 = _af2()
    guard = af2.BudgetGuard(ceiling_usd=100.0, usd_per_gpu_hour=3600.0, billing_overhead=1.0)
    guard.record(60.0)
    after_success = guard.spent_usd
    guard.record_failure()
    assert guard.spent_usd > after_success, "a failure must cost something"


def test_a_failure_does_not_corrupt_the_observed_per_job_time() -> None:
    """Failures are charged pessimistically, so folding them into the mean would
    make every projection pessimistic too."""
    af2 = _af2()
    guard = af2.BudgetGuard(ceiling_usd=100.0, usd_per_gpu_hour=2.10)
    for _ in range(4):
        guard.record(120.0)
    before = guard.observed_minutes_per_job()
    guard.record_failure()
    assert guard.observed_minutes_per_job() == pytest.approx(before)


def test_failures_are_charged_at_the_timeout_before_anything_is_measured() -> None:
    """With no observation to go on, the pessimistic assumption is the safe one."""
    af2 = _af2()
    guard = af2.BudgetGuard(ceiling_usd=1e9, usd_per_gpu_hour=3600.0, billing_overhead=1.0)
    guard.record_failure(attempts=1)
    assert guard.spent_usd == pytest.approx(float(af2.PARAMS.timeout_s))


def test_the_driver_charges_failures() -> None:
    """The guard can only stop a systematically failing run if it is told."""
    af2 = _af2()
    source = _function_source(af2, "drive")
    assert "guard.record_failure()" in source


def test_only_one_driver_may_dispatch_at_a_time() -> None:
    """Two drivers compute the same jobs and you pay twice.

    metrics.json is not a mutex: both containers check it before either writes.
    This happened on 28 July with three concurrent drivers.
    """
    af2 = _af2()
    source = _function_source(af2, "drive")
    assert "run_lease.json" in source
    assert "LEASE_STALE_AFTER_S" in source
    assert '"refused": True' in source
    assert source.index("lease_path.is_file()") < source.index("predict.map")


def test_the_lease_outlasts_a_batch_by_a_wide_margin() -> None:
    """A lease shorter than a batch expires under a driver that is working.

    The heartbeat is written between batches, so a batch running to the timeout
    must not be long enough to let the lease go stale. Otherwise a second driver
    concludes the first is dead, takes the lease, and both dispatch, which is
    the exact failure the lease exists to prevent.
    """
    af2 = _af2()
    assert af2.LEASE_STALE_AFTER_S >= 2 * af2.TIMEOUT_S


def test_the_lease_goes_stale_so_a_crash_does_not_block_the_grid() -> None:
    af2 = _af2()
    assert af2.LEASE_STALE_AFTER_S <= 6 * 3600, "a crashed driver must not block for a day"
    source = _function_source(af2, "drive")
    assert "take_lease()" in source
    assert "lease_path.unlink()" in source, "a finished driver must release its lease"


def test_only_the_holder_may_release_the_lease() -> None:
    """Deleting another driver's lease would silently permit a second dispatcher."""
    af2 = _af2()
    source = _function_source(af2, "drive")
    release = source[
        source.index("lease_path.unlink()") - 400 : source.index("lease_path.unlink()")
    ]
    assert '"run_id") == run_id' in release


def test_an_unreadable_ledger_refuses_rather_than_resetting_the_budget() -> None:
    """Reading a corrupt ledger as zero hands back the entire budget."""
    af2 = _af2()
    source = _function_source(af2, "drive")
    assert "could not be read" in source
    assert "hand back" in source


def test_the_fold_marker_requires_a_structure_not_just_scores() -> None:
    """Scores without a PDB produce a complete-looking metrics.json with no
    structure behind it, and a note blaming a missing native."""
    af2 = _af2()
    source = _function_source(af2, "predict")
    assert "structures_written" in source
    assert "expected_residues" in source, "the scores must describe this job's length"


def test_a_failure_is_charged_at_the_timeout_not_at_a_fast_success() -> None:
    """The eighty-fold undercharge.

    Failures are overwhelmingly jobs that ran until something killed them;
    successes are jobs that finished quickly. Charging failures at the success
    mean made a one minute success turn a two-attempt timeout into 22 cents
    instead of 17.64 dollars, inside the guard whose whole purpose is stopping
    runaway spend.
    """
    af2 = _af2()
    guard = af2.BudgetGuard(ceiling_usd=1e9, usd_per_gpu_hour=3600.0, billing_overhead=1.0)
    guard.record(60.0)  # one fast success
    before = guard.spent_usd
    guard.record_failure(attempts=2)
    charged = guard.spent_usd - before
    assert charged == pytest.approx(2 * af2.PARAMS.timeout_s), (
        "a failure must be charged at the full timeout per attempt"
    )
    assert charged >= 20 * 60.0, "and must not be anchored to the fast success"


def test_the_ceiling_is_a_total_not_a_per_launch_allowance() -> None:
    """Relaunching four times under a $60 ceiling authorised $240 by accident."""
    af2 = _af2()
    source = _function_source(af2, "drive")
    assert "spend_ledger.json" in source
    assert "prior_usd" in source
    assert "budget_usd - prior_usd" in source


def test_folding_is_not_repeated_when_only_the_metrics_fail() -> None:
    """Cheap post-processing shared a retry boundary with expensive inference.

    metrics.json was the only completion marker, so any defect in
    collect_metrics made Modal refold the complex from scratch.
    """
    af2 = _af2()
    source = _function_source(af2, "predict")
    assert "folded.json" in source
    assert "already_folded" in source
    assert source.index("already_folded = folded_path.is_file()") < source.index("colabfold_run(")
    assert "Refusing to mark this job as folded" in source, (
        "the marker must only be written once the outputs are known to exist"
    )


# ---------------------------------------------------------------------------
# The circuit breaker
# ---------------------------------------------------------------------------


def test_a_run_that_only_fails_stops_itself() -> None:
    """29 July: 20 failed, 0 completed, 141 minutes, ten GPUs, nothing watching."""
    af2 = _af2()
    verdict = af2.failure_verdict(n_completed=0, n_failed=20, consecutive_failures=20)
    assert verdict
    assert "consecutive failures" in verdict


def test_scattered_failures_do_not_stop_a_working_run() -> None:
    """Preemption and the odd bad job are normal; a healthy run must continue."""
    af2 = _af2()
    assert not af2.failure_verdict(n_completed=40, n_failed=2, consecutive_failures=1)


def test_alternating_success_and_failure_is_caught_by_the_rate() -> None:
    """The consecutive counter never trips here, and half the money is wasted."""
    af2 = _af2()
    verdict = af2.failure_verdict(n_completed=5, n_failed=9, consecutive_failures=1)
    assert verdict
    assert "rate of" in verdict


def test_one_early_failure_does_not_stop_the_run() -> None:
    """A rate of 100% over one job is not evidence of anything."""
    af2 = _af2()
    assert not af2.failure_verdict(n_completed=0, n_failed=1, consecutive_failures=1)


def test_consecutive_failures_are_counted_from_the_end_of_the_batch() -> None:
    """Where a run that has just broken shows it."""
    af2 = _af2()
    ok = {"wall_clock_s": 120.0}
    assert af2.consecutive_failures([ok, ok, ok]) == 0
    assert af2.consecutive_failures([ok, RuntimeError(), RuntimeError()]) == 2
    assert af2.consecutive_failures([RuntimeError(), ok, RuntimeError()]) == 1
    assert af2.consecutive_failures([RuntimeError()] * 5) == 5


def test_a_result_without_a_wall_clock_counts_as_a_failure() -> None:
    """A dict that carries no timing did not fold anything."""
    af2 = _af2()
    assert af2.consecutive_failures([{"error": "boom"}]) == 1


def test_the_driver_consults_the_circuit_breaker() -> None:
    af2 = _af2()
    source = _function_source(af2, "drive")
    assert "failure_verdict(" in source
    assert "consecutive_failures(batch_results)" in source


def test_the_hardware_is_one_that_has_actually_worked() -> None:
    """Chosen on evidence, not on the price list.

    L4 was picked purely on price at 0.80 an hour, and four consecutive jobs of
    under 200 residues then hit the 1200 second timeout having produced no
    ColabFold output at all: two oneDNN lines at import and then silence. The
    only step in between is CUDA initialisation, which Modal documents as
    intermittently hanging on some L4 instances.

    Every job that has ever completed in this project ran on an A100. Cost is
    held down by the timeout, the container count and the circuit breaker, not
    by picking the cheapest card and hoping.
    """
    from interface_charge.config import DEFAULT_CONFIG, MODAL_GPU_RATES_USD_PER_HOUR

    params = DEFAULT_CONFIG.af2
    assert params.gpu_type in MODAL_GPU_RATES_USD_PER_HOUR, (
        "a GPU with no recorded rate cannot be budgeted for"
    )
    assert params.gpu_type.startswith(("A100", "A10G", "L40S")), (
        f"{params.gpu_type} has not been shown to complete a job in this project"
    )


def test_containers_are_few_enough_to_amortise_the_model_load() -> None:
    """Every container loads several GB of parameters, and that load is billed.

    Forty containers over 99 jobs paid it forty times instead of six. Wide
    fan-out also multiplies exposure to preemption, and a container reclaimed
    mid-job loses its work having been billed for it.
    """
    from interface_charge.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.af2.max_containers <= 10


def test_the_budget_guard_knows_what_the_hardware_costs() -> None:
    """The rate was hardcoded at 2.10 regardless of which card was requested.

    Switching to an L4 left the guard pricing every second at A100 rates. That
    errs towards stopping rather than overspending, so it was not dangerous, but
    a guard that does not know the price of the hardware cannot be reasoned
    about, and the same defect in the other direction would be.
    """
    from interface_charge.config import MODAL_GPU_RATES_USD_PER_HOUR, AF2Params

    for gpu, rate in MODAL_GPU_RATES_USD_PER_HOUR.items():
        assert AF2Params(gpu_type=gpu).rate_usd_per_hour() == rate


def test_an_explicit_rate_still_wins() -> None:
    from interface_charge.config import AF2Params

    assert AF2Params(gpu_type="L4", usd_per_gpu_hour=9.99).rate_usd_per_hour() == 9.99


def test_an_unknown_gpu_has_no_price_and_says_so() -> None:
    """Silently costing an unknown card at zero would disable the ceiling."""
    from interface_charge.config import AF2Params

    with pytest.raises(ValueError, match="not a guard"):
        AF2Params(gpu_type="RTX4090").rate_usd_per_hour()


def test_a_single_stuck_job_cannot_cost_much() -> None:
    """The number that mattered most last night, and nobody was watching it.

    A job that hangs is capped by the timeout times the hourly rate. It was two
    hours on an A100, which is 4.20 a time, and 2ZXE spent that three times over
    and produced nothing.
    """
    from interface_charge.config import DEFAULT_CONFIG

    params = DEFAULT_CONFIG.af2
    worst_case = params.timeout_s / 3600.0 * params.rate_usd_per_hour()
    assert worst_case <= 1.00, f"a stuck job can still cost ${worst_case:.2f}"
    # And the whole batch that must fail before the breaker reacts. That is the
    # real exposure: a systematic fault burns a full chunk before anything stops.
    batch = worst_case * params.max_consecutive_failures
    assert batch <= 4.00, f"a fully failing batch costs ${batch:.2f} before it stops"


def test_the_ledger_can_be_corrected_from_provider_billing() -> None:
    """The pessimistic charge is right for stopping a run and wrong as a record.

    Failures are charged at the full timeout per attempt, because a guard that
    under-charges them cannot stop a run that is only failing. But 20 failures at
    the old two-hour timeout wrote $352.80 into the ledger against a real bill
    several times smaller, and then refused every subsequent launch. The
    provider's number is authoritative; this estimate is not, and must be
    correctable by someone holding the real one.
    """
    af2 = _af2()
    source = _function_source(af2, "launch")
    assert "set_ledger_usd" in source
    assert "provider billing" in source


def test_an_incremental_budget_is_available() -> None:
    """A cumulative ceiling is correct and unusable on its own.

    Asking for twelve more dollars should not require knowing and then adding
    the running total by hand.
    """
    af2 = _af2()
    source = _function_source(af2, "launch")
    assert "additional_usd" in source
    assert "recorded + additional_usd" in source


def test_correcting_the_ledger_alone_does_not_launch_anything() -> None:
    """Setting a number is an accounting action, not an authorisation to spend."""
    af2 = _af2()
    source = _function_source(af2, "launch")
    correction = source[source.index("set_ledger_usd >= 0.0") :]
    assert "if budget_usd <= 0.0 and additional_usd <= 0.0:" in correction
    assert "return" in correction


def test_the_driver_is_probed_before_jax_is_asked_for_devices() -> None:
    """Four jobs hung for 1200s each with no output beyond the import banner.

    The only step between JAX importing and ColabFold's first print is CUDA
    device initialisation, and Modal documents that hanging intermittently.
    nvidia-smi forces the driver up first, so a broken container dies in seconds
    with something readable instead of hanging where there is nothing to see.
    """
    af2 = _af2()
    source = _function_source(af2, "require_gpu_backend")
    assert "nvidia-smi" in source
    assert source.index("nvidia-smi") < source.index("jax.devices()")
    assert "timeout=120" in source, "the probe itself must not be able to hang"
