"""The interface RMSD step, exercised against real native structures.

This metric has never been computed for a single job. ``predict`` looked for the
native at a path on the results volume that nothing has ever written, so
``interface_rmsd_a`` was null in all 37 rows and the note blamed a missing
native. The column was present and entirely empty, which reads as data.

The tests use the actual PDB files in ``data/native/`` where they are available,
because the hazards here are all in real numbering: author residue numbers that
start at 105, unmodelled gaps, insertion codes. A synthetic fixture numbered
from 1 would pass while the real thing silently compared the wrong residues.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "14_interface_rmsd.py"
NATIVES = REPO_ROOT / "data" / "native"

pytestmark = pytest.mark.skipif(
    not NATIVES.is_dir() or not any(NATIVES.glob("*.pdb")),
    reason="native structures are not committed; this step needs the real ones",
)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
        check=False,
    )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_it_fails_cleanly_without_a_metrics_table(tmp_path: Path) -> None:
    result = run("--from-dir", str(tmp_path), "--metrics", str(tmp_path / "absent.csv"))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "collect.py" in result.stderr


def test_it_fails_cleanly_without_the_downloaded_predictions(tmp_path: Path) -> None:
    metrics = tmp_path / "af2_metrics.csv"
    pd.DataFrame([{"key": "x", "pdb_id": "1B4U"}]).to_csv(metrics, index=False)
    result = run("--from-dir", str(tmp_path / "absent"), "--metrics", str(metrics))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "modal volume get" in result.stderr


# ---------------------------------------------------------------------------
# Every row is accounted for, one way or the other
# ---------------------------------------------------------------------------


def test_a_missing_prediction_is_recorded_with_its_reason(tmp_path: Path) -> None:
    """The failure this whole script exists to stop: a null with no explanation."""
    metrics = tmp_path / "af2_metrics.csv"
    pd.DataFrame(
        [
            {
                "key": "1B4U_B_betap0_0000_rep0",
                "pdb_id": "1B4U",
                "designed_chain": "B",
                "beta": 0.0,
                "replicate": 0,
                "predicted_pdb": "nothing_here.pdb",
            }
        ]
    ).to_csv(metrics, index=False)

    result = run(
        "--from-dir",
        str(tmp_path),
        "--metrics",
        str(metrics),
        "--results-dir",
        str(tmp_path / "results"),
    )
    assert result.returncode == 0, result.stderr

    written = pd.read_csv(tmp_path / "results" / "af2_metrics.csv")
    assert written["interface_rmsd_a"].isna().all()
    note = written["interface_rmsd_note"].iloc[0]
    assert note and "missing" in note.lower()
    assert "skipped" in result.stderr


def test_every_row_gets_a_verdict(tmp_path: Path) -> None:
    """No row may come back with neither a number nor a reason."""
    metrics = tmp_path / "af2_metrics.csv"
    pd.DataFrame(
        [
            {
                "key": f"1B4U_B_betap{i}_0000_rep0",
                "pdb_id": "1B4U",
                "designed_chain": "B",
                "beta": float(i),
                "replicate": 0,
                "predicted_pdb": "",
            }
            for i in range(3)
        ]
    ).to_csv(metrics, index=False)

    result = run(
        "--from-dir",
        str(tmp_path),
        "--metrics",
        str(metrics),
        "--results-dir",
        str(tmp_path / "results"),
    )
    assert result.returncode == 0, result.stderr

    written = pd.read_csv(tmp_path / "results" / "af2_metrics.csv")
    assert len(written) == 3
    both_absent = written["interface_rmsd_a"].isna() & written["interface_rmsd_note"].isna()
    assert not both_absent.any(), "a row with no number and no reason is the original bug"


def test_it_writes_a_manifest_recording_what_it_skipped(tmp_path: Path) -> None:
    metrics = tmp_path / "af2_metrics.csv"
    pd.DataFrame(
        [
            {
                "key": "unknown_key",
                "pdb_id": "1B4U",
                "designed_chain": "B",
                "beta": 0.0,
                "replicate": 0,
                "predicted_pdb": "x.pdb",
            }
        ]
    ).to_csv(metrics, index=False)

    result = run(
        "--from-dir",
        str(tmp_path),
        "--metrics",
        str(metrics),
        "--results-dir",
        str(tmp_path / "results"),
    )
    assert result.returncode == 0, result.stderr

    manifest = json.loads((tmp_path / "results" / "af2_metrics.csv.manifest.json").read_text())
    blob = json.dumps(manifest)
    assert "skip_reasons" in blob
    assert "n_computed" in blob


# ---------------------------------------------------------------------------
# The real calculation, against a native superposed on itself
# ---------------------------------------------------------------------------


def test_a_native_against_itself_gives_almost_zero(tmp_path: Path) -> None:
    """The strongest available check without a real prediction to hand.

    Feeding the native in as its own prediction should superpose exactly and
    give an interface RMSD indistinguishable from zero. Anything else means the
    chain mapping, the residue pairing or the superposition is wrong, and it
    would be wrong in a way that produces a plausible number rather than an
    error.
    """
    sys.path.insert(0, str(REPO_ROOT / "modal_app"))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from af2_multimer import build_job_list, interface_rmsd

    definitions = REPO_ROOT / "results" / "interface_definitions.json"
    designs = REPO_ROOT / "data" / "raw" / "designs.csv"
    test_set = REPO_ROOT / "data" / "raw" / "test_set.csv"
    if not (definitions.is_file() and designs.is_file() and test_set.is_file()):
        pytest.skip("the real input tables are local only")

    jobs = build_job_list(designs, definitions, test_set)
    job = next(j for j in jobs if (NATIVES / f"{j.pdb_id}.pdb").is_file())
    native = NATIVES / f"{job.pdb_id}.pdb"

    result = interface_rmsd(
        native_path=native,
        predicted_path=native,
        designed_chain=job.designed_chain,
        chain_order=sorted(job.chains),
    )

    assert result["interface_rmsd_a"] is not None, result["interface_rmsd_note"]
    assert result["interface_rmsd_a"] < 0.01, (
        f"a structure against itself gave {result['interface_rmsd_a']:.3f} A, "
        "so the pairing or superposition is wrong"
    )
    assert result["n_interface_residues_compared"] > 0
