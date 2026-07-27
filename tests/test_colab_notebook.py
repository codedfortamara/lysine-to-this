"""The Colab notebook must stay identical to the Modal source it shares code with.

Two GPU backends computing the same metric differently is the kind of problem
that produces two numbers in one paper and no way to tell which is right. The
notebook is therefore generated from ``modal_app/af2_multimer.py`` rather than
written alongside it, and these tests fail if the committed notebook has fallen
behind the source.

The failure mode being prevented is specific: someone fixes ``ipsae`` in the
Modal app, the notebook keeps the old version, and half the results come back
computed one way and half the other with nothing in either output to say so.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "af2_colab.ipynb"
GENERATOR = ROOT / "scripts" / "make_colab_notebook.py"


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.is_file():
        pytest.fail(f"{NOTEBOOK} is missing; regenerate with {GENERATOR.name}")
    return json.loads(NOTEBOOK.read_text())


def source_of(notebook: dict, cell_type: str = "code") -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == cell_type
    )


# ---------------------------------------------------------------------------
# The reason this file exists
# ---------------------------------------------------------------------------


def test_the_committed_notebook_matches_a_fresh_build() -> None:
    """Regenerate and compare. A difference means the two backends have diverged."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check", "--out", str(NOTEBOOK)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_shared_functions_are_present_verbatim(notebook: dict) -> None:
    """The metrics code in the notebook is the Modal code, character for character."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_colab_notebook import SHARED_NAMES, SOURCE, extract

    shared = extract(ROOT / SOURCE, SHARED_NAMES)
    assert shared in source_of(notebook), (
        "the notebook does not contain the shared block verbatim; regenerate it"
    )


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_it_is_a_valid_notebook(notebook: dict) -> None:
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["source"], list)


def test_it_asks_for_a_gpu(notebook: dict) -> None:
    assert notebook["metadata"]["accelerator"] == "GPU"


def test_the_pins_match_the_modal_image(notebook: dict) -> None:
    """A different colabfold on Colab would be a different experiment."""
    sys.path.insert(0, str(ROOT / "modal_app"))
    from af2_multimer import IMAGE_PACKAGES, JAX_PACKAGE

    source = source_of(notebook)
    for package in [*IMAGE_PACKAGES, JAX_PACKAGE]:
        assert package in source, f"{package!r} is pinned in the Modal image but not the notebook"


# ---------------------------------------------------------------------------
# The hazards the notebook has to handle, since Colab is not Modal
# ---------------------------------------------------------------------------


def test_the_alignment_is_passed_as_a_list(notebook: dict) -> None:
    """The silent-fallback bug, which would be just as silent here."""
    source = source_of(notebook)
    assert "[a3m]" in source, "the a3m must be wrapped in a list, not passed as a string"
    assert "require_parseable_complex_a3m(a3m, lengths)" in source


def test_it_is_resumable(notebook: dict) -> None:
    """Colab sessions disconnect without warning, so progress has to survive."""
    source = source_of(notebook)
    assert "metrics_path.is_file()" in source, "completed jobs must be skipped on re-run"
    assert "drive/MyDrive" in source, "results must be written somewhere that outlives the session"


def test_it_refuses_jobs_too_large_for_the_gpu(notebook: dict) -> None:
    """A T4 cannot fit the largest complexes, and an OOM takes the session with it."""
    source = source_of(notebook)
    assert "MAX_TOTAL_RESIDUES" in source
    assert "total_memory" in source, "the cap must come from the actual GPU, not a guess"


def test_skipped_jobs_are_recorded_rather_than_dropped(notebook: dict) -> None:
    """A job silently missing from the output would bias every downstream average."""
    source = source_of(notebook)
    assert '"skipped": True' in source
    assert '"reason"' in source


def test_it_runs_smallest_first(notebook: dict) -> None:
    """Opposite of Modal, deliberately: bank completed jobs before a disconnect."""
    source = source_of(notebook)
    assert 'jobs.sort(key=lambda j: sum(len(s) for s in j["chains"].values()))' in source


def test_the_msa_treatment_matches_modal(notebook: dict) -> None:
    """Same mixed-mode policy, or the two backends answer different questions."""
    source = source_of(notebook)
    assert '"single_sequence"' in source
    assert "paired_msa=None" in source
    assert "use_pairing=False" in source


def test_the_msa_cache_is_keyed_by_complex_not_by_job(notebook: dict) -> None:
    """Otherwise it hammers the free MMseqs2 server once per beta instead of once."""
    source = source_of(notebook)
    assert "f\"{job['pdb_id']}_{chain_id}.a3m\"" in source


def test_it_identifies_itself_to_the_mmseqs_server(notebook: dict) -> None:
    assert "user_agent=MMSEQS_USER_AGENT" in source_of(notebook)


def test_results_are_tagged_with_their_backend(notebook: dict) -> None:
    """Pooling Colab and Modal results needs the origin recorded on every row."""
    assert '"source": "colab"' in source_of(notebook)
