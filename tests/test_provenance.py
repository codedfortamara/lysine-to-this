"""Run manifests: hashing, stability, and what the hash does and does not cover.

The manifest hash is the mechanism that lets a reviewer confirm a table and a
figure came from the same run. The tests below pin the two properties that make
that work: the hash covers the inputs, and it does not cover the outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interface_charge.provenance import (
    Manifest,
    ManifestError,
    figure_metadata,
    manifest_path_for,
    read_manifest,
    run_manifest,
    sha256_file,
    sha256_payload,
    verify_manifest,
)


@pytest.fixture
def input_file(tmp_path) -> Path:
    path = tmp_path / "input.csv"
    path.write_text("a,b\n1,2\n")
    return path


@pytest.fixture
def manifest(tmp_path, input_file) -> Manifest:
    with run_manifest(
        script=Path(__file__),
        parameters={"threshold": 1.0},
        seeds={"random_seed": 0},
        repo_root=tmp_path,
        inputs=[input_file],
    ) as m:
        pass
    return m


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    import hashlib

    path = tmp_path / "f.bin"
    payload = b"some bytes"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_on_missing_file_raises(tmp_path) -> None:
    with pytest.raises(ManifestError, match="missing file"):
        sha256_file(tmp_path / "nope")


def test_canonical_payload_hash_is_key_order_independent() -> None:
    assert sha256_payload({"a": 1, "b": 2}) == sha256_payload({"b": 2, "a": 1})


# ---------------------------------------------------------------------------
# What the manifest hash covers
# ---------------------------------------------------------------------------


def test_manifest_records_everything_required(manifest) -> None:
    payload = manifest.to_dict()
    assert payload["inputs"][0]["sha256"]
    assert "git" in payload
    assert "packages" in payload["environment"]
    assert payload["parameters"] == {"threshold": 1.0}
    assert payload["seeds"] == {"random_seed": 0}
    assert payload["created_utc"].endswith("+00:00")
    assert payload["manifest_hash"]
    assert payload["status"] == "completed"


def test_hash_changes_when_an_input_changes(tmp_path, input_file) -> None:
    def build() -> str:
        with run_manifest(
            script=Path(__file__),
            parameters={},
            repo_root=tmp_path,
            inputs=[input_file],
        ) as m:
            pass
        return m.manifest_hash

    before = build()
    input_file.write_text("a,b\n3,4\n")
    assert build() != before


def test_hash_changes_when_a_parameter_changes(tmp_path, input_file) -> None:
    def build(threshold: float) -> str:
        with run_manifest(
            script=Path(__file__),
            parameters={"threshold": threshold},
            repo_root=tmp_path,
            inputs=[input_file],
        ) as m:
            pass
        return m.manifest_hash

    assert build(1.0) != build(2.0)


def test_hash_changes_when_a_seed_changes(tmp_path, input_file) -> None:
    def build(seed: int) -> str:
        with run_manifest(
            script=Path(__file__),
            parameters={},
            seeds={"random_seed": seed},
            repo_root=tmp_path,
            inputs=[input_file],
        ) as m:
            pass
        return m.manifest_hash

    assert build(0) != build(1)


def test_hash_does_not_change_when_an_output_is_added(manifest, tmp_path) -> None:
    """The property that lets a figure be stamped before it exists.

    If outputs participated in the hash, a table and a figure from one run would
    carry different hashes and the whole mechanism would be useless.
    """
    before = manifest.manifest_hash
    output = tmp_path / "out.csv"
    output.write_text("x\n1\n")
    manifest.add_output(output)
    assert manifest.manifest_hash == before


def test_hash_does_not_change_with_the_timestamp(tmp_path, input_file) -> None:
    """Two identical runs share a hash. It identifies provenance, not an occasion."""
    hashes = []
    for _ in range(2):
        with run_manifest(
            script=Path(__file__),
            parameters={"threshold": 1.0},
            seeds={"random_seed": 0},
            repo_root=tmp_path,
            inputs=[input_file],
        ) as m:
            pass
        hashes.append(m.manifest_hash)
    assert hashes[0] == hashes[1]


def test_notes_do_not_change_the_hash(manifest) -> None:
    before = manifest.manifest_hash
    manifest.note("n_skipped", 3)
    assert manifest.manifest_hash == before


def test_short_hash_is_a_prefix(manifest) -> None:
    assert manifest.manifest_hash.startswith(manifest.short_hash)
    assert len(manifest.short_hash) == 12


# ---------------------------------------------------------------------------
# Failure recording
# ---------------------------------------------------------------------------


def test_a_failed_run_still_produces_a_manifest(tmp_path, input_file) -> None:
    """Distinguishing "output from a crashed run" from "never produced"."""
    holder: dict[str, Manifest] = {}
    with (
        pytest.raises(RuntimeError, match="boom"),
        run_manifest(
            script=Path(__file__), parameters={}, repo_root=tmp_path, inputs=[input_file]
        ) as m,
    ):
        holder["m"] = m
        raise RuntimeError("boom")

    manifest = holder["m"]
    assert manifest.status == "failed"
    assert manifest.notes["error_type"] == "RuntimeError"
    assert "boom" in manifest.notes["error_message"]


# ---------------------------------------------------------------------------
# Round trip and verification
# ---------------------------------------------------------------------------


def test_write_and_read_round_trip(manifest, tmp_path) -> None:
    path = manifest.write(tmp_path / "run.manifest.json")
    payload = read_manifest(path)
    assert payload["manifest_hash"] == manifest.manifest_hash
    assert json.loads(path.read_text()) == payload


def test_manifest_path_sits_beside_the_output() -> None:
    assert manifest_path_for(Path("results/x.csv")).name == "x.csv.manifest.json"


def test_verify_passes_on_an_untouched_run(tmp_path, input_file) -> None:
    with run_manifest(
        script=Path(__file__), parameters={}, repo_root=tmp_path, inputs=[input_file]
    ) as m:
        output = tmp_path / "out.csv"
        output.write_text("x\n1\n")
        m.add_output(output)
    path = m.write(tmp_path / "run.manifest.json")

    report = verify_manifest(path, repo_root=tmp_path)
    assert report["ok"] is True
    assert report["changed"] == []
    assert report["missing"] == []


def test_verify_detects_a_changed_input(tmp_path, input_file) -> None:
    with run_manifest(
        script=Path(__file__), parameters={}, repo_root=tmp_path, inputs=[input_file]
    ) as m:
        pass
    path = m.write(tmp_path / "run.manifest.json")

    input_file.write_text("a,b\n9,9\n")
    report = verify_manifest(path, repo_root=tmp_path)
    assert report["ok"] is False
    assert report["changed"][0]["path"].endswith("input.csv")


def test_verify_detects_a_missing_output(tmp_path, input_file) -> None:
    with run_manifest(
        script=Path(__file__), parameters={}, repo_root=tmp_path, inputs=[input_file]
    ) as m:
        output = tmp_path / "out.csv"
        output.write_text("x\n1\n")
        m.add_output(output)
    path = m.write(tmp_path / "run.manifest.json")

    (tmp_path / "out.csv").unlink()
    report = verify_manifest(path, repo_root=tmp_path)
    assert report["ok"] is False
    assert any("out.csv" in entry for entry in report["missing"])


def test_verify_on_missing_manifest_raises(tmp_path) -> None:
    with pytest.raises(ManifestError, match="not found"):
        verify_manifest(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Figure metadata
# ---------------------------------------------------------------------------


def test_figure_metadata_carries_the_manifest_hash(manifest) -> None:
    metadata = figure_metadata(manifest, title="Figure A")
    assert manifest.manifest_hash in metadata["Subject"]
    assert manifest.manifest_hash in metadata["Keywords"]
    assert metadata["Title"] == "Figure A"


def test_save_figure_embeds_the_hash_in_a_real_png(manifest, tmp_path) -> None:
    """End to end: the hash must survive into the written file."""
    from PIL import PngImagePlugin

    from interface_charge.plotting import new_figure, save_figure

    fig, ax = new_figure()
    ax.plot([0, 1], [0, 1])
    written = save_figure(fig, tmp_path / "fig", manifest, title="T", formats=("png",))

    assert len(written) == 1
    with PngImagePlugin.PngImageFile(written[0]) as image:
        text = image.text
    assert manifest.manifest_hash in text["Subject"]


def test_save_figure_records_outputs_on_the_manifest(manifest, tmp_path) -> None:
    from interface_charge.plotting import new_figure, save_figure

    fig, ax = new_figure()
    ax.plot([0, 1], [0, 1])
    before = len(manifest.outputs)
    save_figure(fig, tmp_path / "fig2", manifest, formats=("png", "pdf"))
    assert len(manifest.outputs) == before + 2
