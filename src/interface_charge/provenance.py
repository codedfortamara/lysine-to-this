"""Run manifests. Every number in the paper traces to exactly one run.

Every script writes a JSON manifest beside its output recording the SHA256 of
each input file, the git commit and working-tree state, the versions of every
package that could change a result, the random seeds, the full parameter block,
and a UTC timestamp. Figures carry the manifest hash in their file metadata.

The point of the manifest hash
------------------------------
The hash is computed over the *inputs* to a run: the code commit, the input
file digests, the parameters and the seeds. It deliberately excludes the
outputs. That ordering is what makes it useful: the hash is known before the
first output is written, so a table and a figure produced by the same run can
both be stamped with it, and a reviewer asking whether Figure 2 and Table 1
came from the same run can compare two strings instead of taking it on trust.

Output digests are recorded too, appended after the fact, so a manifest also
proves which files a run produced. They simply do not participate in the hash.

Verification
------------
:func:`verify_manifest` re-hashes the recorded inputs and reports any that have
changed since the run. Run it before submission. If an input has moved on and
the outputs have not, the outputs are stale and must be regenerated.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "FileRecord",
    "Manifest",
    "ManifestError",
    "figure_metadata",
    "manifest_path_for",
    "read_manifest",
    "run_manifest",
    "sha256_file",
    "verify_manifest",
]

#: Packages whose version can change a numerical result. Recorded on every run.
#: Extend this list rather than recording the whole environment, which would
#: produce a manifest that differs on every machine for no scientific reason.
TRACKED_PACKAGES: tuple[str, ...] = (
    "biopython",
    "numpy",
    "pandas",
    "freesasa",
    "matplotlib",
    "scipy",
    "modal",
)

MANIFEST_SUFFIX = ".manifest.json"

#: Bumped when the manifest layout changes in a way that breaks readers.
MANIFEST_SCHEMA_VERSION = 1


class ManifestError(RuntimeError):
    """Raised when a manifest cannot be written, read or verified."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 of a file's bytes, streamed so that large structures are fine."""
    if not path.is_file():
        raise ManifestError(f"cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_payload(payload: Any) -> str:
    """SHA256 of the canonical JSON encoding of a Python object."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state(repo_root: Path) -> dict[str, Any]:
    """Commit, branch and dirty flag for the working tree.

    ``dirty`` being true means the run cannot be reproduced from the commit
    alone. That is not an error while developing, but a manifest with
    ``dirty: true`` must not back a number in a submitted paper, and
    :func:`verify_manifest` says so.
    """
    commit = _git(["rev-parse", "HEAD"], repo_root)
    status = _git(["status", "--porcelain"], repo_root)
    return {
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
        "dirty": bool(status) if status is not None else None,
        "dirty_paths": sorted(line[3:] for line in status.splitlines()[:50]) if status else [],
    }


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed version of each tracked package, or None if absent."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment() -> dict[str, Any]:
    """Interpreter and platform details that can change a floating point result."""
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FileRecord:
    """One file, identified by path and content digest."""

    path: str
    sha256: str
    size_bytes: int

    @classmethod
    def of(cls, path: Path, relative_to: Path | None = None) -> FileRecord:
        recorded = path
        if relative_to is not None:
            try:
                recorded = path.resolve().relative_to(relative_to.resolve())
            except ValueError:
                recorded = path.resolve()
        return cls(
            path=str(recorded),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )


@dataclass
class Manifest:
    """The full provenance record of one script run."""

    script: str
    script_sha256: str | None
    created_utc: str
    git: dict[str, Any]
    environment: dict[str, Any]
    parameters: dict[str, Any]
    seeds: dict[str, Any]
    inputs: list[FileRecord] = field(default_factory=list)
    outputs: list[FileRecord] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    schema_version: int = MANIFEST_SCHEMA_VERSION

    # -- hashing ---------------------------------------------------------

    def hash_payload(self) -> dict[str, Any]:
        """The subset of the manifest that the manifest hash covers.

        Outputs, status and the timestamp are excluded. Two runs of the same
        code over the same inputs with the same parameters therefore share a
        manifest hash, which is what makes the hash a statement about
        *provenance* rather than about *when someone happened to press go*.
        """
        return {
            "script": self.script,
            "script_sha256": self.script_sha256,
            "git_commit": self.git.get("commit"),
            "git_dirty": self.git.get("dirty"),
            "environment": self.environment,
            "parameters": self.parameters,
            "seeds": self.seeds,
            "inputs": [asdict(record) for record in self.inputs],
            "schema_version": self.schema_version,
        }

    @property
    def manifest_hash(self) -> str:
        """Stable identifier of this run's inputs, code and parameters."""
        return sha256_payload(self.hash_payload())

    @property
    def short_hash(self) -> str:
        """First twelve hex characters of :attr:`manifest_hash`, for figure captions."""
        return self.manifest_hash[:12]

    # -- mutation --------------------------------------------------------

    def add_input(self, path: Path, relative_to: Path | None = None) -> FileRecord:
        """Record an input file. Changes the manifest hash, so do this first."""
        record = FileRecord.of(path, relative_to=relative_to)
        self.inputs.append(record)
        return record

    def add_output(self, path: Path, relative_to: Path | None = None) -> FileRecord:
        """Record an output file. Does not change the manifest hash."""
        record = FileRecord.of(path, relative_to=relative_to)
        self.outputs.append(record)
        return record

    def note(self, key: str, value: Any) -> None:
        """Attach a free-form observation, for example a count of skipped rows."""
        self.notes[key] = value

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["manifest_hash"] = self.manifest_hash
        return payload

    def write(self, path: Path) -> Path:
        """Write the manifest as JSON and return the path written."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n")
        return path


def manifest_path_for(output_path: Path) -> Path:
    """The manifest path that sits beside a given output file."""
    return output_path.with_name(output_path.name + MANIFEST_SUFFIX)


def read_manifest(path: Path) -> dict[str, Any]:
    """Load a manifest JSON file."""
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Entry point used by the scripts
# ---------------------------------------------------------------------------


@contextmanager
def run_manifest(
    script: Path,
    parameters: dict[str, Any],
    seeds: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    inputs: list[Path] | None = None,
) -> Iterator[Manifest]:
    """Context manager wrapping one script run.

    On a clean exit the manifest status is ``completed``. If the body raises,
    the status is ``failed`` and the exception type and message are recorded,
    then the exception propagates. A failed run therefore still leaves a
    manifest, which is what lets you tell "this output is from a crashed run"
    apart from "this output was never produced".

    The caller is responsible for calling :meth:`Manifest.write` with the path
    it wants, normally via :func:`manifest_path_for`. Writing is not automatic
    because a script may produce several outputs and has to choose which one
    the manifest sits beside.
    """
    repo_root = repo_root or Path(__file__).resolve().parents[2]

    manifest = Manifest(
        script=script.name,
        script_sha256=sha256_file(script) if script.is_file() else None,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        git=git_state(repo_root),
        environment=environment(),
        parameters=parameters,
        seeds=seeds or {},
    )

    for path in inputs or []:
        manifest.add_input(path, relative_to=repo_root)

    try:
        yield manifest
    except BaseException as exc:
        manifest.status = "failed"
        manifest.note("error_type", type(exc).__name__)
        manifest.note("error_message", str(exc)[:2000])
        raise
    else:
        manifest.status = "completed"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_metadata(manifest: Manifest, title: str | None = None) -> dict[str, str]:
    """Metadata dict for ``matplotlib.figure.Figure.savefig``.

    Matplotlib passes these through to the file. For PNG they become tEXt
    chunks; for PDF and SVG they map onto the document information keys. Either
    way the manifest hash travels with the figure file, so a figure that has
    drifted out of a directory can still be traced back to its run.
    """
    payload = {
        "Title": title or "interface-charge-rcsb figure",
        "Author": "interface-charge-rcsb",
        "Subject": f"manifest_hash={manifest.manifest_hash}",
        "Creator": f"interface-charge-rcsb, git={manifest.git.get('commit')}",
        "Keywords": (
            f"manifest_hash={manifest.manifest_hash} "
            f"git_commit={manifest.git.get('commit')} "
            f"git_dirty={manifest.git.get('dirty')} "
            f"created={manifest.created_utc}"
        ),
    }
    return {k: v for k, v in payload.items() if v is not None}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_manifest(path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    """Re-hash a manifest's recorded inputs and outputs and report drift.

    Returns a report dict with an ``ok`` flag and lists of any files that are
    missing or whose digest no longer matches. Also flags a manifest recorded
    against a dirty working tree, which cannot be reproduced from its commit.
    """
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    payload = read_manifest(path)

    changed: list[dict[str, str]] = []
    missing: list[str] = []

    for kind in ("inputs", "outputs"):
        for record in payload.get(kind, []):
            recorded_path = Path(record["path"])
            candidate = recorded_path if recorded_path.is_absolute() else repo_root / recorded_path
            if not candidate.is_file():
                missing.append(f"{kind}:{record['path']}")
                continue
            current = sha256_file(candidate)
            if current != record["sha256"]:
                changed.append(
                    {
                        "kind": kind,
                        "path": record["path"],
                        "recorded_sha256": record["sha256"],
                        "current_sha256": current,
                    }
                )

    dirty = payload.get("git", {}).get("dirty")
    return {
        "manifest": str(path),
        "manifest_hash": payload.get("manifest_hash"),
        "status": payload.get("status"),
        "git_dirty": dirty,
        "missing": missing,
        "changed": changed,
        "ok": not missing and not changed and payload.get("status") == "completed",
        "publication_ready": (
            not missing and not changed and payload.get("status") == "completed" and dirty is False
        ),
    }


def describe_environment_for_log() -> str:
    """One-line environment summary for a script's stdout banner."""
    env = environment()
    versions = ", ".join(
        f"{name} {version}" for name, version in env["packages"].items() if version
    )
    return f"python {env['python']} on {env['machine']}; {versions}"


def default_repo_root() -> Path:
    """Repository root, honouring an override for out-of-tree runs."""
    override = os.environ.get("INTERFACE_CHARGE_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]
