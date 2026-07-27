"""The container's dependency pins must satisfy colabfold's own constraints.

This exists because they did not. The image asked for ``biopython>=1.83`` and
``numpy>=1.26`` while colabfold 1.5.5 declares ``biopython<1.83`` and
``numpy<2.0.0``, so the build was unresolvable. Nothing caught it locally: the
conflict only surfaces inside a Modal image build, several minutes in, after the
apt layer has already been built.

The constraints below are transcribed from colabfold 1.5.5's published metadata
on PyPI. They are checked against the pins the image actually uses, which is why
those pins live at module level rather than inside the ``MODAL_AVAILABLE``
block: this test runs whether or not Modal is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))

from af2_multimer import (
    COLABFOLD_VERSION,
    IMAGE_PACKAGES,
    IMAGE_PYTHON_VERSION,
    JAX_MAX_WITH_LINEAR_UTIL,
    JAX_PACKAGE,
    LOCAL_PYTHON_SOURCES,
)

#: From ``pypi.org/pypi/colabfold/1.5.5/json``, ``info.requires_dist``.
COLABFOLD_REQUIRES: dict[str, tuple[str | None, str | None]] = {
    "biopython": (None, "1.83"),
    "numpy": ("1.22.0", "2.0.0"),
    "jax": ("0.4.20", "0.5.0"),
    "matplotlib": ("3.2.2", "4.0.0"),
    "pandas": ("1.3.4", "2.0.0"),
}

#: ``info.requires_python`` for the same release.
COLABFOLD_PYTHON = ("3.9", "3.12")


def as_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def parse_pin(spec: str) -> tuple[str, list[tuple[str, str]]]:
    """Split ``name[extra]>=1.2,<3.4`` into a name and a list of (operator, version)."""
    import re

    match = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?(.*)$", spec)
    assert match, f"could not parse pin {spec!r}"
    name = match.group(1).lower()
    constraints = []
    for part in match.group(3).split(","):
        part = part.strip()
        if not part:
            continue
        operator = re.match(r"^(>=|<=|==|<|>|!=|~=)", part)
        assert operator, f"could not parse constraint {part!r} in {spec!r}"
        constraints.append((operator.group(1), part[len(operator.group(1)) :].strip()))
    return name, constraints


@pytest.mark.parametrize("spec", IMAGE_PACKAGES)
def test_every_image_pin_is_parseable(spec: str) -> None:
    name, constraints = parse_pin(spec)
    assert name
    assert constraints, f"{spec!r} has no version constraint, so the build is not reproducible"


def test_pins_do_not_contradict_colabfold() -> None:
    """The failure that cost a build: a lower bound above colabfold's upper bound."""
    for spec in IMAGE_PACKAGES:
        name, constraints = parse_pin(spec)
        if name not in COLABFOLD_REQUIRES:
            continue
        floor, ceiling = COLABFOLD_REQUIRES[name]
        for operator, version in constraints:
            if operator == ">=" and ceiling is not None:
                assert as_tuple(version) < as_tuple(ceiling), (
                    f"{spec!r} demands {name} at least {version}, but colabfold "
                    f"{COLABFOLD_VERSION} requires below {ceiling}. The image will "
                    "not resolve."
                )
            if operator in ("<", "<=") and floor is not None:
                assert as_tuple(version) >= as_tuple(floor), (
                    f"{spec!r} caps {name} at {version}, below colabfold's minimum of {floor}."
                )


def test_the_jax_pin_sits_inside_colabfold_window() -> None:
    name, constraints = parse_pin(JAX_PACKAGE)
    assert name == "jax"
    pinned = [v for op, v in constraints if op == "=="]
    assert pinned, "jax should be pinned exactly, since the CUDA build must match"
    floor, ceiling = COLABFOLD_REQUIRES["jax"]
    assert as_tuple(floor) <= as_tuple(pinned[0]) < as_tuple(ceiling)


def test_jax_is_old_enough_for_haiku() -> None:
    """The constraint colabfold's own bound does not express, and which cost a run.

    colabfold 1.5.5 pins dm-haiku==0.0.10, which imports jax.linear_util. JAX
    removed that module in 0.4.24. So a version can satisfy colabfold's declared
    jax>=0.4.20,<0.5.0 and still die at import with

        AttributeError: module 'jax' has no attribute 'linear_util'

    which is exactly what happened with 0.4.28. Verified against the published
    wheels: jax/linear_util.py is present through 0.4.23 and gone from 0.4.24.
    """
    pinned = next(v for op, v in parse_pin(JAX_PACKAGE)[1] if op == "==")
    assert as_tuple(pinned) <= as_tuple(JAX_MAX_WITH_LINEAR_UTIL), (
        f"jax {pinned} is at or above 0.4.24, which removed jax.linear_util. "
        f"dm-haiku 0.0.10 imports it, so every prediction will fail at import. "
        f"The highest usable version is {JAX_MAX_WITH_LINEAR_UTIL}."
    )


def test_the_cuda_extra_pulls_a_gpu_build() -> None:
    """At 0.4.23 the plain cuda12 extra is not the monolithic CUDA wheel.

    cuda12_pip resolves to jaxlib==0.4.23+cuda12.cudnn89, which is the build the
    jax-releases index carries a cp311 wheel for. Getting this wrong is not a
    crash, it is a silent fall back to CPU, which on this grid means jobs that
    take hours instead of minutes and a bill to match.
    """
    name, _ = parse_pin(JAX_PACKAGE)
    assert name == "jax"
    assert "[cuda12_pip]" in JAX_PACKAGE, (
        f"{JAX_PACKAGE!r} does not request the cuda12_pip extra; a CPU jaxlib "
        "would run but take orders of magnitude longer"
    )


def test_the_image_python_is_one_colabfold_supports() -> None:
    """colabfold 1.5.5 does not publish wheels for 3.12 or later."""
    floor, ceiling = COLABFOLD_PYTHON
    assert as_tuple(floor) <= as_tuple(IMAGE_PYTHON_VERSION) < as_tuple(ceiling)


def test_colabfold_itself_is_pinned_exactly() -> None:
    """A floating colabfold would silently move every bound above."""
    matching = [s for s in IMAGE_PACKAGES if parse_pin(s)[0] == "colabfold"]
    assert len(matching) == 1
    assert f"=={COLABFOLD_VERSION}" in matching[0]


def test_the_alphafold_extra_is_requested() -> None:
    """Without it colabfold installs with no model to run."""
    matching = [s for s in IMAGE_PACKAGES if parse_pin(s)[0] == "colabfold"]
    assert "[alphafold]" in matching[0]


# ---------------------------------------------------------------------------
# Local source
# ---------------------------------------------------------------------------


def repository_packages() -> set[str]:
    """Top-level packages under ``src/``, which is what the container can miss."""
    src = Path(__file__).resolve().parents[1] / "src"
    return {p.name for p in src.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}


def top_level_imports() -> set[str]:
    """Packages ``af2_multimer`` imports at module scope, by reading its source.

    Parsed rather than introspected because the question is what the *container*
    will need at import time, and the container imports the module fresh.
    """
    import ast

    source = (Path(__file__).resolve().parents[1] / "modal_app" / "af2_multimer.py").read_text()
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:  # module scope only, not function-local imports
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_every_repository_package_imported_at_module_scope_is_shipped() -> None:
    """The failure this exists for: containers crash-looping on ModuleNotFoundError.

    Modal mounts the entrypoint module and nothing else. A package imported from
    ``src/`` resolves locally through the ``sys.path`` insert at the top of the
    file, so a dry run and the whole test suite pass, and the first sign of
    trouble is every container dying on import after the GPUs are allocated.
    """
    needed = top_level_imports() & repository_packages()
    missing = needed - set(LOCAL_PYTHON_SOURCES)
    assert not missing, (
        f"af2_multimer imports {sorted(missing)} at module scope, but "
        f"LOCAL_PYTHON_SOURCES ships {LOCAL_PYTHON_SOURCES}. Every container will "
        "fail to start."
    )


def test_nothing_is_shipped_that_does_not_exist() -> None:
    """A stale entry would be a silent no-op rather than an error."""
    unknown = set(LOCAL_PYTHON_SOURCES) - repository_packages()
    assert not unknown, f"LOCAL_PYTHON_SOURCES names {sorted(unknown)}, absent from src/"


def test_interface_charge_is_the_package_actually_required() -> None:
    """Guards the test above against passing because both sides are empty."""
    assert "interface_charge" in repository_packages()
    assert "interface_charge" in top_level_imports()
