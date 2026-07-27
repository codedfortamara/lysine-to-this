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
    JAX_PACKAGE,
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
