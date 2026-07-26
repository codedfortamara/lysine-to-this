"""Shared argument parsing, so that the scripts stay thin.

Every script takes the same handful of options (config override, output
directory, extra-column tolerance, verbosity) and every script needs the same
banner and the same config resolution. Putting that here keeps
``scripts/*.py`` down to argument parsing plus calls into the package, which is
the stated convention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, Config, load_overrides
from .provenance import describe_environment_for_log

__all__ = [
    "add_common_arguments",
    "banner",
    "fail",
    "resolve_config",
]


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the options every script shares."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "TOML file overriding values in interface_charge.config. "
            "Unknown keys raise rather than being ignored."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for outputs and manifests. Defaults to results/ in the repository.",
    )
    parser.add_argument(
        "--allow-extra-columns",
        action="store_true",
        help=(
            "Tolerate columns beyond the documented contract. Off by default so "
            "that a renamed column cannot pass unnoticed. If you use this, "
            "record the change in data/README.md."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing outputs rather than reusing them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the progress banner.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> Config:
    """Return the configuration for this run, applying any TOML override."""
    if args.config is None:
        return DEFAULT_CONFIG
    return load_overrides(args.config)


def banner(script_name: str, args: argparse.Namespace, extra: dict[str, Any] | None = None) -> None:
    """Print a one-off header describing what is about to run."""
    if getattr(args, "quiet", False):
        return
    print(f"=== {script_name} ===", file=sys.stderr)
    print(describe_environment_for_log(), file=sys.stderr)
    if extra:
        for key, value in extra.items():
            print(f"  {key}: {value}", file=sys.stderr)


def fail(message: str, code: int = 2) -> None:
    """Exit with a message on stderr.

    Used for the "real data is absent" path, which must be a clean, explanatory
    failure rather than a traceback, because it is the most common way these
    scripts will be run wrongly.
    """
    print(f"\nERROR: {message}\n", file=sys.stderr)
    raise SystemExit(code)
