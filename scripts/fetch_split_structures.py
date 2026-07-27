#!/usr/bin/env python
"""Download the RCSB structures named in a split file.

Standalone by design: no imports from this project, so it can be dropped
anywhere and run. Windows-friendly, since it takes arguments rather than
needing a shell one-liner with quoting that ``cmd.exe`` cannot parse.

Typical use, fetching the sixty structures behind the upstream RCSB arm::

    python scripts/fetch_split_structures.py --out data/pdb --limit 60

With no ``--split-file`` it pulls the split list straight from the collaborator's
repository, so it works without a local checkout of that repository.

Existing files are skipped, so re-running after an interruption resumes rather
than starting over. Downloads go to a temporary name and are renamed on success,
so an interrupted run cannot leave a truncated structure behind that would later
parse into a subtly wrong answer.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_SPLIT_URL = (
    "https://raw.githubusercontent.com/SyedMohammedSameer/ZetaDial/"
    "main/models/finetune_split_test.txt"
)
RCSB_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.{fmt}"


def read_ids(split_file: Path | None, split_url: str, limit: int) -> list[str]:
    if split_file is not None:
        text = split_file.read_text()
    else:
        print(f"fetching split list from {split_url}", file=sys.stderr)
        with urllib.request.urlopen(split_url, timeout=60) as response:
            text = response.read().decode()
    ids = [line.strip().upper() for line in text.split() if line.strip()]
    return ids[:limit] if limit > 0 else ids


def download(pdb_id: str, out_dir: Path, fmt: str, attempts: int, timeout: float) -> str:
    """Return 'skipped', 'ok' or an error string."""
    target = out_dir / f"{pdb_id}.{fmt}"
    if target.is_file() and target.stat().st_size > 0:
        return "skipped"

    url = RCSB_TEMPLATE.format(pdb_id=pdb_id, fmt=fmt)
    tmp = target.with_suffix(target.suffix + ".partial")
    backoff = 2.0

    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "interface-charge-rcsb/0.1 (research use)"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if not payload:
                raise OSError("empty response body")
            tmp.write_bytes(payload)
            tmp.replace(target)
            return "ok"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            if attempt == attempts:
                return f"{type(exc).__name__}: {exc}"
            time.sleep(backoff)
            backoff *= 2.0
    return "unreachable"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/pdb"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split-url", default=DEFAULT_SPLIT_URL)
    parser.add_argument("--limit", type=int, default=60, help="0 for all")
    parser.add_argument("--format", default="pdb", choices=("pdb", "cif"))
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        ids = read_ids(args.split_file, args.split_url, args.limit)
    except OSError as exc:
        print(f"\nERROR: could not read the split list: {exc}", file=sys.stderr)
        return 2

    print(f"{len(ids)} structure(s) to fetch into {args.out}\n", file=sys.stderr)

    failures: list[tuple[str, str]] = []
    fetched = skipped = 0

    for index, pdb_id in enumerate(ids, start=1):
        status = download(pdb_id, args.out, args.format, args.attempts, args.timeout)
        if status == "ok":
            fetched += 1
            print(f"  [{index}/{len(ids)}] {pdb_id}", file=sys.stderr)
        elif status == "skipped":
            skipped += 1
            print(f"  [{index}/{len(ids)}] {pdb_id} (already present)", file=sys.stderr)
        else:
            failures.append((pdb_id, status))
            print(f"  [{index}/{len(ids)}] {pdb_id} FAILED: {status}", file=sys.stderr)

    print(
        f"\n{fetched} downloaded, {skipped} already present, {len(failures)} failed",
        file=sys.stderr,
    )
    if failures:
        print("\nFailed:", file=sys.stderr)
        for pdb_id, reason in failures[:20]:
            print(f"  {pdb_id}: {reason}", file=sys.stderr)
        print(
            "\nRe-run this command to retry only the failures; anything already "
            "downloaded is skipped.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
