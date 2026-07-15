#!/usr/bin/env python3
"""Verify every file recorded in the public release checksum manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "release_checksums.csv",
    )
    args = parser.parse_args()

    failures: list[str] = []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        relative = Path(row["path"])
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative.as_posix()}")
            continue
        expected_bytes = int(row["bytes"])
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            failures.append(
                f"size: {relative.as_posix()} expected={expected_bytes} actual={actual_bytes}"
            )
            continue
        actual_sha256 = sha256(path)
        if actual_sha256 != row["sha256"]:
            failures.append(
                f"sha256: {relative.as_posix()} expected={row['sha256']} actual={actual_sha256}"
            )

    print(f"release files checked: {len(rows)}; failures: {len(failures)}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
