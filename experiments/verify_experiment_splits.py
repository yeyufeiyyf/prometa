#!/usr/bin/env python3
"""Verify every dedicated experiment split/cohort and its provenance."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments" / "experiment_split_manifest.csv",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest).fillna("")
    failures: list[str] = []
    if manifest["split"].duplicated().any():
        failures.append("manifest contains duplicate split paths")
    for _, row in manifest.iterrows():
        experiment_id = str(row["experiment_id"])
        split = ROOT / str(row["split"])
        source = ROOT / str(row["source_split"])
        expected_prefix = Path("splits") / "by_experiment" / experiment_id
        try:
            relative_split = split.relative_to(ROOT)
        except ValueError:
            relative_split = split
        if expected_prefix not in relative_split.parents:
            failures.append(f"{experiment_id}: split is outside its experiment directory")
        if not split.is_file():
            failures.append(f"{experiment_id}: missing split {split}")
            continue
        if sha256(split) != str(row["split_sha256"]):
            failures.append(f"{experiment_id}: split SHA-256 mismatch")
        if not source.is_file():
            failures.append(f"{experiment_id}: missing source split/cohort {source}")
            continue
        if sha256(source) != str(row["source_sha256"]):
            failures.append(f"{experiment_id}: source SHA-256 mismatch")
        if (
            row["artifact_kind"] == "support_query_split"
            and str(row["source_sha256"]) != str(row["split_sha256"])
        ):
            failures.append(f"{experiment_id}: split copy is not byte-identical")

    counts = manifest["artifact_kind"].value_counts().to_dict()
    print(
        f"split/cohort artifacts: {len(manifest)}; "
        f"support/query copies: {counts.get('support_query_split', 0)}; "
        f"frozen cohorts: {counts.get('frozen_evaluation_cohort', 0)}; "
        f"failures: {len(failures)}"
    )
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
