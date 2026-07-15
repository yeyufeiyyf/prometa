#!/usr/bin/env python3
"""Verify every dedicated experiment checkpoint and its source provenance."""

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
        default=ROOT / "experiments" / "experiment_checkpoint_manifest.csv",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest).fillna("")
    failures: list[str] = []
    if manifest["checkpoint"].duplicated().any():
        failures.append("manifest contains duplicate checkpoint paths")
    for _, row in manifest.iterrows():
        experiment_id = str(row["experiment_id"])
        checkpoint = ROOT / str(row["checkpoint"])
        source = ROOT / str(row["source_checkpoint"])
        expected_prefix = Path("checkpoints") / "by_experiment" / experiment_id
        try:
            relative_checkpoint = checkpoint.relative_to(ROOT)
        except ValueError:
            relative_checkpoint = checkpoint
        if expected_prefix not in relative_checkpoint.parents:
            failures.append(f"{experiment_id}: checkpoint is outside its experiment directory")
        if not checkpoint.is_file():
            failures.append(f"{experiment_id}: missing checkpoint {checkpoint}")
            continue
        if sha256(checkpoint) != str(row["checkpoint_sha256"]):
            failures.append(f"{experiment_id}: checkpoint SHA-256 mismatch")
        if not checkpoint.with_suffix(".json").is_file():
            failures.append(f"{experiment_id}: missing JSON sidecar")
        if not source.is_file():
            failures.append(f"{experiment_id}: missing source artifact {source}")
            continue
        if sha256(source) != str(row["source_sha256"]):
            failures.append(f"{experiment_id}: source SHA-256 mismatch")
        if (
            row["artifact_kind"] == "encoder_state_dict"
            and str(row["source_sha256"]) != str(row["checkpoint_sha256"])
        ):
            failures.append(f"{experiment_id}: encoder copy is not byte-identical")

    counts = manifest["artifact_kind"].value_counts().to_dict()
    print(
        f"checkpoint artifacts: {len(manifest)}; "
        f"encoder copies: {counts.get('encoder_state_dict', 0)}; "
        f"frozen prediction checkpoints: "
        f"{counts.get('frozen_prediction_checkpoint', 0)}; failures: {len(failures)}"
    )
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
