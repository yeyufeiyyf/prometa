#!/usr/bin/env python3
"""Build deterministic source-only train/validation Target partitions.

The held-out target E3 is never read when constructing these partitions.  The
output makes the manuscript's 80/20 source-Target development split auditable
without changing any retained checkpoint or reported numerical result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 2025, 3407)
SOURCE_E3S = ("CRBN", "VHL")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_record(row: pd.Series) -> dict[str, Any]:
    return {
        "Compound ID": str(row["Compound ID"]),
        "Label_bin": int(row["Label_bin"]),
        "Target": str(row["Target"]),
        "E3 ligase": str(row["E3 ligase"]),
        "context_key_compound_e3_target": str(
            row["context_key_compound_e3_target"]
        ),
    }


def target_statistics(
    contexts: pd.DataFrame,
    source_e3: str,
    required_per_class: int,
) -> pd.DataFrame:
    source = contexts[
        contexts["E3 ligase"].astype(str).str.strip().eq(source_e3)
        & contexts["Target"].notna()
    ].copy()
    rows: list[dict[str, Any]] = []
    for target, group in source.groupby("Target", sort=True):
        positive = int(group["Label_bin"].eq(1).sum())
        negative = int(group["Label_bin"].eq(0).sum())
        rows.append(
            {
                "source_e3": source_e3,
                "target": str(target),
                "rows": int(len(group)),
                "positive": positive,
                "negative": negative,
                "eligible_validation_episode": bool(
                    positive >= required_per_class
                    and negative >= required_per_class
                ),
            }
        )
    return pd.DataFrame(rows)


def partition_targets(
    statistics: pd.DataFrame,
    seed: int,
    validation_fraction: float,
) -> pd.DataFrame:
    """Split Targets while retaining eligible tasks on both sides when possible."""

    pieces: list[pd.DataFrame] = []
    for eligible, group in statistics.groupby(
        "eligible_validation_episode", sort=True
    ):
        targets = sorted(group["target"].astype(str))
        rng = random.Random(int(seed) + (100_003 if bool(eligible) else 200_003))
        rng.shuffle(targets)
        n_validation = max(1, int(math.ceil(validation_fraction * len(targets))))
        if len(targets) > 1:
            n_validation = min(n_validation, len(targets) - 1)
        validation = set(targets[:n_validation])
        part = group.copy()
        part["partition"] = part["target"].map(
            lambda target: "source_validation"
            if str(target) in validation
            else "source_train"
        )
        pieces.append(part)
    result = pd.concat(pieces, ignore_index=True)
    result["seed"] = int(seed)
    return result[
        [
            "source_e3",
            "seed",
            "target",
            "partition",
            "rows",
            "positive",
            "negative",
            "eligible_validation_episode",
        ]
    ].sort_values(["source_e3", "seed", "partition", "target"], kind="stable")


def sample_validation_episode(
    group: pd.DataFrame,
    seed: int,
    k_shot: int,
    q_query: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(int(seed))
    support: list[dict[str, Any]] = []
    query: list[dict[str, Any]] = []
    for label in (1, 0):
        candidates = list(group[group["Label_bin"].eq(label)].index)
        rng.shuffle(candidates)
        selected = candidates[: k_shot + q_query]
        support.extend(row_record(group.loc[index]) for index in selected[:k_shot])
        query.extend(row_record(group.loc[index]) for index in selected[k_shot:])
    return {"support": support, "query": query}


def build_validation_split(
    contexts: pd.DataFrame,
    membership: pd.DataFrame,
    source_e3: str,
    seed: int,
    validation_fraction: float,
    k_shot: int,
    q_query: int,
) -> dict[str, Any]:
    train_targets = sorted(
        membership.loc[membership["partition"].eq("source_train"), "target"].astype(str)
    )
    validation_targets = sorted(
        membership.loc[
            membership["partition"].eq("source_validation"), "target"
        ].astype(str)
    )
    eligible_validation = sorted(
        membership.loc[
            membership["partition"].eq("source_validation")
            & membership["eligible_validation_episode"].astype(bool),
            "target",
        ].astype(str)
    )
    source = contexts[
        contexts["E3 ligase"].astype(str).str.strip().eq(source_e3)
        & contexts["Target"].notna()
    ].copy()
    episodes: dict[str, list[dict[str, Any]]] = {}
    for position, target in enumerate(eligible_validation):
        group = source[source["Target"].astype(str).eq(target)]
        episode = sample_validation_episode(
            group,
            seed=int(seed) + 10_007 * (position + 1),
            k_shot=k_shot,
            q_query=q_query,
        )
        episodes[target] = [episode]
    return {
        "meta": {
            "role": "source_only_model_selection",
            "source_e3": source_e3,
            "seed": int(seed),
            "validation_fraction": float(validation_fraction),
            "partition_unit": "Target",
            "k_shot": int(k_shot),
            "q_query": int(q_query),
            "source_train_targets": train_targets,
            "source_validation_targets": validation_targets,
            "eligible_validation_targets": eligible_validation,
            "held_out_target_e3_access": "none",
        },
        "meta_valid": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        type=Path,
        default=ROOT / "data" / "strict_crbn_vhl_contexts.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "splits" / "source_development",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--train-q", type=int, default=3)
    args = parser.parse_args()

    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between zero and one")
    contexts = pd.read_csv(args.contexts, low_memory=False)
    contexts["Compound ID"] = contexts["Compound ID"].astype(str)
    contexts["Label_bin"] = contexts["Label_bin"].astype(int)
    args.output.mkdir(parents=True, exist_ok=True)

    all_membership: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    required_per_class = int(args.k) + int(args.train_q)
    for source_e3 in SOURCE_E3S:
        statistics = target_statistics(contexts, source_e3, required_per_class)
        for seed in SEEDS:
            membership = partition_targets(
                statistics, seed, args.validation_fraction
            )
            all_membership.append(membership)
            split = build_validation_split(
                contexts,
                membership,
                source_e3,
                seed,
                args.validation_fraction,
                args.k,
                args.train_q,
            )
            split_path = args.output / f"{source_e3.lower()}_source_development_seed{seed}.json"
            split_path.write_text(json.dumps(split, indent=2) + "\n")
            manifest_rows.append(
                {
                    "source_e3": source_e3,
                    "seed": seed,
                    "split": split_path.relative_to(ROOT).as_posix(),
                    "split_sha256": sha256(split_path),
                    "source_train_targets": int(
                        membership["partition"].eq("source_train").sum()
                    ),
                    "source_validation_targets": int(
                        membership["partition"].eq("source_validation").sum()
                    ),
                    "eligible_validation_targets": len(split["meta_valid"]),
                    "held_out_target_e3_access": "none",
                }
            )

    membership_path = args.output / "source_target_partitions.csv"
    pd.concat(all_membership, ignore_index=True).to_csv(membership_path, index=False)
    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(
        f"wrote {len(manifest_rows)} source-only development splits to {args.output}"
    )


if __name__ == "__main__":
    main()
