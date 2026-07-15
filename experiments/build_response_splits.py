#!/usr/bin/env python3
"""Rebuild the strict CRBN/VHL splits used by response Tables S3 and S9.

Sampling is class-balanced and without replacement: each eligible target task
contributes K positive and K negative support examples.  A fixed-Q setting then
uses Q examples per class; ``Q=0`` uses every remaining example.  The random
seed formula is intentionally frozen because it is part of the released
benchmark definition.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 2025, 3407)
ORIGINAL_SETTINGS = {
    ("CRBN", "VHL"): ((2, 3), (2, 5)),
    ("VHL", "CRBN"): ((2, 3), (2, 5)),
}
S9_SETTINGS = {
    ("CRBN", "VHL"): ((1, 1), (1, 0), (2, 3), (2, 5), (2, 0), (5, 0)),
    ("VHL", "CRBN"): (
        (1, 1),
        (1, 0),
        (2, 3),
        (2, 5),
        (2, 0),
        (5, 0),
        (1, 10),
        (2, 10),
        (5, 10),
    ),
}
BCR_ABL = re.compile(r"^BCR-ABL(?:$| )", flags=re.IGNORECASE)


def normalize_e3(value: Any) -> str:
    text = str(value).strip().upper()
    if "CRBN" in text:
        return "CRBN"
    if "VHL" in text:
        return "VHL"
    return text


def record(row: pd.Series) -> dict[str, Any]:
    compound_id = str(row["Compound ID"])
    e3 = normalize_e3(row["E3 ligase"])
    task = str(row["Target"]).strip()
    return {
        "Context ID": f"{compound_id}||{e3}||{task}",
        "Compound ID": compound_id,
        "Label_bin": int(row["Label_bin"]),
        "E3 ligase": e3,
        "Task": task,
    }


def sample_task(
    group: pd.DataFrame,
    rng: random.Random,
    k_shot: int,
    q_query: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    positive = group[group["Label_bin"].eq(1)]
    negative = group[group["Label_bin"].eq(0)]
    minimum = k_shot + (q_query if q_query else 1)
    if len(positive) < minimum or len(negative) < minimum:
        return None
    positive_indices = list(positive.index)
    negative_indices = list(negative.index)
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)
    support_indices = positive_indices[:k_shot] + negative_indices[:k_shot]
    if q_query:
        query_indices = (
            positive_indices[k_shot : k_shot + q_query]
            + negative_indices[k_shot : k_shot + q_query]
        )
    else:
        query_indices = positive_indices[k_shot:] + negative_indices[k_shot:]
    return (
        [record(group.loc[index]) for index in support_indices],
        [record(group.loc[index]) for index in query_indices],
    )


def build_one(
    contexts: pd.DataFrame,
    source: str,
    target: str,
    k_shot: int,
    q_query: int,
    seed: int,
    exclude_bcr_abl: bool,
) -> dict[str, Any]:
    source_frame = contexts[contexts["_E3"] == source]
    target_frame = contexts[contexts["_E3"] == target]
    excluded_mask = (
        target_frame["Target"].astype(str).str.contains(BCR_ABL, na=False)
        if exclude_bcr_abl
        else pd.Series(False, index=target_frame.index)
    )
    excluded_frame = target_frame[excluded_mask]
    rng = random.Random(seed + 1009 * k_shot + 9176 * q_query)
    episodes: dict[str, list[dict[str, Any]]] = {}
    query_rows = 0
    for task, group in target_frame.groupby("Target", sort=True):
        sampled = sample_task(group, rng, k_shot, q_query)
        if sampled is None:
            continue
        # Sampling happens before exclusion.  This preserves every historical
        # draw for tasks ordered after BCR-ABL.
        if exclude_bcr_abl and BCR_ABL.search(str(task)):
            continue
        support, query = sampled
        episodes[str(task)] = [{"support": support, "query": query}]
        query_rows += len(query)
    stem = (
        f"{source.lower()}_train_{target.lower()}_test_strict_"
        f"K{k_shot}Q{q_query}_seed{seed}"
    )
    return {
        "meta": {
            "split": stem,
            "direction": f"{source}->{target}",
            "source": source,
            "target": target,
            "k_shot": k_shot,
            "q_query": q_query,
            "query_setting": "all_remaining" if q_query == 0 else f"Q{q_query}",
            "seed": seed,
            "source_examples": int(len(source_frame)),
            "target_tasks_total": int(
                target_frame.loc[~excluded_mask, "Target"].nunique()
            ),
            "eligible_tasks": len(episodes),
            "query_rows_total": query_rows,
            "context_col": "context_key_compound_e3_target",
            "excluded_target_rows": int(len(excluded_frame)),
            "excluded_target_tasks": int(excluded_frame["Target"].nunique()),
            "exclude_target_task_regex": BCR_ABL.pattern if exclude_bcr_abl else "",
        },
        "meta_test": episodes,
    }


def write_profile(
    contexts: pd.DataFrame,
    output: Path,
    settings: dict[tuple[str, str], tuple[tuple[int, int], ...]],
    exclude_bcr_abl: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for (source, target), kq_settings in settings.items():
        for k_shot, q_query in kq_settings:
            for seed in SEEDS:
                split = build_one(
                    contexts,
                    source,
                    target,
                    k_shot,
                    q_query,
                    seed,
                    exclude_bcr_abl,
                )
                path = output / f"{split['meta']['split']}.json"
                path.write_text(json.dumps(split, indent=2) + "\n")
                try:
                    display_path = path.relative_to(ROOT)
                except ValueError:
                    display_path = path
                rows.append({"path": str(display_path), **split["meta"]})
    pd.DataFrame(rows).to_csv(output / "split_manifest.csv", index=False)
    print(f"wrote {len(rows)} splits to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        type=Path,
        default=ROOT / "data" / "strict_crbn_vhl_contexts.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "splits" / "response_letter",
    )
    parser.add_argument(
        "--profile",
        choices=("all", "original", "bcr_abl_excluded"),
        default="all",
    )
    args = parser.parse_args()
    contexts = pd.read_csv(args.contexts, low_memory=False)
    contexts["Compound ID"] = contexts["Compound ID"].astype(str)
    contexts["Label_bin"] = contexts["Label_bin"].astype(int)
    contexts["_E3"] = contexts["E3 ligase"].map(normalize_e3)
    if args.profile in {"all", "original"}:
        write_profile(
            contexts,
            args.output_root / "original",
            ORIGINAL_SETTINGS,
            exclude_bcr_abl=False,
        )
    if args.profile in {"all", "bcr_abl_excluded"}:
        write_profile(
            contexts,
            args.output_root / "bcr_abl_excluded",
            S9_SETTINGS,
            exclude_bcr_abl=True,
        )


if __name__ == "__main__":
    main()
