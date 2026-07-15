#!/usr/bin/env python3
"""Freeze the rare-E3 common-query cohort as directly evaluable split files.

The response letter's rare-E3 table is evaluated on a 90-row cohort defined by
input/label coverage shared during the original comparison.  External-model
outputs are intentionally not released; ``common_query_cohort.csv`` is the
minimal frozen eligibility list needed to reproduce ProMeta on that cohort.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def compound_id(example: Any) -> str:
    if isinstance(example, dict):
        return str(example.get("Compound ID", example.get("compound_id", "")))
    return str(example)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "splits" / "response_letter" / "rare_e3",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=(
            ROOT
            / "splits"
            / "response_letter"
            / "rare_e3_common_query"
            / "common_query_cohort.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "splits" / "response_letter" / "rare_e3_common_query",
    )
    args = parser.parse_args()
    cohort = pd.read_csv(args.cohort)
    args.output.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    for source_path in sorted(args.source.glob("*.json")):
        split = json.loads(source_path.read_text())
        seed = int(split.get("meta", {}).get("seed", source_path.stem.rsplit("seed", 1)[1]))
        selected = cohort[cohort["seed"].eq(seed)]
        retained = 0
        for task, episodes in split.get("meta_test", {}).items():
            for episode_index, episode in enumerate(episodes):
                expected = selected[
                    selected["task"].astype(str).eq(str(task))
                    & selected["episode"].astype(int).eq(episode_index)
                ]
                counts = Counter(
                    str(row.compound_id) for row in expected.itertuples(index=False)
                )
                query = []
                for example in episode.get("query", []):
                    key = compound_id(example)
                    if counts[key] > 0:
                        query.append(example)
                        counts[key] -= 1
                missing = sum(counts.values())
                if missing:
                    raise ValueError(
                        f"{source_path.name} {task} episode {episode_index}: "
                        f"{missing} cohort rows were not found"
                    )
                episode["query"] = query
                retained += len(query)
                audit_rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "episode": episode_index,
                        "support_rows": len(episode.get("support", [])),
                        "query_rows": len(query),
                    }
                )
        split.setdefault("meta", {})["query_rows_total"] = retained
        split["meta"]["query_cohort"] = "rare-E3 common-query label-agree"
        output_path = args.output / source_path.name
        output_path.write_text(json.dumps(split, indent=2) + "\n")
    pd.DataFrame(audit_rows).to_csv(args.output / "split_manifest.csv", index=False)
    print(f"wrote {len(audit_rows)} episodes and {sum(r['query_rows'] for r in audit_rows)} query rows")


if __name__ == "__main__":
    main()
