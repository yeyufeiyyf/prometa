#!/usr/bin/env python3
"""Summarize the ProMeta rare-E3 common-query predictions.

The response letter pools all episode-level common-query rows for the overall
and per-ligase values in Supplementary Table S4.  This is intentionally
different from averaging one metric value per random seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prometa.evaluation import metric_values


ROOT = Path(__file__).resolve().parents[1]


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    if "prob" not in frame.columns and "score" in frame.columns:
        frame = frame.rename(columns={"score": "prob"})
    required = {"task", "y_true", "prob"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    groups = [("Overall", frame)] + [
        (str(ligase), subset) for ligase, subset in frame.groupby("task", sort=True)
    ]
    for ligase, subset in groups:
        metrics = metric_values(subset)
        rows.append(
            {
                "scope": "common_query_label_agree",
                "ligase": ligase,
                "method": "ProMeta",
                "AUROC": metrics["AUROC"],
                "AUPRC": metrics["AUPRC"],
                "BALACC": metrics["BALACC"],
                "Accuracy": metrics["Accuracy"],
                "n": int(len(subset)),
                "pos": int(subset["y_true"].astype(int).sum()),
                "neg": int(len(subset) - subset["y_true"].astype(int).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "results" / "recomputed" / "s4_s5_rare_e3" / "query_predictions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "recomputed" / "s4_s5_rare_e3" / "pooled_metrics.csv",
    )
    args = parser.parse_args()

    metrics = summarize(pd.read_csv(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
