#!/usr/bin/env python3
"""Evaluate the historical frozen-prediction checkpoints (S7, R9, VZ185)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from prometa.evaluation import split_metrics, summarize_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "experiment_checkpoint_manifest.csv"


def load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "prometa_frozen_prediction_checkpoint_v1":
        raise ValueError(f"Not a frozen ProMeta prediction checkpoint: {path}")
    return payload


def s7_summary(experiment_id: str, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = pd.DataFrame(
        [record for payload in payloads for record in payload["predictions"]]
    )
    metrics = split_metrics(predictions)
    summary = summarize_metrics(metrics).iloc[0].to_dict()
    expected = payloads[0].get("expected_summary", {})
    for metric in ("AUROC", "AUPRC", "BALACC"):
        key = f"{metric}_mean"
        if key in expected and abs(float(summary[key]) - float(expected[key])) > 1e-12:
            raise AssertionError(
                f"{experiment_id} {key}: recomputed={summary[key]} expected={expected[key]}"
            )
    return {"experiment_id": experiment_id, "response_table": "S7", **summary}


def case_summary(experiment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame(payload["predictions"])
    expected = payload.get("expected_summary", {})
    if payload["response_table"] == "VZ185":
        correct = frame["legacy_excel_prometa_pred_correct_vs_legacy_label"].astype(bool)
        result = {
            "experiment_id": experiment_id,
            "response_table": "VZ185",
            "variant": payload["variant"],
            "n_rows": len(frame),
            "n_correct": int(correct.sum()),
            "accuracy": float(correct.mean()),
        }
    else:
        labels = frame["y_true"].astype(int).to_numpy()
        probabilities = frame["main_checkpoint_probability"].astype(float).to_numpy()
        predictions = frame["main_checkpoint_prediction"].astype(int).to_numpy()
        result = {
            "experiment_id": experiment_id,
            "response_table": "R9",
            "variant": payload["variant"],
            "n_rows": len(frame),
            "n_correct": int((predictions == labels).sum()),
            "accuracy": float((predictions == labels).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "auroc": float(roc_auc_score(labels, probabilities)),
            "auprc": float(average_precision_score(labels, probabilities)),
        }
    for key in ("n_correct", "accuracy", "balanced_accuracy", "auroc", "auprc"):
        if key in expected and key in result and abs(float(result[key]) - float(expected[key])) > 1e-12:
            raise AssertionError(
                f"{experiment_id} {key}: recomputed={result[key]} expected={expected[key]}"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "recomputed" / "frozen_checkpoint_summary.csv",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest).fillna("")
    manifest = manifest[manifest["artifact_kind"] == "frozen_prediction_checkpoint"]
    summaries: list[dict[str, Any]] = []
    for experiment_id, entries in manifest.groupby("experiment_id", sort=True):
        payloads = [load_payload(ROOT / path) for path in entries["checkpoint"]]
        table = str(entries.iloc[0]["response_table"])
        if table == "S7":
            summaries.append(s7_summary(experiment_id, payloads))
        else:
            if len(payloads) != 1:
                raise ValueError(f"Expected one frozen checkpoint for {experiment_id}")
            summaries.append(case_summary(experiment_id, payloads[0]))

    output = pd.DataFrame(summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))
    print(f"Verified {len(output)} frozen-prediction experiments: {args.output}")


if __name__ == "__main__":
    main()
