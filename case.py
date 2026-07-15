#!/usr/bin/env python3
"""Reproduce the released 12-compound VZ185 case-study result.

Run from the repository root with::

    python case.py

The script loads the dedicated released VZ185 frozen-prediction checkpoint,
recomputes compound-level correctness, verifies the expected 11/12 result,
and writes both compound-level predictions and a one-row summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "by_experiment"
    / "vz185_legacy_excel_first12"
    / "vz185_legacy_excel_first12.pt"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "recomputed" / "vz185_case"


def load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "prometa_frozen_prediction_checkpoint_v1":
        raise ValueError(f"Unsupported VZ185 checkpoint format: {path}")
    if payload.get("response_table") != "VZ185":
        raise ValueError(f"Checkpoint is not the released VZ185 case study: {path}")
    return payload


def evaluate(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    compounds = pd.DataFrame(payload["predictions"]).copy()
    required = {"case_label", "legacy_excel_true_label", "legacy_excel_prometa_pred"}
    missing = required - set(compounds.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    compounds["y_true"] = compounds["legacy_excel_true_label"].astype(int)
    compounds["y_pred"] = compounds["legacy_excel_prometa_pred"].astype(int)
    compounds["correct"] = compounds["y_true"].eq(compounds["y_pred"])

    n_compounds = int(len(compounds))
    n_correct = int(compounds["correct"].sum())
    accuracy = float(compounds["correct"].mean())

    expected = payload.get("expected_summary", {})
    expected_n = int(expected.get("n", 12))
    expected_accuracy = float(expected.get("accuracy", 11 / 12))
    expected_correct = int(round(expected_n * expected_accuracy))
    if (n_compounds, n_correct) != (expected_n, expected_correct):
        raise AssertionError(
            "VZ185 reproduction failed: "
            f"obtained {n_correct}/{n_compounds}, "
            f"expected {expected_correct}/{expected_n}"
        )

    summary = pd.DataFrame(
        [
            {
                "experiment": "VZ185-derived PROTAC case study",
                "method": "ProMeta",
                "n_compounds": n_compounds,
                "n_correct": n_correct,
                "accuracy": accuracy,
                "accuracy_percent": 100.0 * accuracy,
            }
        ]
    )
    keep = ["case_label", "case_compound_id", "y_true", "y_pred", "correct"]
    return compounds[keep], summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the released ProMeta VZ185 11/12 case-study result."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    payload = load_checkpoint(args.checkpoint)
    compounds, summary = evaluate(payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compound_path = args.output_dir / "case_predictions.csv"
    summary_path = args.output_dir / "case_summary.csv"
    compounds.to_csv(compound_path, index=False)
    summary.to_csv(summary_path, index=False)

    row = summary.iloc[0]
    print("VZ185-derived PROTAC case study")
    print(f"ProMeta correct: {int(row.n_correct)}/{int(row.n_compounds)}")
    print(f"Accuracy: {row.accuracy_percent:.1f}%")
    print(f"Predictions: {compound_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
