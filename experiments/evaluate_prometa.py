#!/usr/bin/env python3
"""Re-evaluate released ProMeta checkpoints on response-letter splits."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch

from prometa.data import graph_index_from_cache, read_contexts
from prometa.evaluation import evaluate_split, split_metrics, summarize_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "response_letter_experiments.csv"
SEED_PATTERN = re.compile(r"seed(?P<seed>\d+)")


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def seed_from_path(path: Path) -> int:
    match = SEED_PATTERN.search(str(path))
    if match is None:
        raise ValueError(f"Cannot infer seed from {path}")
    return int(match.group("seed"))


def aliased_graph_index(
    graph_index: dict,
    contexts: pd.DataFrame,
    graph_id_column: str,
) -> dict:
    if not graph_id_column or graph_id_column == "Compound ID":
        return graph_index
    expanded = dict(graph_index)
    for _, row in contexts[["Compound ID", graph_id_column]].dropna().iterrows():
        graph_id = str(row[graph_id_column])
        if graph_id in graph_index:
            expanded[str(row["Compound ID"])] = graph_index[graph_id]
    return expanded


def run_row(
    row: pd.Series,
    base_graph_index: dict,
    device: torch.device,
    output_root: Path,
    fixed_source_checkpoint_root: Path | None = None,
) -> None:
    contexts = read_contexts(resolve(str(row["contexts"])))
    graph_index = aliased_graph_index(
        base_graph_index,
        contexts,
        str(row.get("graph_id_column", "Compound ID")),
    )
    experiment_id = str(row["experiment_id"])
    split_root = ROOT / "splits" / "by_experiment" / experiment_id
    split_paths = sorted(split_root.glob(f"{experiment_id}_seed*.json"))
    if not split_paths:
        raise FileNotFoundError(f"No dedicated splits for {experiment_id}: {split_root}")

    prediction_frames = []
    for split_path in split_paths:
        seed = seed_from_path(split_path)
        if fixed_source_checkpoint_root is None:
            checkpoint = (
                ROOT
                / "checkpoints"
                / "by_experiment"
                / experiment_id
                / f"{experiment_id}_seed{seed}.pt"
            )
        else:
            checkpoint = (
                fixed_source_checkpoint_root
                / str(row["source"])
                / f"protonet_encoder_seed{seed}.pt"
            )
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        metadata = {
            "direction": row["direction"],
            "source": row["source"],
            "target": row["target"],
            "k_shot": int(row["k_shot"]),
            "q_query": int(row["q_query"]),
            "query_setting": row["query_setting"],
            "seed": seed,
        }
        predictions = evaluate_split(
            split_path=split_path,
            checkpoint=checkpoint,
            graph_index=graph_index,
            contexts=contexts,
            device=device,
            head=str(row["head"]),
            fingerprint_bits=int(row["fingerprint_bits"]),
            fingerprint_weight=float(row["fingerprint_weight"]),
            metadata_overrides=metadata,
        )
        predictions["experiment_id"] = row["experiment_id"]
        predictions["response_table"] = row["response_table"]
        predictions["checkpoint"] = str(checkpoint)
        prediction_frames.append(predictions)

    output_dir = output_root / str(row["experiment_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = split_metrics(predictions)
    summary = summarize_metrics(metrics)
    predictions.to_csv(output_dir / "query_predictions.csv", index=False)
    metrics.to_csv(output_dir / "split_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(f"[{row['experiment_id']}] {summary[['AUROC', 'AUPRC', 'BALACC']].to_dict('records')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "recomputed")
    parser.add_argument(
        "--fixed-source-checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Override each registry checkpoint with "
            "ROOT/<source>/protonet_encoder_seed<seed>.pt. This is used to "
            "evaluate one fixed source-trained encoder family across multiple "
            "target support/query configurations."
        ),
    )
    args = parser.parse_args()

    config = pd.read_csv(args.config).fillna("")
    config = config[config["runner"] == "checkpoint_eval"].copy()
    if args.experiment:
        config = config[config["experiment_id"].isin(args.experiment)].copy()
    if config.empty:
        raise ValueError("No checkpoint_eval experiments selected")

    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    graph_index = graph_index_from_cache(ROOT / "data" / "prometa_graphs.npz")
    for _, row in config.iterrows():
        run_row(
            row,
            graph_index,
            device,
            args.output_root,
            fixed_source_checkpoint_root=args.fixed_source_checkpoint_root,
        )


if __name__ == "__main__":
    main()
