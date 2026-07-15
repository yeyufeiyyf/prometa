#!/usr/bin/env python3
"""Recompute the ProMeta + RDKit-3D rows reported in response Table S8."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prometa.data import (
    graph_index_from_cache,
    graphs_for_examples,
    iter_episodes,
    read_contexts,
    read_split,
)
from prometa.evaluation import (
    encode_graphs,
    fingerprint_index,
    load_encoder,
    split_metrics,
    summarize_metrics,
)
from prometa.geometry import (
    descriptor_index,
    descriptor_matrix,
    learn_support_weights,
    predict,
)


ROOT = Path(__file__).resolve().parents[1]


def evaluate_one(
    split_path: Path,
    checkpoint: Path,
    contexts: pd.DataFrame,
    graph_index: dict,
    fingerprints: dict[str, np.ndarray],
    descriptors: dict[str, np.ndarray],
    device: torch.device,
) -> pd.DataFrame:
    split = read_split(split_path)
    meta = split["meta"]
    encoder = load_encoder(checkpoint, device)
    fallback_labels = dict(zip(contexts["Compound ID"], contexts["Label_bin"]))
    rows: list[dict[str, object]] = []
    for task, episode_index, support_examples, query_examples in iter_episodes(split):
        support, support_ids, support_labels = graphs_for_examples(
            support_examples, graph_index, fallback_labels
        )
        query, query_ids, query_labels = graphs_for_examples(
            query_examples, graph_index, fallback_labels
        )
        if not support or not query or set(support_labels.tolist()) != {0, 1}:
            continue
        support_embedding = encode_graphs(encoder, support, device)
        query_embedding = encode_graphs(encoder, query, device)
        support_fingerprint = np.stack([fingerprints[item] for item in support_ids])
        query_fingerprint = np.stack([fingerprints[item] for item in query_ids])
        support_descriptors = descriptor_matrix(descriptors, support_ids)
        query_descriptors = descriptor_matrix(descriptors, query_ids)
        learned = learn_support_weights(
            support_embedding,
            support_fingerprint,
            support_descriptors,
            support_labels,
            device,
        )
        probabilities = predict(
            support_embedding,
            support_fingerprint,
            support_descriptors,
            support_labels,
            query_embedding,
            query_fingerprint,
            query_descriptors,
            learned,
            device,
        )
        fp_weight, fp_prior, descriptor_weight, descriptor_prior = learned["weights"]
        for compound_id, y_true, probability in zip(
            query_ids, query_labels, probabilities
        ):
            rows.append(
                {
                    "split": split_path.stem,
                    "direction": meta["direction"],
                    "source": meta["source"],
                    "target": meta["target"],
                    "k_shot": int(meta["k_shot"]),
                    "q_query": int(meta["q_query"]),
                    "query_setting": meta["query_setting"],
                    "seed": int(meta["seed"]),
                    "method": "ProMeta + 3D descriptor fusion",
                    "task": task,
                    "episode": episode_index,
                    "compound_id": compound_id,
                    "y_true": int(y_true),
                    "prob": float(probability),
                    "learned_fp_weight": fp_weight,
                    "learned_fp_prior_weight": fp_prior,
                    "learned_3d_weight": descriptor_weight,
                    "learned_3d_prior_weight": descriptor_prior,
                    "learn_status": learned["status"],
                    "loo_loss_start": learned["loo_loss_start"],
                    "loo_loss_end": learned["loo_loss_end"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        type=Path,
        default=ROOT / "data" / "strict_crbn_vhl_contexts.csv",
    )
    parser.add_argument(
        "--graph-cache",
        type=Path,
        default=ROOT / "data" / "prometa_graphs.npz",
    )
    parser.add_argument(
        "--descriptor-cache",
        type=Path,
        default=ROOT / "data" / "rdkit_3d_descriptors.csv",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=ROOT / "splits" / "by_experiment",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=ROOT / "checkpoints" / "by_experiment",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "recomputed" / "s8_rdkit_3d",
    )
    args = parser.parse_args()
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    contexts = read_contexts(args.contexts)
    graph_index = graph_index_from_cache(args.graph_cache)
    fingerprints = fingerprint_index(contexts, bits=512)
    descriptors = descriptor_index(args.descriptor_cache)
    frames = []
    for source, target in (("CRBN", "VHL"), ("VHL", "CRBN")):
        for q_query in (3, 5):
            experiment_id = (
                f"s8_rdkit_3d_{source.lower()}_{target.lower()}_k2q{q_query}"
            )
            for seed in (42, 2025, 3407):
                split_path = (
                    args.splits
                    / experiment_id
                    / f"{experiment_id}_seed{seed}.json"
                )
                checkpoint = (
                    args.checkpoints
                    / experiment_id
                    / f"{experiment_id}_seed{seed}.pt"
                )
                if not split_path.is_file():
                    raise FileNotFoundError(split_path)
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                frames.append(
                    evaluate_one(
                        split_path,
                        checkpoint,
                        contexts,
                        graph_index,
                        fingerprints,
                        descriptors,
                        device,
                    )
                )
    predictions = pd.concat(frames, ignore_index=True)
    metrics = split_metrics(predictions)
    summary = summarize_metrics(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "query_predictions.csv", index=False)
    metrics.to_csv(args.output / "split_metrics.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    print(summary[["direction", "q_query", "AUROC", "AUPRC", "BALACC"]].to_string(index=False))


if __name__ == "__main__":
    main()
