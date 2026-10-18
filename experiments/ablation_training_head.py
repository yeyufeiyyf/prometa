#!/usr/bin/env python3
"""Run one ProMeta1-matched learning-mechanism ablation cell for ProMeta2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn

from prometa.ablations.experiment import (
    ExperimentData,
    binary_metrics,
    build_experiment_data,
    build_episode_specs,
    eligible_training_tasks,
    load_eval_episodes,
    model_forward,
    set_seed,
    sha256_file,
    split_source_validation_tasks,
    validation_episode_loss,
)
from prometa.ablations.data import ATOM_FEATURE_NAMES, BOND_FEATURE_NAMES
from prometa.ablations.model import E3V2Config, E3V2Model


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "prometa2",
    "prometa2_support_linear",
    "prometa2_no_episodic",
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_full_model(checkpoint: Path, protein_dim: int, device: torch.device) -> E3V2Model:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload["model_config"])
    config["pretrained_protein_dim"] = protein_dim
    model = E3V2Model(E3V2Config(**config))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def evaluate(
    model: E3V2Model,
    data: ExperimentData,
    split_path: Path,
    *,
    source_e3: str,
    target_e3: str,
    seed: int,
    variant: str,
    head: str,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta, episodes = load_eval_episodes(split_path, data)
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for episode_index, episode in enumerate(episodes):
            support, _ = model_forward(
                model, episode.support_ids, data, device=device,
                intervention="correct", source_e3=source_e3, seed=seed,
            )
            query, _ = model_forward(
                model, episode.query_ids, data, device=device,
                intervention="correct", source_e3=source_e3, seed=seed,
            )
            labels = np.asarray(episode.support_labels, dtype=int)
            if head == "prototype":
                scored = model.score_episode(
                    support,
                    torch.as_tensor(labels, dtype=torch.long, device=device),
                    query,
                )
                classes = scored["classes"].detach().cpu().tolist()
                scores = torch.softmax(scored["logits"], dim=1)[
                    :, classes.index(1)
                ].detach().cpu().numpy()
            elif head == "support_linear":
                classifier = LogisticRegression(
                    penalty="l2", C=1.0, solver="liblinear",
                    class_weight="balanced", max_iter=1000, random_state=0,
                )
                classifier.fit(support["embedding"].detach().cpu().numpy(), labels)
                scores = classifier.predict_proba(
                    query["embedding"].detach().cpu().numpy()
                )[:, 1]
            else:
                raise ValueError(head)
            for row_id, label, score in zip(
                episode.query_ids, episode.query_labels, scores, strict=True
            ):
                row = data.row_lookup.loc[row_id]
                rows.append(
                    {
                        "variant": variant,
                        "direction": f"{source_e3}->{target_e3}",
                        "source_e3": source_e3,
                        "target_e3": target_e3,
                        "seed": seed,
                        "k_shot": int(meta["k_shot"]),
                        "q_query": int(meta["q_query"]),
                        "protocol_version": meta["protocol_version"],
                        "task": episode.task,
                        "episode": episode_index,
                        "row_id": row_id,
                        "compound_id": str(row["Compound ID"]),
                        "y_true": int(label),
                        "y_score": float(score),
                    }
                )
    frame = pd.DataFrame(rows)
    return frame, {**frame.iloc[0][[
        "variant", "direction", "source_e3", "target_e3", "seed",
        "k_shot", "q_query", "protocol_version",
    ]].to_dict(), **binary_metrics(frame)}


def unique_rows(tasks: dict[str, pd.DataFrame]) -> list[str]:
    return sorted(
        set().union(*(set(frame["row_id"].astype(str)) for frame in tasks.values()))
    )


def train_non_episodic(
    data: ExperimentData,
    *,
    source_e3: str,
    seed: int,
    config: dict[str, Any],
    comparison_epochs: int,
    device: torch.device,
) -> tuple[E3V2Model, dict[str, Any], nn.Linear]:
    set_seed(seed)
    model_config = E3V2Config(
        variant="grounded_interaction",
        node_dim=len(ATOM_FEATURE_NAMES),
        edge_dim=len(BOND_FEATURE_NAMES),
        pretrained_protein_dim=data.protein_dim,
        distance="squared_euclidean",
        temperature=1.0,
        learnable_temperature=True,
    )
    model = E3V2Model(model_config).to(device)
    classifier = nn.Linear(model_config.embedding_dim, 2).to(device)
    tasks = eligible_training_tasks(
        data.training_contexts, source_e3=source_e3,
        k_shot=int(config["k_shot"]), q_query=int(config["train_q"]),
    )
    train_tasks, validation_tasks, selection = split_source_validation_tasks(
        tasks, source_e3=source_e3, seed=seed,
        fraction=float(config["source_validation_fraction"]),
    )
    train_ids = unique_rows(train_tasks)
    validation_ids = unique_rows(validation_tasks)
    validation_specs = []
    validation_episode_hash = None
    if validation_tasks:
        validation_count = max(len(validation_tasks), len(validation_tasks) * 2)
        validation_specs, validation_episode_hash = build_episode_specs(
            validation_tasks,
            count=validation_count,
            seed=seed + 104729,
            k_shot=int(config["k_shot"]),
            q_query=int(config["train_q"]),
        )
    labels = data.row_lookup.loc[train_ids, "Label_bin"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = torch.as_tensor(
        [len(labels) / max(2.0 * count, 1.0) for count in counts],
        dtype=torch.float32, device=device,
    )
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *classifier.parameters()],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = None
    best_model = None
    best_classifier = None
    stale = 0
    history = []
    batch_size = int(config["supervised_batch_size"])
    examples_per_episode = 2 * (int(config["k_shot"]) + int(config["train_q"]))
    examples_per_epoch = int(config["episodic_episodes_per_epoch"]) * examples_per_episode
    total_example_exposures = 0
    total_optimizer_updates = 0
    for epoch in range(int(comparison_epochs)):
        model.train()
        classifier.train()
        batch_losses = []
        epoch_indices = []
        while len(epoch_indices) < examples_per_epoch:
            epoch_indices.extend(generator.permutation(len(train_ids)).tolist())
        epoch_indices = np.asarray(epoch_indices[:examples_per_epoch], dtype=int)
        for start in range(0, len(epoch_indices), batch_size):
            indices = epoch_indices[start : start + batch_size]
            ids = [train_ids[index] for index in indices]
            output, _ = model_forward(
                model, ids, data, device=device, intervention="correct",
                source_e3=source_e3, seed=seed,
            )
            y = torch.as_tensor(labels[indices], dtype=torch.long, device=device)
            loss = nn.functional.cross_entropy(
                classifier(output["embedding"]), y, weight=weights
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*model.parameters(), *classifier.parameters()], 5.0
            )
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            total_example_exposures += len(ids)
            total_optimizer_updates += 1
        validation_loss = validation_episode_loss(
            model, data, validation_specs, device=device,
            source_e3=source_e3, seed=seed,
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(batch_losses)),
            "source_validation_loss": validation_loss,
        })
        if np.isfinite(validation_loss) and validation_loss < best_loss - float(config["validation_min_delta"]):
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_classifier = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
            stale = 0
        else:
            stale += 1
    if best_model is not None:
        model.load_state_dict(best_model)
        classifier.load_state_dict(best_classifier)
    return model.eval(), {
        "training_mode": "ordinary_supervised_individual_examples",
        "inference_head": "prototype",
        "train_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "class_counts": counts.astype(int).tolist(),
        "training_budget_basis": config["training_budget_basis"],
        "matched_episodic_epochs": int(config["episodic_train_epochs"]),
        "matched_full_epochs_completed": int(comparison_epochs),
        "matched_episodic_episodes_per_epoch": int(config["episodic_episodes_per_epoch"]),
        "examples_per_episodic_step": examples_per_episode,
        "examples_per_supervised_epoch": examples_per_epoch,
        "total_source_example_exposures": total_example_exposures,
        "total_optimizer_updates": total_optimizer_updates,
        "best_epoch": best_epoch,
        "best_source_validation_loss": None if best_epoch is None else best_loss,
        "epochs_completed": len(history),
        "selection": selection,
        "selection_objective": "source_only_prototype_episode_loss",
        "validation_episode_sha256": validation_episode_hash,
        "history": history,
    }, classifier.eval()


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-e3", required=True)
    parser.add_argument("--target-e3", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--head-only", action="store_true", help="Compare prototype/support-linear heads without launching supervised retraining")
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    ROOT = (config_path.parent / config.get("path_base", "..")).resolve()
    source, target, seed = args.source_e3, args.target_e3, args.seed
    direction_dir = f"{source}_to_{target}"
    output = resolve(config["outputs"]["results"]) / direction_dir / f"seed_{seed}"
    success = output / "_SUCCESS"
    if success.exists() and not args.force:
        print(success)
        return
    output.mkdir(parents=True, exist_ok=True)
    data_paths = {key: resolve(value) for key, value in config["data"].items()}
    data = build_experiment_data(
        contexts_path=data_paths["contexts"], sequence_path=data_paths["sequence_table"],
        component_path=data_paths["component_table"],
        protein_embeddings_path=data_paths["protein_embeddings"],
        source_e3=source, target_e3=target, supervised_train_scope="source_e3_only",
    )
    split_path = data_paths["split_root"] / direction_dir.lower() / f"seed_{seed}" / "k2_q3.json"
    full_checkpoint = (
        resolve(config["full_checkpoints"]) / direction_dir / f"seed_{seed}" /
        "grounded_interaction.pt"
    )
    full_model = load_full_model(full_checkpoint, data.protein_dim, device)
    prediction_frames = []
    metric_rows = []
    for variant, head in (("prometa2", "prototype"), ("prometa2_support_linear", "support_linear")):
        predictions, metrics = evaluate(
            full_model, data, split_path, source_e3=source, target_e3=target,
            seed=seed, variant=variant, head=head, device=device,
        )
        prediction_frames.append(predictions)
        metric_rows.append(metrics)
    # The public control evaluates both heads on the configured frozen split.
    # Historical figure summaries are not substituted as per-query references.
    if args.head_only:
        pd.concat(prediction_frames, ignore_index=True).to_csv(output / "query_predictions.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
        print(f"Compared prototype and support-linear heads: {output}")
        return
    full_manifest_path = (
        resolve(config["full_results"]) / direction_dir / f"seed_{seed}" /
        "run_manifest.json"
    )
    full_manifest = json.loads(full_manifest_path.read_text())
    non_episodic, training_audit, supervised_head = train_non_episodic(
        data, source_e3=source, seed=seed, config=config,
        comparison_epochs=int(full_manifest["training"]["epochs_completed"]),
        device=device,
    )
    full_selection = full_manifest["source_validation_selection"]
    control_selection = training_audit["selection"]
    for field in ("train_row_sha256", "validation_row_sha256", "train_tasks", "validation_tasks"):
        if control_selection.get(field) != full_selection.get(field):
            raise RuntimeError(f"Full/control source split mismatch for {field}")
    if training_audit["validation_episode_sha256"] != full_manifest["source_validation_episode_sha256"]:
        raise RuntimeError("Full/control validation episodes do not match")
    expected_exposures = (
        int(full_manifest["training"]["epochs_completed"])
        * int(config["episodic_episodes_per_epoch"])
        * 2 * (int(config["k_shot"]) + int(config["train_q"]))
    )
    if training_audit["total_source_example_exposures"] != expected_exposures:
        raise RuntimeError("Full/control source-example exposure budgets do not match")
    predictions, metrics = evaluate(
        non_episodic, data, split_path, source_e3=source, target_e3=target,
        seed=seed, variant="prometa2_no_episodic", head="prototype", device=device,
    )
    prediction_frames.append(predictions)
    metric_rows.append(metrics)
    checkpoint = resolve(config["outputs"]["checkpoints"]) / direction_dir / f"seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": non_episodic.state_dict(),
        "supervised_head_state_dict": supervised_head.state_dict(),
        "training_audit": training_audit,
        "source_e3": source, "target_e3": target, "seed": seed,
    }, checkpoint)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_metrics = pd.DataFrame(metric_rows)
    all_predictions.to_csv(output / "query_predictions.csv", index=False)
    all_metrics.to_csv(output / "split_metrics.csv", index=False)
    manifest = {
        "completed": True,
        "source_e3": source, "target_e3": target, "seed": seed,
        "variants": list(VARIANTS),
        "config_sha256": sha256_file(config_path),
        "input_sha256": {key: sha256_file(path) if path.is_file() else None for key, path in data_paths.items()},
        "split": str(split_path), "split_sha256": sha256_file(split_path),
        "full_checkpoint": str(full_checkpoint),
        "full_checkpoint_sha256": sha256_file(full_checkpoint),
        "full_prediction_max_abs_delta": max_delta,
        "full_manifest": str(full_manifest_path),
        "full_manifest_sha256": sha256_file(full_manifest_path),
        "full_source_validation_selection": full_manifest["source_validation_selection"],
        "full_source_validation_episode_sha256": full_manifest["source_validation_episode_sha256"],
        "matched_source_example_exposures": expected_exposures,
        "data_audit": data.audit,
        "non_episodic_training": training_audit,
        "code_sha256": {
            "runner": sha256_file(Path(__file__)),
            "model": sha256_file(ROOT / "e3_v2" / "model.py"),
            "experiment": sha256_file(ROOT / "e3_v2" / "experiment.py"),
            "data": sha256_file(ROOT / "e3_v2" / "data.py")
        },
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_sha256": sha256_file(output / "query_predictions.csv"),
        "metrics_sha256": sha256_file(output / "split_metrics.csv"),
        "elapsed_seconds": time.time() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    success.write_text(stable_hash(manifest) + "\n")
    print(json.dumps({"output": str(output), "metrics": metric_rows}))


if __name__ == "__main__":
    main()
