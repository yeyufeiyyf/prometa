#!/usr/bin/env python3
"""Source-reference zero-shot evaluation on the matched K5 target queries.

K=0 means no labeled target-E3 support. Class prototypes use five
deterministically chosen source-E3 training rows per class and training task.
Source validation tasks and all target-E3 labels are excluded from references.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import os

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


ROOT = Path(os.environ["PROMETA_STUDY_WORKSPACE"]).resolve() / "k1_to_k7_20261007"
YYF = ROOT.parent
PROJECT = YYF / "prometa_as"
PRIOR = YYF / "table_s4_prometa_rerun"

from prometa.data import context_key, stable_context_id  # noqa: E402
from prometa.experiment import (  # noqa: E402
    build_experiment_data, eligible_training_tasks, model_forward,
    split_source_validation_tasks, stable_int,
)
from prometa.model import E3V2Config, E3V2Model  # noqa: E402


DIRECTIONS = (("CRBN", "VHL"), ("VHL", "CRBN"))
SEEDS = (42, 2025, 3407)
REFERENCE_PER_TASK_CLASS = 5
KEY = ["task", "row_id"]


def one_run(source: str, target: str, seed: int, paths: dict,
            config: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    data = build_experiment_data(
        contexts_path=paths["contexts"], sequence_path=paths["sequence_table"],
        component_path=paths["component_table"],
        protein_embeddings_path=paths["protein_embeddings"],
        source_e3=source, target_e3=target,
        supervised_train_scope="source_e3_only")
    eligible = eligible_training_tasks(
        data.training_contexts, source_e3=source, k_shot=2, q_query=3)
    tasks, validation, selection = split_source_validation_tasks(
        eligible, source_e3=source, seed=seed,
        fraction=float(config["source_validation_fraction"]))
    manifest = json.loads((PRIOR / "runs" / f"{source}_to_{target}" /
                           f"seed_{seed}" / "train_k2q3/run_manifest.json").read_text())
    assert {name: len(frame) for name, frame in tasks.items()} == manifest["training_tasks"]
    assert {name: len(frame) for name, frame in validation.items()} == manifest["source_validation_tasks"]
    assert selection["train_row_sha256"] == manifest["source_validation_selection"]["train_row_sha256"]
    assert not set(tasks) & set(validation)

    checkpoint = (Path(os.environ.get("PROMETA_ARTIFACTS_ROOT", Path(__file__).resolve().parents[1] / "artifacts")).resolve() / "study_checkpoints" / "kcurve" /
                  f"{source}_to_{target}" / f"seed_{seed}" / "model.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = E3V2Model(E3V2Config(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    references = []
    for name, task_frame in sorted(tasks.items()):
        for label in (1, 0):
            candidates = task_frame.loc[task_frame.Label_bin.eq(label)].copy()
            assert len(candidates) >= REFERENCE_PER_TASK_CLASS
            candidates["rank"] = candidates.row_id.map(
                lambda rid: stable_int("k0_source_reference", seed, name, label, rid))
            selected = candidates.sort_values(["rank", "row_id"]).head(
                REFERENCE_PER_TASK_CLASS)
            for _, row in selected.iterrows():
                references.append({
                    "direction": f"{source}->{target}", "seed": seed,
                    "source_task": name, "label": label,
                    "row_id": str(row.row_id),
                    "compound_id": str(row["Compound ID"]),
                })
    ref = pd.DataFrame(references)
    assert len(ref) == len(tasks) * 2 * REFERENCE_PER_TASK_CLASS
    assert len(ref) == ref.row_id.nunique()
    assert set(data.row_lookup.loc[ref.row_id, "canonical_e3"]) == {source}
    assert set(ref.label) == {0, 1}
    support_ids = ref.row_id.tolist()
    support_labels = torch.tensor(ref.label.tolist(), dtype=torch.long)

    split_path = (PRIOR / "splits" / f"{source.lower()}_to_{target.lower()}" /
                  f"seed_{seed}" / "k5_q0.json")
    split = json.loads(split_path.read_text())
    assert split["meta"]["q_query"] == 0
    expected_k5 = pd.read_csv(
        ROOT / "evaluation" / f"{source}_to_{target}" /
        f"seed_{seed}" / "query_predictions.csv", dtype={"row_id": str})
    expected_k5 = expected_k5.loc[expected_k5.k_shot.eq(5)]
    assert len(expected_k5) == split["meta"]["query_rows_total"]

    predictions = []
    with torch.no_grad():
        support, _ = model_forward(
            model, support_ids, data, device=torch.device("cpu"),
            intervention="correct", source_e3=source, seed=seed)
        for task, episodes in sorted(split["meta_test"].items()):
            assert len(episodes) == 1
            rows = episodes[0]["query"]
            query_ids = [stable_context_id(context_key(
                row["Compound ID"], task, target)) for row in rows]
            query, _ = model_forward(
                model, query_ids, data, device=torch.device("cpu"),
                intervention="correct", source_e3=source, seed=seed)
            scored = model.score_episode(support, support_labels, query)
            classes = scored["classes"].tolist()
            positive_index = classes.index(1)
            scores = torch.softmax(scored["logits"], dim=1)[:, positive_index].tolist()
            for row, rid, score in zip(rows, query_ids, scores):
                predictions.append({
                    "direction": f"{source}->{target}", "seed": seed,
                    "k": 0, "task": task + "#0", "row_id": rid,
                    "compound_id": str(row["Compound ID"]),
                    "y_true": int(row["Label_bin"]), "y_score": score,
                })
    pred = pd.DataFrame(predictions)
    assert pred.y_score.between(0, 1).all()
    paired = pred.merge(expected_k5[KEY + ["y_true"]], on=KEY,
                        validate="one_to_one", suffixes=("", "_k5"))
    assert len(paired) == len(pred) == len(expected_k5)
    assert paired.y_true.eq(paired.y_true_k5).all()
    truth, score = pred.y_true, pred.y_score
    result = {
        "direction": f"{source}->{target}", "seed": seed, "k": 0,
        "tasks": pred.task.nunique(), "query_rows": len(pred),
        "positive_queries": int(truth.sum()),
        "negative_queries": int(len(pred) - truth.sum()),
        "source_training_tasks": len(tasks),
        "source_reference_rows": len(ref),
        "auroc": roc_auc_score(truth, score),
        "auprc": average_precision_score(truth, score),
        "balacc": balanced_accuracy_score(truth, score >= 0.5),
    }
    audit = {
        "direction": f"{source}->{target}", "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "target_E3_support_rows": 0,
        "source_validation_tasks_excluded": sorted(validation),
        "source_training_tasks": sorted(tasks),
        "source_reference_rule": "stable_int(seed, task, label, row_id); first 5 per task/class",
        "source_reference_rows": len(ref),
        "query_split": str(split_path.resolve()),
        "K5_query_ids_and_labels_matched": True,
    }
    return pred, result, ref, audit


def main() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config = json.loads((PROJECT / "main_bidirectional/config_prometa_training.json").read_text())
    assert config["zero_shot"]["reference_rows_per_training_task_per_class"] == 5
    assert config["zero_shot"]["exclude_source_validation_tasks"] is True
    paths = {name: (PROJECT / "code" / relative).resolve()
             for name, relative in config["data"].items()}
    all_predictions, all_metrics, all_references, audits = [], [], [], []
    for source, target in DIRECTIONS:
        for seed in SEEDS:
            pred, metric, ref, audit = one_run(source, target, seed, paths, config)
            all_predictions.append(pred)
            all_metrics.append(metric)
            all_references.append(ref)
            audits.append(audit)
            print(f"{source}->{target} seed {seed}: AUROC {metric['auroc']:.3f}", flush=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    per_seed = pd.DataFrame(all_metrics)
    summary = per_seed.groupby("direction", sort=False).agg(
        seeds=("seed", "nunique"), tasks=("tasks", "first"),
        query_rows=("query_rows", "first"),
        positive_queries=("positive_queries", "first"),
        negative_queries=("negative_queries", "first"),
        source_training_tasks=("source_training_tasks", "first"),
        source_reference_rows=("source_reference_rows", "first"),
        auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc=("auprc", "mean"), auprc_sd=("auprc", "std"),
        balacc=("balacc", "mean"), balacc_sd=("balacc", "std"),
    ).reset_index()
    assert len(per_seed) == 6 and len(summary) == 2
    predictions.to_csv(ROOT / "k0_query_predictions.csv", index=False)
    per_seed.to_csv(ROOT / "k0_per_seed_metrics.csv", index=False)
    summary.to_csv(ROOT / "k0_summary.csv", index=False)
    pd.concat(all_references, ignore_index=True).to_csv(
        ROOT / "k0_source_reference_manifest.csv", index=False)
    (ROOT / "k0_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(summary[["direction", "tasks", "query_rows", "auroc", "auroc_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
