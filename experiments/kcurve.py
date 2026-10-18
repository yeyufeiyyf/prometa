#!/usr/bin/env python3
"""Evaluate the saved ProMeta K2Q3 models with K=1..7 target supports.

Split selection extends the audited Table S4 rerun rule. The model is held
fixed within each direction/seed; only test-time support size changes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(os.environ["PROMETA_STUDY_WORKSPACE"]).resolve()
ARTIFACTS = Path(os.environ.get("PROMETA_ARTIFACTS_ROOT", Path(__file__).resolve().parents[1] / "artifacts")).resolve()
PROJECT = ROOT / "prometa_as"
PRIOR = ROOT / "table_s4_prometa_rerun"
OUT = ROOT / "k1_to_k7_20261007"
DATA_PATH = PROJECT / "data-prometa/tables/strict_crbn_vhl_contexts_fair_esm_v1.csv"
DIRECTIONS = (("CRBN", "VHL"), ("VHL", "CRBN"))
SEEDS = (42, 2025, 3407)
KS = tuple(range(1, 8))
KEY = ["task", "row_id"]

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                   ensure_ascii=False).encode()).hexdigest()


def split_path(source: str, target: str, seed: int, k: int) -> Path:
    return OUT / "splits" / f"{source.lower()}_to_{target.lower()}" / f"seed_{seed}" / f"k{k}_q0.json"


def output_path(source: str, target: str, seed: int) -> Path:
    return OUT / "evaluation" / f"{source}_to_{target}" / f"seed_{seed}"


def make_splits() -> None:
    import pandas as pd

    frame = pd.read_csv(DATA_PATH, dtype={"Compound ID": str})
    if frame["context_key_compound_e3_target"].duplicated().any():
        raise ValueError("Duplicated strict context IDs")
    source_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    audit_rows = []
    for source, target in DIRECTIONS:
        target_frame = frame.loc[frame["E3 ligase"].eq(target)]
        for seed in SEEDS:
            for k in KS:
                episodes = {}
                for task, group in target_frame.groupby("Target", sort=True):
                    classes = {}
                    for label in (1, 0):
                        rows = group.loc[group.Label_bin.eq(label)].sort_values(
                            "context_key_compound_e3_target").to_dict("records")
                        rng = random.Random(int(digest([seed, target, task, label])[:16], 16))
                        rng.shuffle(rows)
                        classes[label] = rows
                    if any(len(rows) < k + 1 for rows in classes.values()):
                        continue

                    def example(row: dict) -> dict:
                        return {
                            "Context ID": row["context_key_compound_e3_target"],
                            "Compound ID": row["Compound ID"],
                            "Label_bin": int(row["Label_bin"]),
                            "E3 ligase": target,
                            "Task": task,
                        }

                    support = [example(row) for label in (1, 0)
                               for row in classes[label][:k]]
                    query = [example(row) for label in (1, 0)
                             for row in classes[label][k:]]
                    assert set(x["Context ID"] for x in support).isdisjoint(
                        x["Context ID"] for x in query)
                    episodes[task] = [{"support": support, "query": query}]
                if not episodes:
                    raise ValueError(f"No K{k} tasks for {source}->{target}")
                meta = {
                    "protocol_version": "e3_v3_nested_k_v1",
                    "source": source, "target": target,
                    "direction": f"{source}->{target}", "seed": seed,
                    "k_shot": k, "q_query": 0,
                    "query_setting": "all_remaining",
                    "eligible_tasks": len(episodes),
                    "eligible_task_names": sorted(episodes),
                    "support_rows_total": sum(len(v[0]["support"]) for v in episodes.values()),
                    "query_rows_total": sum(len(v[0]["query"]) for v in episodes.values()),
                    "task_universe_hash": digest(sorted(episodes)),
                    "query_pool_hash": digest(sorted(
                        row["Context ID"] for values in episodes.values()
                        for row in values[0]["query"])),
                    "context_table_sha256": source_hash,
                    "split_rule": "per-task-class stable shuffle; first K supports, next Q or all remaining queries",
                }
                contents = (json.dumps({"meta": meta, "meta_test": episodes},
                                       indent=2) + "\n").encode()
                if k in (1, 2, 5):
                    prior = PRIOR / "splits" / f"{source.lower()}_to_{target.lower()}" / f"seed_{seed}" / f"k{k}_q0.json"
                    assert prior.read_bytes() == contents, (prior, "split differs")
                path = split_path(source, target, seed, k)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
                audit_rows.append({"direction": f"{source}->{target}", "seed": seed,
                                   "k": k, "tasks": len(episodes),
                                   "query_rows": meta["query_rows_total"],
                                   "split_sha256": hashlib.sha256(contents).hexdigest()})
    pd.DataFrame(audit_rows).to_csv(OUT / "split_audit.csv", index=False)
    assert len(audit_rows) == 42
    print("Generated 42 nested splits; 18 existing K1/K2/K5 splits match byte-for-byte", flush=True)


def evaluate_one(job: tuple[str, str, int]) -> str:
    import numpy as np
    import pandas as pd
    import torch
    from prometa.experiment import build_experiment_data, evaluate_splits
    from prometa.model import E3V2Config, E3V2Model

    source, target, seed = job
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    checkpoint = ARTIFACTS / "study_checkpoints" / "kcurve" / f"{source}_to_{target}" / f"seed_{seed}" / "model.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = E3V2Model(E3V2Config(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    config = json.loads((PROJECT / "main_bidirectional/config_prometa_training.json").read_text())
    paths = {name: (PROJECT / "code" / relative).resolve()
             for name, relative in config["data"].items()}
    data = build_experiment_data(
        contexts_path=paths["contexts"], sequence_path=paths["sequence_table"],
        component_path=paths["component_table"],
        protein_embeddings_path=paths["protein_embeddings"],
        source_e3=source, target_e3=target,
        supervised_train_scope="source_e3_only")
    predictions, metrics, diagnostics = evaluate_splits(
        model, data, split_paths=[split_path(source, target, seed, k) for k in KS],
        interventions=["correct"], device=torch.device("cpu"),
        source_e3=source, seed=seed, variant_name="grounded_interaction")
    assert set(predictions.k_shot) == set(KS)
    assert len(metrics) == 7
    assert predictions.y_score.between(0, 1).all()

    # The extension must reproduce the previously completed K1/K2/K5 results.
    for k in (1, 2, 5):
        old = pd.read_csv(
            PRIOR / "evaluation" / f"{source}_to_{target}" /
            f"seed_{seed}" / f"k{k}_q0/query_predictions.csv",
            dtype={"row_id": str})
        new = predictions.loc[predictions.k_shot.eq(k)]
        paired = old[KEY + ["y_true", "y_score"]].merge(
            new[KEY + ["y_true", "y_score"]], on=KEY, validate="one_to_one",
            suffixes=("_old", "_new"))
        assert len(paired) == len(old) == len(new)
        assert paired.y_true_old.eq(paired.y_true_new).all()
        assert np.allclose(paired.y_score_old, paired.y_score_new,
                           rtol=0, atol=1e-5), (source, target, seed, k,
                                               float(np.max(np.abs(paired.y_score_old-paired.y_score_new))))

    out = output_path(source, target, seed)
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "query_predictions.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    (out / "audit.json").write_text(json.dumps({
        "checkpoint": str(checkpoint.resolve()),
        "splits": [str(split_path(source, target, seed, k).resolve()) for k in KS],
        "prior_k1_k2_k5_predictions_matched": True,
        "diagnostics": diagnostics,
    }, indent=2) + "\n")
    (out / "_SUCCESS").write_text("ok\n")
    return f"{source}->{target}, seed {seed}: 7 K settings, {len(predictions)} predictions"


def summarize() -> None:
    import pandas as pd
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    all_rows = []
    paired_rows = []
    manifest_rows = []
    for source, target in DIRECTIONS:
        direction = f"{source}->{target}"
        for seed in SEEDS:
            out = output_path(source, target, seed)
            assert (out / "_SUCCESS").exists(), out
            predictions = pd.read_csv(out / "query_predictions.csv", dtype={"row_id": str})
            frames = {k: predictions.loc[predictions.k_shot.eq(k)].copy() for k in KS}
            ids = {k: set(map(tuple, frames[k][KEY].to_numpy())) for k in KS}
            assert all(ids[large] <= ids[small] for small, large in zip(KS, KS[1:]))
            reference = frames[7][KEY + ["y_true"]].rename(
                columns={"y_true": "reference_label"})
            assert reference.reference_label.nunique() == 2
            manifest_rows.append(reference.assign(direction=direction, seed=seed))
            for k in KS:
                frame = frames[k]
                assert len(frame) == len(frame.drop_duplicates(KEY))
                for mode, work in (
                    ("all_eligible", frame),
                    ("common_K7_queries", reference.merge(
                        frame[KEY + ["y_true", "y_score"]], on=KEY,
                        validate="one_to_one"))):
                    if mode == "common_K7_queries":
                        assert len(work) == len(reference)
                        assert work.reference_label.eq(work.y_true).all()
                    truth, score = work.y_true, work.y_score
                    row = {
                        "direction": direction, "seed": seed, "k": k,
                        "mode": mode, "tasks": work.task.nunique(),
                        "query_rows": len(work),
                        "positive_queries": int(truth.sum()),
                        "negative_queries": int(len(work) - truth.sum()),
                        "auroc": roc_auc_score(truth, score),
                        "auprc": average_precision_score(truth, score),
                        "balacc": balanced_accuracy_score(truth, score >= 0.5),
                    }
                    (all_rows if mode == "all_eligible" else paired_rows).append(row)
    per_seed = pd.DataFrame(all_rows + paired_rows).sort_values(
        ["mode", "direction", "seed", "k"])
    summary = per_seed.groupby(["mode", "direction", "k"], sort=False).agg(
        seeds=("seed", "nunique"), tasks=("tasks", "first"),
        query_rows=("query_rows", "first"),
        positive_queries=("positive_queries", "first"),
        negative_queries=("negative_queries", "first"),
        auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc=("auprc", "mean"), auprc_sd=("auprc", "std"),
        balacc=("balacc", "mean"), balacc_sd=("balacc", "std"),
    ).reset_index()
    assert len(per_seed) == 84 and len(summary) == 28
    per_seed.to_csv(OUT / "per_seed_metrics.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat(manifest_rows, ignore_index=True).to_csv(
        OUT / "common_K7_query_manifest.csv", index=False)
    print(summary[["mode", "direction", "k", "tasks", "query_rows",
                   "auroc", "auroc_sd"]].to_string(index=False), flush=True)


def audit_outputs() -> None:
    import pandas as pd
    from prometa.data import context_key, stable_context_id

    records = []
    for source, target in DIRECTIONS:
        direction = f"{source}->{target}"
        for seed in SEEDS:
            out = output_path(source, target, seed)
            assert (out / "_SUCCESS").exists(), out
            predictions = pd.read_csv(out / "query_predictions.csv", dtype={"row_id": str})
            checkpoint = ARTIFACTS / "study_checkpoints" / "kcurve" / f"{source}_to_{target}" / f"seed_{seed}" / "model.pt"
            assert checkpoint.is_file()
            support_by_k = {}
            for k in KS:
                split = json.loads(split_path(source, target, seed, k).read_text())
                meta = split["meta"]
                assert (meta["source"], meta["target"], meta["seed"],
                        meta["k_shot"], meta["q_query"]) == (source, target, seed, k, 0)
                expected = []
                supports = {}
                for task, episodes in split["meta_test"].items():
                    assert len(episodes) == 1
                    episode = episodes[0]
                    support, query = episode["support"], episode["query"]
                    assert len(support) == 2 * k
                    assert sorted(row["Label_bin"] for row in support) == [0] * k + [1] * k
                    support_ids = {row["Context ID"] for row in support}
                    query_ids = {row["Context ID"] for row in query}
                    assert len(query_ids) == len(query)
                    assert support_ids.isdisjoint(query_ids)
                    supports[task] = {
                        label: [row["Context ID"] for row in support
                                if row["Label_bin"] == label]
                        for label in (0, 1)
                    }
                    for row in query:
                        rid = stable_context_id(context_key(
                            row["Compound ID"], task, target))
                        expected.append((task + "#0", rid, int(row["Label_bin"])))
                found_frame = predictions.loc[predictions.k_shot.eq(k)]
                found = [(str(row.task), str(row.row_id), int(row.y_true))
                         for row in found_frame.itertuples()]
                assert sorted(expected) == sorted(found), (direction, seed, k)
                assert len(found) == meta["query_rows_total"]
                assert len(supports) == meta["eligible_tasks"]
                support_by_k[k] = supports
                records.append({"direction": direction, "seed": seed,
                                "k": k, "tasks": len(supports),
                                "query_rows": len(found), "passed": True})
            for smaller, larger in zip(KS, KS[1:]):
                for task in support_by_k[larger]:
                    assert task in support_by_k[smaller]
                    for label in (0, 1):
                        assert (support_by_k[smaller][task][label] ==
                                support_by_k[larger][task][label][:smaller])
    assert len(records) == 42
    pd.DataFrame(records).to_csv(OUT / "final_audit.csv", index=False)
    print("PASS: 42/42 evaluated K settings; nested supports and exact query rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("splits", "evaluate", "summary", "audit", "all"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.action in ("splits", "all"):
        OUT.mkdir(parents=True, exist_ok=True)
        make_splits()
    if args.action in ("evaluate", "all"):
        jobs = [(source, target, seed) for source, target in DIRECTIONS
                for seed in SEEDS if not (output_path(source, target, seed) / "_SUCCESS").exists()]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(evaluate_one, jobs):
                print(result, flush=True)
    if args.action in ("summary", "all"):
        summarize()
    if args.action in ("audit", "all"):
        audit_outputs()


if __name__ == "__main__":
    main()
