#!/usr/bin/env python3
"""Rerun the ProMeta-only Supplementary Table S4 sensitivity settings."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ["PROMETA_STUDY_WORKSPACE"]).resolve()
PROJECT = ROOT / "prometa_as"
OUT = ROOT / "table_s4_prometa_rerun"
SEEDS = (42, 2025, 3407)
DIRECTIONS = (("CRBN", "VHL"), ("VHL", "CRBN"))
SETTINGS = {
    "VHL": ((1, 1), (1, 0), (2, 3), (2, 5), (2, 0), (5, 0)),
    "CRBN": ((1, 1), (1, 0), (2, 3), (2, 5), (2, 0), (5, 0),
            (1, 10), (2, 10), (5, 10)),
}
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def split_path(source, target, seed, k, q):
    return OUT / "splits" / f"{source.lower()}_to_{target.lower()}" / f"seed_{seed}" / f"k{k}_q{q}.json"


def checkpoint_path(source, target, seed, k, q):
    return (OUT / "checkpoints" / f"{source}_to_{target}" / f"seed_{seed}"
            / f"eval_k{k}q{q}" / "model.pt")


def materialize_checkpoints():
    """Give every evaluation cell a named copy of its original source weight."""
    records = []
    for source, target in DIRECTIONS:
        for seed in SEEDS:
            for k, q in SETTINGS[target]:
                train_q = 5 if (k, q) == (2, 5) else 3
                original = (OUT / "checkpoints" / f"{source}_to_{target}"
                            / f"seed_{seed}" / f"train_k2q{train_q}" / "model.pt")
                named = checkpoint_path(source, target, seed, k, q)
                if not original.is_file():
                    raise FileNotFoundError(original)
                source_hash = hashlib.sha256(original.read_bytes()).hexdigest()
                if named.is_file():
                    if hashlib.sha256(named.read_bytes()).hexdigest() != source_hash:
                        raise ValueError(f"Named checkpoint differs from its source: {named}")
                else:
                    named.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, named)
                records.append({"direction": f"{source}_to_{target}", "seed": seed,
                                "evaluation_k": k, "evaluation_q": q,
                                "trained_k": 2, "trained_q": train_q,
                                "checkpoint": str(named.relative_to(OUT)),
                                "source_checkpoint": str(original.relative_to(OUT)),
                                "sha256": source_hash,
                                "relationship": "byte_identical_copy"})
    path = OUT / "checkpoint_mapping.json"
    path.write_text(json.dumps({"note": "45 named evaluation files copy 12 source-trained weights; matching names do not imply independent training.",
                                "checkpoints": records}, indent=2) + "\n")
    print(f"Materialized {len(records)} setting/seed checkpoint files", flush=True)


def make_splits():
    import pandas as pd

    data_path = PROJECT / "data-prometa/tables/strict_crbn_vhl_contexts_fair_esm_v1.csv"
    frame = pd.read_csv(data_path, dtype={"Compound ID": str})
    if frame["context_key_compound_e3_target"].duplicated().any():
        raise ValueError("Duplicated strict context IDs")
    audit = []
    for source, target in DIRECTIONS:
        target_frame = frame.loc[frame["E3 ligase"].eq(target)]
        for seed in SEEDS:
            for k, q in SETTINGS[target]:
                episodes = {}
                for task, group in target_frame.groupby("Target", sort=True):
                    classes = {}
                    for label in (1, 0):
                        rows = group.loc[group["Label_bin"].eq(label)].sort_values(
                            "context_key_compound_e3_target"
                        ).to_dict("records")
                        rng = random.Random(int(digest([seed, target, task, label])[:16], 16))
                        rng.shuffle(rows)
                        classes[label] = rows
                    required = k + (q if q else 1)
                    if any(len(rows) < required for rows in classes.values()):
                        continue

                    def example(row):
                        return {
                            "Context ID": row["context_key_compound_e3_target"],
                            "Compound ID": row["Compound ID"],
                            "Label_bin": int(row["Label_bin"]),
                            "E3 ligase": target,
                            "Task": task,
                        }

                    support = [example(row) for label in (1, 0) for row in classes[label][:k]]
                    query = [example(row) for label in (1, 0) for row in
                             (classes[label][k:k+q] if q else classes[label][k:])]
                    support_ids = {row["Context ID"] for row in support}
                    query_ids = {row["Context ID"] for row in query}
                    if support_ids & query_ids or len(query_ids) != len(query):
                        raise ValueError("Support/query overlap or duplicate query")
                    episodes[task] = [{"support": support, "query": query}]
                if not episodes:
                    raise ValueError(f"No eligible tasks: {source}->{target} K{k} Q{q}")
                path = split_path(source, target, seed, k, q)
                path.parent.mkdir(parents=True, exist_ok=True)
                meta = {
                    "protocol_version": "e3_v3_nested_k_v1",
                    "source": source, "target": target,
                    "direction": f"{source}->{target}", "seed": seed,
                    "k_shot": k, "q_query": q,
                    "query_setting": f"Q{q}" if q else "all_remaining",
                    "eligible_tasks": len(episodes),
                    "eligible_task_names": sorted(episodes),
                    "support_rows_total": sum(len(v[0]["support"]) for v in episodes.values()),
                    "query_rows_total": sum(len(v[0]["query"]) for v in episodes.values()),
                    "task_universe_hash": digest(sorted(episodes)),
                    "query_pool_hash": digest(sorted(
                        row["Context ID"] for values in episodes.values()
                        for row in values[0]["query"])),
                    "context_table_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    "split_rule": "per-task-class stable shuffle; first K supports, next Q or all remaining queries",
                }
                path.write_text(json.dumps({"meta": meta, "meta_test": episodes}, indent=2) + "\n")
                audit.append({"direction": meta["direction"], "seed": seed, "k": k,
                              "q": q, "tasks": len(episodes),
                              "support_rows": meta["support_rows_total"],
                              "query_rows": meta["query_rows_total"],
                              "split_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(audit).to_csv(OUT / "split_audit.csv", index=False)
    print(f"Generated {len(audit)} audited split files", flush=True)


def train_one(job):
    source, target, seed, q = job
    import torch
    from prometa.experiment import run_experiment

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # ProcessPool workers can handle more than one job.
    config = json.loads((PROJECT / "main_bidirectional/config_prometa_training.json").read_text())
    settings = {key: value for key, value in config.items() if key not in
                ("experiment_tag", "protocol", "seeds", "directions", "variants",
                 "data", "outputs", "archive_note", "reported_k", "zero_shot")}
    settings.update(k_shot=2, train_q=q, test_k=[2], test_q=q,
                    split_protocol="e3_v3_nested_k_v1")
    paths = {key: (PROJECT / "code" / value).resolve()
             for key, value in config["data"].items()}
    paths["split_root"] = OUT / "splits"
    name = f"{source}_to_{target}/seed_{seed}/train_k2q{q}"
    result = run_experiment(
        variant_name="grounded_interaction", source_e3=source,
        target_e3=target, seed=seed, device=torch.device("cpu"),
        paths=paths, output_dir=OUT / "runs" / name,
        checkpoint_path=OUT / "checkpoints" / name / "model.pt",
        settings=settings,
    )
    return {"job": name, "elapsed_seconds": result["elapsed_seconds"],
            "best_epoch": result["training"]["best_epoch"]}


def evaluate_one(job):
    source, target, seed, k, q = job
    import pandas as pd
    import torch
    from prometa.experiment import build_experiment_data, evaluate_splits
    from prometa.model import E3V2Config, E3V2Model

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cp_path = checkpoint_path(source, target, seed, k, q)
    payload = torch.load(cp_path, map_location="cpu", weights_only=False)
    model = E3V2Model(E3V2Config(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    config = json.loads((PROJECT / "main_bidirectional/config_prometa_training.json").read_text())
    paths = {key: (PROJECT / "code" / value).resolve()
             for key, value in config["data"].items()}
    data = build_experiment_data(
        contexts_path=paths["contexts"], sequence_path=paths["sequence_table"],
        component_path=paths["component_table"],
        protein_embeddings_path=paths["protein_embeddings"],
        source_e3=source, target_e3=target,
        supervised_train_scope="source_e3_only",
    )
    pred, metric, diag = evaluate_splits(
        model, data, split_paths=[split_path(source, target, seed, k, q)],
        interventions=["correct"], device=torch.device("cpu"),
        source_e3=source, seed=seed, variant_name="grounded_interaction")
    path = OUT / "evaluation" / f"{source}_to_{target}/seed_{seed}/k{k}_q{q}"
    path.mkdir(parents=True, exist_ok=True)
    pred.to_csv(path / "query_predictions.csv", index=False)
    metric.to_csv(path / "metrics.csv", index=False)
    (path / "audit.json").write_text(json.dumps({"checkpoint": str(cp_path),
        "split": str(split_path(source,target,seed,k,q)), "diagnostics": diag}, indent=2) + "\n")
    (path / "_SUCCESS").write_text("ok\n")
    from sklearn.metrics import balanced_accuracy_score
    return {"direction": f"{source}->{target}", "seed": seed,
            "k": k, "q": q, "query_rows": len(pred),
            "auroc": float(metric["auroc"].iloc[0]),
            "auprc": float(metric["auprc"].iloc[0]),
            "balacc": float(balanced_accuracy_score(pred["y_true"], pred["y_score"].ge(.5)))}


def audit_outputs():
    import math
    import pandas as pd
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    from prometa.data import context_key, stable_context_id

    checks = []
    for source, target in DIRECTIONS:
        for seed in SEEDS:
            for train_q in (3, 5):
                run = OUT / "runs" / f"{source}_to_{target}/seed_{seed}/train_k2q{train_q}"
                manifest = json.loads((run / "run_manifest.json").read_text())
                assert (run / "_SUCCESS").exists()
                assert manifest["training"]["selection_used_target_e3"] is False
                assert manifest["settings"]["supervised_train_scope"] == "source_e3_only"
                assert manifest["seed"] == seed
            for k, q in SETTINGS[target]:
                named = checkpoint_path(source, target, seed, k, q)
                source_q = 5 if (k, q) == (2, 5) else 3
                original = (OUT / "checkpoints" / f"{source}_to_{target}"
                            / f"seed_{seed}" / f"train_k2q{source_q}" / "model.pt")
                assert named.is_file() and original.is_file()
                assert hashlib.sha256(named.read_bytes()).digest() == hashlib.sha256(original.read_bytes()).digest()
                split = json.loads(split_path(source, target, seed, k, q).read_text())
                meta = split["meta"]
                assert (meta["source"], meta["target"], meta["seed"], meta["k_shot"], meta["q_query"]) == (source, target, seed, k, q)
                expected = []
                for task, episodes in split["meta_test"].items():
                    for ep in episodes:
                        support = ep["support"]
                        query = ep["query"]
                        assert len(support) == 2*k
                        assert sorted(x["Label_bin"] for x in support) == [0]*k + [1]*k
                        assert q == 0 or len(query) == 2*q
                        assert {x["Context ID"] for x in support}.isdisjoint(
                            {x["Context ID"] for x in query})
                        for row in query:
                            rid = stable_context_id(context_key(row["Compound ID"], task, target))
                            expected.append((task+"#0", rid, int(row["Label_bin"])))
                assert len(expected) == meta["query_rows_total"]
                output = OUT / "evaluation" / f"{source}_to_{target}/seed_{seed}/k{k}_q{q}"
                assert (output / "_SUCCESS").exists()
                pred = pd.read_csv(output / "query_predictions.csv")
                found = [(str(row.task), str(row.row_id), int(row.y_true))
                         for row in pred.itertuples()]
                assert sorted(expected) == sorted(found)
                assert pred["y_score"].between(0, 1).all()
                metric = pd.read_csv(output / "metrics.csv").iloc[0]
                truth, scores = pred["y_true"], pred["y_score"]
                assert math.isclose(metric["auroc"], roc_auc_score(truth, scores), abs_tol=1e-12)
                assert math.isclose(metric["auprc"], average_precision_score(truth, scores), abs_tol=1e-12)
                checks.append({"direction": f"{source}->{target}", "seed": seed,
                               "k": k, "q": q, "tasks": len(split["meta_test"]),
                               "query_rows": len(pred),
                               "balacc": balanced_accuracy_score(truth, scores.ge(.5)),
                               "passed": True})
            all_splits = {k: json.loads(split_path(source,target,seed,k,0).read_text())["meta_test"]
                          for k in (1,2,5)}
            for small, large in ((1,2),(2,5)):
                for task in set(all_splits[small]) & set(all_splits[large]):
                    for label in (0,1):
                        a = [r["Context ID"] for r in all_splits[small][task][0]["support"]
                             if r["Label_bin"] == label]
                        b = [r["Context ID"] for r in all_splits[large][task][0]["support"]
                             if r["Label_bin"] == label]
                        assert a == b[:small]
            for k,q in SETTINGS[target]:
                if not q:
                    continue
                fixed = json.loads(split_path(source,target,seed,k,q).read_text())["meta_test"]
                remaining = all_splits[k]
                for task, episodes in fixed.items():
                    assert episodes[0]["support"] == remaining[task][0]["support"]
                    assert {r["Context ID"] for r in episodes[0]["query"]}.issubset(
                        {r["Context ID"] for r in remaining[task][0]["query"]})
    pd.DataFrame(checks).to_csv(OUT / "final_audit.csv", index=False)
    print(f"PASS: {len(checks)} split/prediction/metric units, 12 source-only training runs", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("splits", "train", "checkpoints", "evaluate", "all", "status", "audit"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.action in ("splits", "all"):
        make_splits()
    if args.action in ("train", "all"):
        jobs = [(s,t,seed,q) for s,t in DIRECTIONS for seed in SEEDS for q in (3,5)]
        jobs = [j for j in jobs if not (OUT / "runs" /
                f"{j[0]}_to_{j[1]}/seed_{j[2]}/train_k2q{j[3]}/_SUCCESS").exists()]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(train_one, jobs):
                print(json.dumps(result), flush=True)
    if args.action in ("checkpoints", "train", "evaluate", "all"):
        materialize_checkpoints()
    if args.action in ("evaluate", "all"):
        jobs = [(s,t,seed,k,q) for s,t in DIRECTIONS for seed in SEEDS
                for k,q in SETTINGS[t]]
        jobs = [j for j in jobs if not (OUT / "evaluation" /
                f"{j[0]}_to_{j[1]}/seed_{j[2]}/k{j[3]}_q{j[4]}/_SUCCESS").exists()]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(evaluate_one, jobs):
                print(json.dumps(result), flush=True)
    if args.action in ("status", "all", "evaluate"):
        import pandas as pd
        from sklearn.metrics import balanced_accuracy_score
        rows = []
        for s,t in DIRECTIONS:
            for seed in SEEDS:
                for k,q in SETTINGS[t]:
                    p = OUT / "evaluation" / f"{s}_to_{t}/seed_{seed}/k{k}_q{q}"
                    if (p / "_SUCCESS").exists():
                        m = pd.read_csv(p / "metrics.csv").iloc[0]
                        pred = pd.read_csv(p / "query_predictions.csv")
                        rows.append({"direction": f"{s}->{t}", "seed": seed,
                            "k": k, "q": q, "query_rows": int(m["n_query"]),
                            "auroc": m["auroc"], "auprc": m["auprc"],
                            "balacc": balanced_accuracy_score(pred["y_true"], pred["y_score"].ge(.5))})
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(OUT / "per_seed_metrics.csv", index=False)
            summary = df.groupby(["direction","k","q"]).agg(
                seeds=("seed","nunique"), query_rows=("query_rows","mean"),
                auroc=("auroc","mean"), auroc_sd=("auroc","std"),
                auprc=("auprc","mean"), auprc_sd=("auprc","std"),
                balacc=("balacc","mean"), balacc_sd=("balacc","std"))
            summary.to_csv(OUT / "summary.csv")
            print(summary.to_string(), flush=True)
        print(f"Completed evaluation units: {len(rows)}/45", flush=True)
    if args.action in ("audit", "all"):
        audit_outputs()


if __name__ == "__main__":
    main()
