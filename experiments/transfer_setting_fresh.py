"""Audit the 45 freshly trained transfer cells and recompute S2/S3 ProMeta metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(artifacts: Path, output: Path, *, infer: bool = False,
             direction: str | None = None, k: int | None = None,
             q: int | None = None, seed: int | None = None) -> pd.DataFrame:
    base = artifacts / "transfer_setting_fresh_20261015"
    jobs = json.loads((base / "jobs.json").read_text())
    assert len(jobs) == 45
    jobs = [job for job in jobs
            if (direction is None or f"{job['source']}_to_{job['target']}" == direction)
            and (k is None or job["k"] == k) and (q is None or job["q"] == q)
            and (seed is None or job["seed"] == seed)]
    if not jobs:
        raise ValueError("No released transfer checkpoint matches the requested direction/K/Q/seed.")
    if infer:
        import torch
        from prometa.experiment import build_experiment_data, evaluate_splits
        from prometa.model import E3V2Config, E3V2Model

        torch.set_num_threads(1)
        feature_root = artifacts / "current_transfer/prometa_as/data-prometa"
        feature_paths = dict(
            contexts_path=feature_root / "tables/strict_crbn_vhl_contexts_fair_esm_v1.csv",
            sequence_path=feature_root / "tables/protac_with_seq_fair_esm_v1.csv",
            component_path=feature_root / "tables/protac-fine_esm_complete_v1.csv",
            protein_embeddings_path=feature_root / "embeddings/protein_embeddings_fair_v1.pt")
        data_cache = {}
    rows = []
    hashes = set()
    score_differences = []
    for job in jobs:
        stem = Path(job["id"])
        run = base / "runs" / stem
        checkpoint = base / "checkpoints" / stem / "model.pt"
        manifest = json.loads((run / "run_manifest.json").read_text())
        audit = json.loads((run / "matched_source_audit.json").read_text())
        direction = f"{job['source'].lower()}_to_{job['target'].lower()}"
        split_path = base / "splits" / direction / f"seed_{job['seed']}" / f"k{job['k']}_q{job['q']}.json"
        split = json.loads(split_path.read_text())
        assert manifest["artifact_sha256"]["checkpoint"] == sha(checkpoint)
        assert manifest["artifact_sha256"]["query_predictions"] == sha(run / "query_predictions.csv")
        assert audit["prior_checkpoint_loaded"] is False
        assert audit["trained_from_random_initialization"] is True
        assert manifest["training"]["selection_used_target_e3"] is False
        assert manifest["settings"]["k_shot"] == job["k"]
        assert manifest["settings"]["train_q"] == job["q"]
        hashes.add(sha(checkpoint))
        if infer:
            pair = (job["source"], job["target"])
            if pair not in data_cache:
                data_cache[pair] = build_experiment_data(
                    **feature_paths, source_e3=pair[0], target_e3=pair[1],
                    supervised_train_scope="source_e3_only")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = E3V2Model(E3V2Config(**payload["model_config"]))
            model.load_state_dict(payload["model_state_dict"])
            frame, _, _ = evaluate_splits(
                model, data_cache[pair], split_paths=[split_path], interventions=["correct"],
                device=torch.device("cpu"), source_e3=pair[0], seed=job["seed"],
                variant_name="grounded_interaction")
            prediction_path = output / "predictions" / stem / "query_predictions.csv"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(prediction_path, index=False)
            frame["compound_id"] = frame.compound_id.astype(str)
            archived = pd.read_csv(run / "query_predictions.csv")
            paired = archived[["task", "row_id", "y_true", "y_score"]].merge(
                frame[["task", "row_id", "y_true", "y_score"]],
                on=["task", "row_id"], suffixes=("_archived", "_new"),
                validate="one_to_one")
            assert len(paired) == len(frame) and paired.y_true_archived.eq(paired.y_true_new).all()
            difference = float(np.max(np.abs(paired.y_score_archived-paired.y_score_new)))
            assert difference <= 1e-5, (job["id"], difference)
            score_differences.append(difference)
        else:
            frame = pd.read_csv(run / "query_predictions.csv", dtype={"compound_id": str})
        expected = sorted((task + "#0", str(row["Compound ID"]), int(row["Label_bin"]))
                          for task, episodes in split["meta_test"].items()
                          for row in episodes[0]["query"])
        actual = sorted(zip(frame.task, frame.compound_id, frame.y_true.astype(int)))
        assert expected == actual, job["id"]
        assert frame.y_score.between(0, 1).all()
        task_auc = frame.groupby("task").apply(
            lambda group: roc_auc_score(group.y_true, group.y_score),
            include_groups=False)
        rows.append({"method": "ProMeta setting-fresh",
                     "direction": f"{job['source']}->{job['target']}",
                     "seed": job["seed"], "k": job["k"], "q": job["q"],
                     "query_rows": len(frame), "tasks": len(task_auc),
                     "pooled_auroc": roc_auc_score(frame.y_true, frame.y_score),
                     "macro_auroc": float(task_auc.mean()),
                     "checkpoint_sha256": sha(checkpoint)})
    assert len(hashes) == len(jobs)
    if len(jobs) == 45:
        assert sum(row["query_rows"] for row in rows) == 4695
        assert sum(row["tasks"] for row in rows) == 510
    computed = pd.DataFrame(rows).sort_values(["direction", "k", "q", "seed"])
    reference = pd.read_csv(base / "full_query_per_seed.csv")
    merged = computed.merge(reference, on=["method", "direction", "seed", "k", "q"],
                            suffixes=("", "_reference"), validate="one_to_one")
    assert len(merged) == len(jobs)
    for name in ("query_rows", "tasks"):
        assert merged[name].eq(merged[f"{name}_reference"]).all()
    for name in ("pooled_auroc", "macro_auroc"):
        np.testing.assert_allclose(merged[name], merged[f"{name}_reference"], atol=1e-12, rtol=0)
    summary = computed.groupby(["direction", "k", "q"], as_index=False).agg(
        seeds=("seed", "nunique"), query_rows=("query_rows", "mean"),
        pooled_auroc_mean=("pooled_auroc", "mean"),
        pooled_auroc_sd=("pooled_auroc", "std"),
        macro_auroc_mean=("macro_auroc", "mean"),
        macro_auroc_sd=("macro_auroc", "std"))
    assert summary.seeds.eq(3 if seed is None else 1).all()
    expected_summary = pd.read_csv(base / "full_query_summary.csv")
    checked = summary.merge(expected_summary, on=["direction", "k", "q"],
                            suffixes=("", "_reference"), validate="one_to_one")
    assert len(checked) == len(summary)
    # A single seed is checked against its per-seed reference above, not a three-seed mean/SD.
    if seed is None:
        for name in ("pooled_auroc_mean", "macro_auroc_mean"):
            np.testing.assert_allclose(checked[name], checked[f"{name}_reference"], atol=1e-12, rtol=0)
        for name in ("pooled_auroc", "macro_auroc"):
            np.testing.assert_allclose(checked[f"{name}_sd"], checked[f"{name}_std"], atol=1e-12, rtol=0)
    output.mkdir(parents=True, exist_ok=True)
    computed.to_csv(output / "transfer_setting_fresh_per_seed.csv", index=False)
    summary.to_csv(output / "transfer_setting_fresh_summary.csv", index=False)
    if infer:
        (output / "inference_audit.json").write_text(json.dumps(
            {"evaluations": len(jobs), "query_rows": sum(row["query_rows"] for row in rows),
             "maximum_absolute_score_difference": max(score_differences),
             "score_tolerance": 1e-5,
             "query_identities_and_labels_equal": True}, indent=2) + "\n")
    print(f"PASS: {len(hashes)} distinct weights; {len(jobs)} matched predictions; "
          f"{sum(row['tasks'] for row in rows)} task-seed episodes; "
          f"{sum(row['query_rows'] for row in rows):,} queries; archived metrics match")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("metrics", "infer"), default="metrics")
    parser.add_argument("--direction", choices=("CRBN_to_VHL", "VHL_to_CRBN"))
    parser.add_argument("--k", type=int)
    parser.add_argument("--q", help="Query count per class, all or 0")
    parser.add_argument("--seed", type=int, help="Omit to reproduce all three paper seeds")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; archived results are never overwritten.")
    try:
        q = None if args.q is None else 0 if args.q.lower() == "all" else int(args.q)
        evaluate(args.artifacts.resolve(), args.output.resolve(), infer=args.mode == "infer",
                 direction=args.direction, k=args.k, q=q, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
