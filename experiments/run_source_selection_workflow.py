#!/usr/bin/env python3
"""Auditable source-only training, validation selection, and target evaluation.

This entry point demonstrates the stated label-access boundary.  It does not
replace the retained historical checkpoints used for numerical reproduction.
Candidate checkpoints are selected only with source-E3 validation episodes;
an optional held-out-E3 split is evaluated only after the checkpoint is frozen.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from prometa.data import graph_index_from_cache, protein_index_from_csv, read_contexts
from prometa.evaluation import evaluate_split
from prometa.training import train_episodic_sequence_aware


ROOT = Path(__file__).resolve().parents[1]


def selected_targets(
    membership: pd.DataFrame,
    source_e3: str,
    seed: int,
    partition: str,
) -> set[str]:
    selected = membership[
        membership["source_e3"].astype(str).eq(source_e3)
        & membership["seed"].astype(int).eq(int(seed))
        & membership["partition"].astype(str).eq(partition)
    ]
    return set(selected["target"].astype(str))


def validate_boundary(
    contexts: pd.DataFrame,
    membership: pd.DataFrame,
    validation_split: dict[str, Any],
    source_e3: str,
    seed: int,
) -> tuple[set[str], set[str]]:
    train_targets = selected_targets(
        membership, source_e3, seed, "source_train"
    )
    validation_targets = selected_targets(
        membership, source_e3, seed, "source_validation"
    )
    if not train_targets or not validation_targets:
        raise ValueError("source train/validation membership is empty")
    if train_targets & validation_targets:
        raise ValueError("source train and validation Targets overlap")
    source_targets = set(
        contexts.loc[
            contexts["E3 ligase"].astype(str).str.strip().eq(source_e3)
            & contexts["Target"].notna(),
            "Target",
        ].astype(str)
    )
    if train_targets | validation_targets != source_targets:
        raise ValueError("source membership does not cover the source-E3 Targets exactly")
    meta = validation_split.get("meta", {})
    if str(meta.get("source_e3")) != source_e3 or int(meta.get("seed")) != int(seed):
        raise ValueError("validation split does not match --source-e3/--seed")
    if str(meta.get("held_out_target_e3_access")) != "none":
        raise ValueError("source selection manifest does not prohibit target-E3 access")
    episode_targets = set(validation_split.get("meta_valid", {}))
    if not episode_targets.issubset(validation_targets):
        raise ValueError("validation episodes contain non-validation Targets")
    return train_targets, validation_targets


def candidate_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {
        "candidate_id",
        "learning_rate",
        "weight_decay",
        "epochs",
        "episodes_per_epoch",
        "protein_embed_dim",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")
    return frame.to_dict("records")


def metrics(predictions: pd.DataFrame) -> tuple[float, float]:
    labels = predictions["y_true"].astype(int).to_numpy()
    probabilities = predictions["prob"].astype(float).to_numpy()
    if set(labels) != {0, 1}:
        raise ValueError("source-validation predictions do not contain both classes")
    return float(roc_auc_score(labels, probabilities)), float(
        average_precision_score(labels, probabilities)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-e3", choices=["CRBN", "VHL"], required=True)
    parser.add_argument("--seed", type=int, default=42)
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
        "--sequence-csv",
        type=Path,
        default=ROOT / "data" / "protac_with_seq.csv",
    )
    parser.add_argument(
        "--membership",
        type=Path,
        default=ROOT / "splits" / "source_development" / "source_target_partitions.csv",
    )
    parser.add_argument(
        "--validation-split",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "experiments" / "source_selection_candidates.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "source_selection",
    )
    parser.add_argument("--target-split", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validation_path = args.validation_split or (
        ROOT
        / "splits"
        / "source_development"
        / f"{args.source_e3.lower()}_source_development_seed{args.seed}.json"
    )
    contexts = read_contexts(args.contexts)
    membership = pd.read_csv(args.membership)
    validation_split = json.loads(validation_path.read_text())
    train_targets, validation_targets = validate_boundary(
        contexts,
        membership,
        validation_split,
        args.source_e3,
        args.seed,
    )
    candidates = candidate_rows(args.candidates)
    plan = {
        "workflow": "source_only_checkpoint_selection_then_frozen_target_evaluation",
        "source_e3": args.source_e3,
        "seed": args.seed,
        "source_train_targets": len(train_targets),
        "source_validation_targets": len(validation_targets),
        "eligible_validation_targets": len(validation_split.get("meta_valid", {})),
        "candidate_ids": [str(row["candidate_id"]) for row in candidates],
        "selection_metric_order": ["AUROC", "AUPRC", "candidate_id"],
        "target_labels_used_for_selection": False,
        "target_split_after_freeze": str(args.target_split) if args.target_split else None,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    graph_index = graph_index_from_cache(args.graph_cache)
    poi_index = protein_index_from_csv(args.sequence_csv, "POI_seq")
    e3_index = protein_index_from_csv(args.sequence_csv, "E3_seq")
    train_contexts = contexts[
        contexts["E3 ligase"].astype(str).str.strip().eq(args.source_e3)
        & contexts["Target"].astype(str).isin(train_targets)
    ].copy()
    output = args.output_root / f"{args.source_e3.lower()}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_dir = output / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        encoder, poi_encoder, e3_encoder, audit = train_episodic_sequence_aware(
            train_contexts,
            graph_index,
            poi_index,
            e3_index,
            source_e3=args.source_e3,
            k_shot=2,
            q_query=3,
            seed=args.seed,
            device=device,
            epochs=int(candidate["epochs"]),
            episodes_per_epoch=int(candidate["episodes_per_epoch"]),
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
            protein_embed_dim=int(candidate["protein_embed_dim"]),
        )
        gcn_path = candidate_dir / "prometa_gcn.pt"
        poi_path = candidate_dir / "prometa_poi.pt"
        e3_path = candidate_dir / "prometa_e3.pt"
        torch.save(encoder.state_dict(), gcn_path)
        torch.save(poi_encoder.state_dict(), poi_path)
        torch.save(e3_encoder.state_dict(), e3_path)
        predictions = evaluate_split(
            split_path=validation_path,
            checkpoint=gcn_path,
            graph_index=graph_index,
            contexts=contexts,
            device=device,
            phase="meta_valid",
            poi_checkpoint=poi_path,
            e3_checkpoint=e3_path,
            poi_index=poi_index,
            e3_index=e3_index,
            metadata_overrides={
                "direction": f"{args.source_e3} source development",
                "source": args.source_e3,
                "target": args.source_e3,
            },
        )
        auroc, auprc = metrics(predictions)
        predictions.to_csv(candidate_dir / "source_validation_predictions.csv", index=False)
        (candidate_dir / "training_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )
        score_rows.append(
            {
                **candidate,
                "source_validation_auroc": auroc,
                "source_validation_auprc": auprc,
                "gcn_checkpoint": str(gcn_path.relative_to(ROOT)),
                "poi_checkpoint": str(poi_path.relative_to(ROOT)),
                "e3_checkpoint": str(e3_path.relative_to(ROOT)),
            }
        )

    scores = pd.DataFrame(score_rows).sort_values(
        ["source_validation_auroc", "source_validation_auprc", "candidate_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    scores.to_csv(output / "source_validation_scores.csv", index=False)
    selected = scores.iloc[0].to_dict()
    selected_dir = output / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_paths: dict[str, str] = {}
    for role in ("gcn", "poi", "e3"):
        source = ROOT / str(selected[f"{role}_checkpoint"])
        destination = selected_dir / f"prometa_{role}.pt"
        shutil.copy2(source, destination)
        selected_paths[role] = str(destination.relative_to(ROOT))

    selection_manifest = {
        **plan,
        "selected_candidate": str(selected["candidate_id"]),
        "selected_source_validation_auroc": float(
            selected["source_validation_auroc"]
        ),
        "selected_source_validation_auprc": float(
            selected["source_validation_auprc"]
        ),
        "selected_checkpoints": selected_paths,
        "historical_result_reproduction": "use retained checkpoints; this workflow is an auditable retraining path",
    }
    (output / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2) + "\n"
    )

    if args.target_split is not None:
        target_predictions = evaluate_split(
            split_path=args.target_split,
            checkpoint=ROOT / selected_paths["gcn"],
            graph_index=graph_index,
            contexts=contexts,
            device=device,
            phase="meta_test",
            poi_checkpoint=ROOT / selected_paths["poi"],
            e3_checkpoint=ROOT / selected_paths["e3"],
            poi_index=poi_index,
            e3_index=e3_index,
            metadata_overrides={"selection_data_scope": "source_validation_only"},
        )
        target_predictions.to_csv(
            output / "held_out_target_predictions.csv", index=False
        )
    print(json.dumps(selection_manifest, indent=2))


if __name__ == "__main__":
    main()
