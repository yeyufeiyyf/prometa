"""Regenerate CDK6 predictions from the three supplied checkpoints and 33 fixed support pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prometa.checkpoints import load_checkpoint, resolve_checkpoint
from prometa.experiment import build_experiment_data, evaluate_splits
from prometa.release_io import sha256
from experiments.cdk6_metrics import calculate


def infer(artifacts: Path, output: Path, device: str = "cpu"):
    case = artifacts / "cdk6"
    base = artifacts / "current_transfer/prometa_as/data-prometa"
    split = case / "splits/cdk6_crbn.json"
    torch.set_num_threads(1)
    data = build_experiment_data(
        contexts_path=case / "cdk6_contexts.csv",
        sequence_path=base / "tables/protac_with_seq_fair_esm_v1.csv",
        component_path=base / "tables/protac-fine_esm_complete_v1.csv",
        protein_embeddings_path=base / "embeddings/protein_embeddings_fair_v1.pt",
        source_e3="VHL", target_e3="CRBN", supervised_train_scope="source_e3_only")
    audits = []
    for seed in (42, 2025, 3407):
        selection = dict(study="cdk6", seed=seed)
        model = load_checkpoint(artifacts, device=device, **selection)
        frame, _, _ = evaluate_splits(
            model, data, split_paths=[split], interventions=["correct"],
            device=torch.device(device), source_e3="VHL", seed=seed,
            variant_name="grounded_interaction")
        frame["compound_id"] = frame.compound_id.astype(str)
        archived = pd.read_csv(case / f"predictions/seed_{seed}.csv", dtype={"compound_id": str})
        paired = archived.merge(frame, on=["seed", "episode", "compound_id"],
                                suffixes=("_archived", "_new"), validate="one_to_one")
        assert len(paired) == len(frame) == len(archived) == 396
        assert paired.y_true_archived.eq(paired.y_true_new).all()
        assert paired.row_id_archived.eq(paired.row_id_new).all()
        assert paired.task_archived.eq(paired.task_new).all()
        difference = float(np.max(np.abs(paired.y_score_archived - paired.y_score_new)))
        assert difference <= 1e-5, (seed, difference)
        pred_dir = output / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(pred_dir / f"seed_{seed}.csv", index=False)
        audits.append(dict(seed=seed, checkpoint_sha256=sha256(resolve_checkpoint(artifacts, **selection)),
                           query_rows=len(frame), maximum_absolute_score_difference=difference))
    result = calculate(case, output, predictions=output / "predictions")
    audit = dict(status="PASS", evaluations=3, support_pairs_per_seed=33,
                 query_rows=1188, query_identities_and_labels_equal=True,
                 score_tolerance=1e-5, maximum_absolute_score_difference=max(
                     item["maximum_absolute_score_difference"] for item in audits),
                 split_sha256=sha256(split), runs=audits, metrics=result)
    (output / "inference_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory.")
    if args.output.resolve() == args.artifacts.resolve() or args.artifacts.resolve() in args.output.resolve().parents:
        parser.error("Do not write inside the reference artifact bundle.")
    infer(args.artifacts.resolve(), args.output.resolve(), args.device)


if __name__ == "__main__":
    main()
