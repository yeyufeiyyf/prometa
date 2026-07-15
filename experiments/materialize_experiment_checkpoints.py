#!/usr/bin/env python3
"""Create one explicitly named checkpoint artifact per reported ProMeta experiment.

Most entries are byte-for-byte copies of the encoder checkpoint that produced
the released result.  Three historical analyses (the S7 ablation bundle, the
compound-family audit, and the legacy VZ185 audit) did not retain a sufficient
standalone model/split pair.  For those entries, this script serializes the
released per-query or per-compound predictions as frozen prediction
checkpoints so their reported values remain exactly reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 2025, 3407)
DEFAULT_REGISTRY = ROOT / "experiments" / "response_letter_experiments.csv"
DEFAULT_OUTPUT = ROOT / "checkpoints" / "by_experiment"
DEFAULT_MANIFEST = ROOT / "experiments" / "experiment_checkpoint_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def alias_path(root: Path, experiment_id: str, seed: int | None = None) -> Path:
    suffix = f"_seed{seed}" if seed is not None else ""
    return root / experiment_id / f"{experiment_id}{suffix}.pt"


def write_sidecar(path: Path, metadata: dict[str, Any]) -> None:
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def copy_encoder(
    rows: list[dict[str, Any]],
    output_root: Path,
    experiment_id: str,
    response_table: str,
    variant: str,
    seed: int,
    source: Path,
    notes: str,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = alias_path(output_root, experiment_id, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_digest = sha256(source)
    destination_digest = sha256(destination)
    if source_digest != destination_digest:
        raise RuntimeError(f"Checkpoint copy hash mismatch: {source} -> {destination}")
    metadata = {
        "experiment_id": experiment_id,
        "response_table": response_table,
        "variant": variant,
        "artifact_kind": "encoder_state_dict",
        "seed": seed,
        "source_checkpoint": relative(source),
        "source_sha256": source_digest,
        "checkpoint_sha256": destination_digest,
        "notes": notes,
    }
    write_sidecar(destination, metadata)
    rows.append({**metadata, "checkpoint": relative(destination), "n_prediction_rows": ""})


def save_frozen_predictions(
    rows: list[dict[str, Any]],
    output_root: Path,
    experiment_id: str,
    response_table: str,
    variant: str,
    predictions: pd.DataFrame,
    source_file: Path,
    notes: str,
    seed: int | None = None,
    expected_summary: dict[str, Any] | None = None,
) -> None:
    destination = alias_path(output_root, experiment_id, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "prometa_frozen_prediction_checkpoint_v1",
        "experiment_id": experiment_id,
        "response_table": response_table,
        "variant": variant,
        "seed": seed,
        "source_file": relative(source_file),
        "source_sha256": sha256(source_file),
        "prediction_columns": predictions.columns.tolist(),
        "predictions": predictions.to_dict(orient="records"),
        "expected_summary": expected_summary or {},
        "notes": notes,
    }
    torch.save(payload, destination)
    metadata = {
        "experiment_id": experiment_id,
        "response_table": response_table,
        "variant": variant,
        "artifact_kind": "frozen_prediction_checkpoint",
        "seed": "" if seed is None else seed,
        "source_checkpoint": relative(source_file),
        "source_sha256": payload["source_sha256"],
        "checkpoint_sha256": sha256(destination),
        "notes": notes,
    }
    write_sidecar(destination, metadata)
    rows.append(
        {
            **metadata,
            "checkpoint": relative(destination),
            "n_prediction_rows": len(predictions),
        }
    )


def materialize_registry_encoders(
    registry: pd.DataFrame,
    output_root: Path,
    rows: list[dict[str, Any]],
) -> None:
    selected = registry[registry["runner"] == "checkpoint_eval"]
    for _, row in selected.iterrows():
        experiment_id = str(row["experiment_id"])
        for seed in SEEDS:
            source = ROOT / str(row["source_checkpoint_pattern"]).format(seed=seed)
            copy_encoder(
                rows,
                output_root,
                experiment_id,
                str(row["response_table"]),
                str(row["variant"]),
                seed,
                source,
                str(row["notes"]),
            )


def materialize_s8(output_root: Path, rows: list[dict[str, Any]]) -> None:
    for source, target in (("CRBN", "VHL"), ("VHL", "CRBN")):
        for q_query in (3, 5):
            experiment_id = (
                f"s8_rdkit_3d_{source.lower()}_{target.lower()}_k2q{q_query}"
            )
            slug = f"{source}_to_{target}_K2Q{q_query}"
            for seed in SEEDS:
                copy_encoder(
                    rows,
                    output_root,
                    experiment_id,
                    "S8",
                    "ProMeta + RDKit 3D descriptors",
                    seed,
                    ROOT / "checkpoints" / "prometa" / slug / f"protonet_encoder_seed{seed}.pt",
                    "K/Q-specific encoder used by evaluate_prometa_3d.py.",
                )


def materialize_s7(output_root: Path, rows: list[dict[str, Any]]) -> None:
    source_file = ROOT / "results" / "released" / "s6_s7_prometa_variants_query_predictions.csv"
    summary_file = ROOT / "results" / "released" / "s6_s7_prometa_variants_summary.csv"
    predictions = pd.read_csv(source_file, low_memory=False)
    summaries = pd.read_csv(summary_file, low_memory=False)
    variants = {
        "ProMeta": "prometa",
        "ProMeta + linear head": "linear",
        "ProMeta w/o episodic training": "no_episodic",
    }
    for direction in ("CRBN->VHL", "VHL->CRBN"):
        direction_slug = direction.replace("->", "_").lower()
        for q_query in (3, 5):
            for method, variant_slug in variants.items():
                experiment_id = f"s7_{direction_slug}_k2q{q_query}_{variant_slug}"
                expected = summaries[
                    (summaries["direction"] == direction)
                    & (summaries["k_shot"] == 2)
                    & (summaries["q_query"] == q_query)
                    & (summaries["method"] == method)
                ]
                expected_summary = expected.iloc[0].to_dict() if len(expected) == 1 else {}
                for seed in SEEDS:
                    subset = predictions[
                        (predictions["direction"] == direction)
                        & (predictions["k_shot"] == 2)
                        & (predictions["q_query"] == q_query)
                        & (predictions["method"] == method)
                        & (predictions["seed"] == seed)
                    ].copy()
                    if subset.empty:
                        raise ValueError(f"No released predictions for {experiment_id}, seed {seed}")
                    save_frozen_predictions(
                        rows,
                        output_root,
                        experiment_id,
                        "S7",
                        method,
                        subset,
                        source_file,
                        (
                            "Frozen per-query checkpoint. The historical S7 run did not retain "
                            "a complete standalone model/input pair for every variant."
                        ),
                        seed=seed,
                        expected_summary=expected_summary,
                    )


def materialize_case_audits(output_root: Path, rows: list[dict[str, Any]]) -> None:
    family_file = ROOT / "results" / "released" / "four_family_main_checkpoint_compound_predictions.csv"
    family_summary_file = ROOT / "results" / "released" / "four_family_main_checkpoint_summary.csv"
    family_predictions = pd.read_csv(family_file, low_memory=False)
    family_summaries = pd.read_csv(family_summary_file, low_memory=False)
    for family in ("BCL-xL", "NAMPT", "BTK", "CDK6"):
        experiment_id = f"r9_{family.lower().replace('-', '_')}"
        subset = family_predictions[
            family_predictions["family_short"].astype(str).str.startswith(f"{family} ")
        ].copy()
        expected = family_summaries[
            family_summaries["family_short"].astype(str).str.startswith(f"{family} ")
        ]
        if subset.empty:
            raise ValueError(f"No released compound-family predictions for {family}")
        save_frozen_predictions(
            rows,
            output_root,
            experiment_id,
            "R9",
            f"ProMeta {family} compound-family audit",
            subset,
            family_file,
            "Frozen compound-level probabilities aggregated from the seed-matched S3 checkpoint family.",
            expected_summary=expected.iloc[0].to_dict() if len(expected) == 1 else {},
        )

    vz_file = ROOT / "results" / "released" / "vz185_prometa_compound_audit.csv"
    vz_summary_file = ROOT / "results" / "released" / "vz185_prometa_summary.csv"
    vz_predictions = pd.read_csv(vz_file, low_memory=False)
    vz_summary = pd.read_csv(vz_summary_file, low_memory=False)
    save_frozen_predictions(
        rows,
        output_root,
        "vz185_legacy_excel_first12",
        "VZ185",
        "ProMeta legacy Excel-order retrospective audit",
        vz_predictions,
        vz_file,
        "Frozen 12-row correctness audit; the historical sheet has no calibrated probabilities.",
        expected_summary=vz_summary.iloc[0].to_dict(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_root} exists; pass --force to replace it")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(args.registry).fillna("")
    rows: list[dict[str, Any]] = []
    materialize_registry_encoders(registry, args.output_root, rows)
    materialize_s8(args.output_root, rows)
    materialize_s7(args.output_root, rows)
    materialize_case_audits(args.output_root, rows)

    manifest = pd.DataFrame(rows).sort_values(
        ["response_table", "experiment_id", "seed"], kind="stable"
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print(f"Wrote {len(manifest)} checkpoint artifacts to {args.output_root}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
