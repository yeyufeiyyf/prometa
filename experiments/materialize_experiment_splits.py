#!/usr/bin/env python3
"""Create one explicitly named split/cohort artifact per ProMeta experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED_PATTERN = re.compile(r"seed(?P<seed>\d+)")
SEEDS = (42, 2025, 3407)
DEFAULT_REGISTRY = ROOT / "experiments" / "response_letter_experiments.csv"
DEFAULT_OUTPUT = ROOT / "splits" / "by_experiment"
DEFAULT_MANIFEST = ROOT / "experiments" / "experiment_split_manifest.csv"


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


def seed_from_path(path: Path) -> int:
    match = SEED_PATTERN.search(str(path))
    if match is None:
        raise ValueError(f"Cannot infer seed from {path}")
    return int(match.group("seed"))


def destination(output_root: Path, experiment_id: str, seed: int | None = None) -> Path:
    suffix = f"_seed{seed}" if seed is not None else "_frozen_cohort"
    return output_root / experiment_id / f"{experiment_id}{suffix}.json"


def copy_split(
    rows: list[dict[str, Any]],
    output_root: Path,
    experiment_id: str,
    response_table: str,
    source: Path,
    notes: str,
) -> None:
    seed = seed_from_path(source)
    target = destination(output_root, experiment_id, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_digest = sha256(source)
    target_digest = sha256(target)
    if source_digest != target_digest:
        raise RuntimeError(f"Split copy hash mismatch: {source} -> {target}")
    rows.append(
        {
            "experiment_id": experiment_id,
            "response_table": response_table,
            "artifact_kind": "support_query_split",
            "seed": seed,
            "split": relative(target),
            "source_split": relative(source),
            "source_sha256": source_digest,
            "split_sha256": target_digest,
            "n_records": "",
            "notes": notes,
        }
    )


def copy_glob(
    rows: list[dict[str, Any]],
    output_root: Path,
    experiment_id: str,
    response_table: str,
    root: Path,
    pattern: str,
    notes: str,
) -> None:
    paths = sorted(root.glob(pattern))
    seeds = {seed_from_path(path) for path in paths}
    if seeds != set(SEEDS):
        raise ValueError(
            f"{experiment_id}: expected seeds {SEEDS}, found {sorted(seeds)} at {root}/{pattern}"
        )
    for path in paths:
        copy_split(rows, output_root, experiment_id, response_table, path, notes)


def materialize_registry(
    registry: pd.DataFrame,
    output_root: Path,
    rows: list[dict[str, Any]],
) -> None:
    for _, row in registry[registry["runner"] == "checkpoint_eval"].iterrows():
        copy_glob(
            rows,
            output_root,
            str(row["experiment_id"]),
            str(row["response_table"]),
            ROOT / str(row["split_root"]),
            str(row["split_glob"]),
            str(row["notes"]),
        )


def original_pattern(source: str, target: str, q_query: int) -> str:
    prefix = "crbn_train_vhl" if source == "CRBN" else "vhl_train_crbn"
    return f"{prefix}_test_strict_K2Q{q_query}_seed*.json"


def materialize_s8(output_root: Path, rows: list[dict[str, Any]]) -> None:
    root = ROOT / "splits" / "response_letter" / "original"
    for source, target in (("CRBN", "VHL"), ("VHL", "CRBN")):
        for q_query in (3, 5):
            experiment_id = (
                f"s8_rdkit_3d_{source.lower()}_{target.lower()}_k2q{q_query}"
            )
            copy_glob(
                rows,
                output_root,
                experiment_id,
                "S8",
                root,
                original_pattern(source, target, q_query),
                "Original fixed-query split used by the RDKit-3D control.",
            )


def materialize_s7(output_root: Path, rows: list[dict[str, Any]]) -> None:
    root = ROOT / "splits" / "response_letter" / "original"
    variants = ("prometa", "linear", "no_episodic")
    for source, target in (("CRBN", "VHL"), ("VHL", "CRBN")):
        direction_slug = f"{source.lower()}_{target.lower()}"
        for q_query in (3, 5):
            for variant in variants:
                experiment_id = f"s7_{direction_slug}_k2q{q_query}_{variant}"
                copy_glob(
                    rows,
                    output_root,
                    experiment_id,
                    "S7",
                    root,
                    original_pattern(source, target, q_query),
                    "Original S7 fixed-query split; predictions are stored in a frozen checkpoint.",
                )


def clean_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def write_frozen_cohort(
    rows: list[dict[str, Any]],
    output_root: Path,
    experiment_id: str,
    response_table: str,
    source: Path,
    frame: pd.DataFrame,
    notes: str,
) -> None:
    target = destination(output_root, experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "prometa_frozen_evaluation_cohort_v1",
        "experiment_id": experiment_id,
        "response_table": response_table,
        "source_file": relative(source),
        "source_sha256": sha256(source),
        "records": clean_records(frame),
        "notes": notes,
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows.append(
        {
            "experiment_id": experiment_id,
            "response_table": response_table,
            "artifact_kind": "frozen_evaluation_cohort",
            "seed": "",
            "split": relative(target),
            "source_split": relative(source),
            "source_sha256": payload["source_sha256"],
            "split_sha256": sha256(target),
            "n_records": len(frame),
            "notes": notes,
        }
    )


def materialize_case_cohorts(output_root: Path, rows: list[dict[str, Any]]) -> None:
    family_file = ROOT / "results" / "released" / "four_family_main_checkpoint_compound_predictions.csv"
    families = pd.read_csv(family_file, low_memory=False)
    for family in ("BCL-xL", "NAMPT", "BTK", "CDK6"):
        subset = families[
            families["family_short"].astype(str).str.startswith(f"{family} ")
        ].copy()
        write_frozen_cohort(
            rows,
            output_root,
            f"r9_{family.lower().replace('-', '_')}",
            "R9",
            family_file,
            subset,
            "Frozen compound-family cohort; no standalone historical support/query split was retained.",
        )

    vz_file = ROOT / "results" / "released" / "vz185_prometa_compound_audit.csv"
    write_frozen_cohort(
        rows,
        output_root,
        "vz185_legacy_excel_first12",
        "VZ185",
        vz_file,
        pd.read_csv(vz_file, low_memory=False),
        "Frozen legacy 12-row cohort; no historical support/query split or calibrated probabilities exist.",
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
    materialize_registry(registry, args.output_root, rows)
    materialize_s8(args.output_root, rows)
    materialize_s7(args.output_root, rows)
    materialize_case_cohorts(args.output_root, rows)
    manifest = pd.DataFrame(rows).sort_values(
        ["response_table", "experiment_id", "seed"], kind="stable"
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print(f"Wrote {len(manifest)} split/cohort artifacts to {args.output_root}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
