#!/usr/bin/env python3
"""Build a clean public archive containing ProMeta artifacts only.

The development worktree contains historical revision material.  Public
releases should be made from the directory produced by this script, never by
uploading the whole development worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TOP_LEVEL_FILES = (
    "README.md",
    "LICENSE",
    "environment.yml",
    "requirements.txt",
    "run_all_experiments.sh",
    ".gitignore",
)
TREE_RULES = {
    "prometa": {"suffixes": {".py"}},
    "data": {"suffixes": {".csv", ".npz"}},
    "checkpoints": {"suffixes": {".pt", ".json"}},
    "splits/response_letter": {"suffixes": {".json", ".csv"}},
    "splits/by_experiment": {"suffixes": {".json"}},
    "results/released": {"suffixes": {".csv"}},
}
EXCLUDED_RELEASE_PARTS = {"k2_all_remaining_rebuilt", "prometa_fixed_source_k2q3"}
EXPERIMENT_FILES = (
    "README.md",
    "build_response_splits.py",
    "build_rare_common_query_splits.py",
    "evaluate_prometa.py",
    "evaluate_prometa_3d.py",
    "evaluate_frozen_checkpoints.py",
    "experiment_checkpoint_manifest.csv",
    "experiment_split_manifest.csv",
    "materialize_experiment_checkpoints.py",
    "materialize_experiment_splits.py",
    "summarize_rare_e3.py",
    "train_prometa.py",
    "verify_experiment_checkpoints.py",
    "verify_experiment_splits.py",
    "verify_response_letter.py",
    "response_letter_experiments.csv",
    "response_letter_expected.csv",
)
DOC_FILES = ("EXPERIMENT_CHECKPOINT_MAP.md",)
FIGURE_FILES = (
    "model_framework.png",
    "bidirectional_prometa_only_bar.png",
    "bidirectional_prometa_only_bar.eps",
    "rare_e3_prometa_only_bar.png",
    "rare_e3_prometa_only_bar.eps",
    "vz185_prometa_only_case.png",
    "vz185_prometa_only_case.eps",
)
SCRIPT_FILES = ("export_prometa_release.py", "verify_release_checksums.py")


def copy_file(relative: Path, output: Path) -> None:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def validate_result_scope() -> None:
    failures: list[str] = []
    for path in sorted((ROOT / "results" / "released").glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        if "method" not in frame.columns:
            continue
        methods = frame["method"].dropna().astype(str).unique()
        invalid = sorted(method for method in methods if not method.lower().startswith("prometa"))
        if invalid:
            failures.append(f"{path.name}: {invalid}")
    if failures:
        raise ValueError("Non-ProMeta result rows found:\n" + "\n".join(failures))


def checksum_manifest(output: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "release_checksums.csv":
            continue
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_csv(output / "release_checksums.csv", index=False)


def build(output: Path, force: bool) -> None:
    validate_result_scope()
    checkpoint_manifest = ROOT / "experiments" / "experiment_checkpoint_manifest.csv"
    if not checkpoint_manifest.is_file():
        raise FileNotFoundError(
            f"{checkpoint_manifest} is missing; run "
            "python experiments/materialize_experiment_checkpoints.py --force"
        )
    split_manifest = ROOT / "experiments" / "experiment_split_manifest.csv"
    if not split_manifest.is_file():
        raise FileNotFoundError(
            f"{split_manifest} is missing; run "
            "python experiments/materialize_experiment_splits.py --force"
        )
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for name in TOP_LEVEL_FILES:
        copy_file(Path(name), output)
    for tree, rule in TREE_RULES.items():
        root = ROOT / tree
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDED_RELEASE_PARTS for part in relative.parts):
                continue
            if path.is_file() and path.suffix in rule["suffixes"]:
                copy_file(path.relative_to(ROOT), output)
    for name in EXPERIMENT_FILES:
        copy_file(Path("experiments") / name, output)
    for name in DOC_FILES:
        copy_file(Path("docs") / name, output)
    for name in FIGURE_FILES:
        copy_file(Path("figures") / name, output)
    for name in SCRIPT_FILES:
        copy_file(Path("scripts") / name, output)

    checksum_manifest(output)
    count = sum(path.is_file() for path in output.rglob("*"))
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"ProMeta-only release: {output}")
    print(f"files: {count}; bytes: {size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "ProMeta")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.output.resolve(), args.force)


if __name__ == "__main__":
    main()
