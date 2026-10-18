#!/usr/bin/env python3
"""Run one direction/seed/variant cell of the E3-v2 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Must be set before the first CUDA BLAS handle is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from prometa.ablations.experiment import run_experiment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "ablation_variants.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-e3", required=True)
    parser.add_argument("--target-e3", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--episodes-per-epoch", type=int)
    parser.add_argument("--ctp-epochs", type=int)
    parser.add_argument("--max-ctp-rows", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {args.device}")
    device = torch.device(args.device)

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ROOT = (config_path.parent / config.get("path_base", "..")).resolve()
    deterministic = bool(config.get("deterministic_algorithms", False))
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    data_config = config["data"]
    paths = {
        "contexts": resolve(data_config["contexts"]),
        "sequence_table": resolve(data_config["sequence_table"]),
        "tuple_table": resolve(data_config["tuple_table"]),
        "component_table": resolve(data_config["component_table"]),
        "protein_embeddings": resolve(data_config["protein_embeddings"]),
        "split_root": resolve(data_config["split_root"]),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Configured experiment inputs are missing: {missing}")

    settings = {
        key: config[key]
        for key in (
            "k_shot",
            "train_q",
            "test_q",
            "epochs",
            "episodes_per_epoch",
            "learning_rate",
            "weight_decay",
            "ctp_epochs",
            "ctp_batch_size",
            "ctp_learning_rate",
            "evaluation_e3_conditions",
        )
    }
    settings.update(
        {
            "split_protocol": config.get("split_protocol", "legacy_s3_q"),
            "test_k": config.get("test_k"),
            "compatibility_loss_weight": config.get(
                "compatibility_loss_weight", 0.0
            ),
            "source_validation_fraction": config.get(
                "source_validation_fraction", 0.0
            ),
            "source_validation_episodes_per_task": config.get(
                "source_validation_episodes_per_task", 1
            ),
            "early_stopping_patience": config.get(
                "early_stopping_patience", 0
            ),
            "validation_min_delta": config.get("validation_min_delta", 0.0),
            "identity_null_control": config.get("identity_null_control"),
            "supervised_train_scope": config.get(
                "supervised_train_scope", "all_non_target_e3"
            ),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "deterministic_algorithms": deterministic,
        }
    )
    if args.smoke:
        settings.update(
            {
                "epochs": 2,
                "episodes_per_epoch": 3,
                "ctp_epochs": 1,
                "ctp_batch_size": 16,
                "max_ctp_rows": 64,
            }
        )
    for argument, key in (
        (args.epochs, "epochs"),
        (args.episodes_per_epoch, "episodes_per_epoch"),
        (args.ctp_epochs, "ctp_epochs"),
        (args.max_ctp_rows, "max_ctp_rows"),
    ):
        if argument is not None:
            if argument < 1:
                raise ValueError(f"--{key.replace('_', '-')} must be positive")
            settings[key] = argument

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else resolve(config["outputs"]["results"])
    )
    if args.checkpoint_root is not None:
        checkpoint_root = args.checkpoint_root.resolve()
    elif args.output_root is not None:
        checkpoint_root = output_root.parent / f"{output_root.name}_checkpoints"
    else:
        checkpoint_root = resolve(config["outputs"]["checkpoints"])
    if args.smoke and args.checkpoint_root is None:
        checkpoint_root = checkpoint_root / "_smoke"
    run_dir = (
        output_root
        / f"{args.source_e3}_to_{args.target_e3}"
        / f"seed_{args.seed}"
        / args.variant
    )
    checkpoint = (
        checkpoint_root
        / f"{args.source_e3}_to_{args.target_e3}"
        / f"seed_{args.seed}"
        / f"{args.variant}.pt"
    )
    manifest = run_experiment(
        variant_name=args.variant,
        source_e3=args.source_e3,
        target_e3=args.target_e3,
        seed=args.seed,
        device=device,
        paths=paths,
        output_dir=run_dir,
        checkpoint_path=checkpoint,
        settings=settings,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "completed": manifest["completed"],
                "variant": manifest["variant"],
                "direction": f"{manifest['source_e3']}->{manifest['target_e3']}",
                "seed": manifest["seed"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "output": str(run_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
