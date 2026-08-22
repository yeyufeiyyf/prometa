#!/usr/bin/env python3
"""Rebuild the direction-specific three-seed main sequence-aware family."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 2025, 3407)
SOURCES = ("CRBN", "VHL")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    child_env = os.environ.copy()
    child_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    for source in SOURCES:
        source_slug = source.lower()
        root = (
            ROOT
            / "checkpoints"
            / "sequence_aware"
            / "retrained_main_2026"
            / f"{source_slug}_source"
        )
        root.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            output = root / f"prometa_{source_slug}_seed{seed}.pt"
            expected = (
                output,
                output.with_name(f"{output.stem}_poi{output.suffix}"),
                output.with_name(f"{output.stem}_e3{output.suffix}"),
                output.with_suffix(".json"),
            )
            if any(path.exists() for path in expected) and not args.force:
                raise FileExistsError(
                    f"Refusing to overwrite an existing triplet for {source} seed {seed}; "
                    "pass --force to retrain it."
                )
            with tempfile.TemporaryDirectory(
                prefix=f".{output.stem}.", dir=root
            ) as temporary_directory:
                temporary_output = Path(temporary_directory) / output.name
                temporary_expected = (
                    temporary_output,
                    temporary_output.with_name(
                        f"{temporary_output.stem}_poi{temporary_output.suffix}"
                    ),
                    temporary_output.with_name(
                        f"{temporary_output.stem}_e3{temporary_output.suffix}"
                    ),
                    temporary_output.with_suffix(".json"),
                )
                command = [
                    sys.executable,
                    "-B",
                    str(ROOT / "experiments" / "train_prometa.py"),
                    "--source-e3",
                    source,
                    "--seed",
                    str(seed),
                    "--k",
                    "2",
                    "--train-q",
                    "3",
                    "--epochs",
                    "100",
                    "--episodes-per-epoch",
                    "100",
                    "--device",
                    args.device,
                    "--cpu-threads",
                    "1",
                    "--output",
                    str(temporary_output),
                ]
                print(f"training source={source} seed={seed}", flush=True)
                subprocess.run(command, cwd=ROOT, env=child_env, check=True)
                for path in temporary_expected:
                    if not path.is_file():
                        raise FileNotFoundError(path)
                audit = json.loads(temporary_expected[-1].read_text(encoding="utf-8"))
                audit["poi_checkpoint"] = str(expected[1].relative_to(ROOT))
                audit["e3_checkpoint"] = str(expected[2].relative_to(ROOT))
                temporary_expected[-1].write_text(
                    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
                )
                for temporary_path, final_path in zip(temporary_expected, expected):
                    temporary_path.replace(final_path)

    subprocess.run(
        [sys.executable, "-B", str(ROOT / "experiments" / "build_retrained_sequence_manifest.py")],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
