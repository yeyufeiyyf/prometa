#!/usr/bin/env python3
"""Save study-named copies of the frozen ProMeta model weights."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from prometa.release_io import DIRECTIONS, PAPER_SEEDS, SEEDS, sha256


def source_checkpoint(artifacts: Path, source: str, target: str, seed: int) -> Path:
    return (artifacts / "context_weights/checkpoints" / f"{source}_to_{target}"
            / f"seed_{seed}" / "grounded_interaction.pt")


def transfer_checkpoint(artifacts: Path, source: str, target: str, seed: int) -> Path:
    return (artifacts / "current_transfer/table_s4_prometa_rerun/checkpoints"
            / f"{source}_to_{target}" / f"seed_{seed}" / "train_k2q3/model.pt")


def records(artifacts: Path):
    for source, target in DIRECTIONS:
        direction = f"{source}_to_{target}"
        for seed in PAPER_SEEDS:
            yield "kcurve", direction, seed, transfer_checkpoint(artifacts, source, target, seed)
        for seed in SEEDS:
            yield "poi", direction, seed, source_checkpoint(artifacts, source, target, seed)
        for seed in PAPER_SEEDS:
            yield "linker", direction, seed, source_checkpoint(artifacts, source, target, seed)
    for family in ("HPK1", "FLT3", "FLT3_10nM"):
        for seed in PAPER_SEEDS:
            yield "families", family, seed, source_checkpoint(artifacts, "VHL", "CRBN", seed)
    for seed in PAPER_SEEDS:
        yield "efficiency", "FLT3", seed, source_checkpoint(artifacts, "VHL", "CRBN", seed)
    for seed in SEEDS:
        original = (artifacts / "rare_e3/results/prometa2_multisource_loeo_v1"
                    / "common_multisource" / f"seed_{seed}" / "grounded_interaction.pt")
        yield "rare_e3", "common_multisource", seed, original
    for seed in PAPER_SEEDS:
        yield "cdk6", "CDK6", seed, artifacts / "cdk6/checkpoints" / f"seed_{seed}" / "model.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    entries = []
    for study, condition, seed, original in records(artifacts):
        named = artifacts / "study_checkpoints" / study / condition / f"seed_{seed}" / "model.pt"
        if not original.is_file():
            raise FileNotFoundError(original)
        digest = sha256(original)
        if named.is_file():
            if sha256(named) != digest:
                raise ValueError(f"Named checkpoint differs from its source: {named}")
        else:
            named.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, named)
        entries.append({"study": study, "condition": condition, "seed": seed,
                        "checkpoint": named.relative_to(artifacts).as_posix(),
                        "source_checkpoint": original.relative_to(artifacts).as_posix(),
                        "sha256": digest, "relationship": "byte_identical_copy"})
    mapping = artifacts / "study_checkpoints/checkpoint_mapping.json"
    mapping.write_text(json.dumps({
        "note": "Study-named files are copies of the original weights, not independent training runs.",
        "checkpoints": entries,
    }, indent=2) + "\n")
    print(f"Materialized {len(entries)} study/condition/seed checkpoint files", flush=True)


if __name__ == "__main__":
    main()
