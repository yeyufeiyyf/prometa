#!/usr/bin/env python3
"""Build the manifest for reproducibly retrained main sequence-aware triplets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "checkpoints" / "sequence_aware" / "retrained_main_2026"
OUTPUT = ROOT / "experiments" / "retrained_sequence_checkpoint_manifest.csv"
SEEDS = (42, 2025, 3407)
SOURCES = ("CRBN", "VHL")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    try:
        state: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected state dictionary: {path}")
    return state


def main() -> None:
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        source_slug = source.lower()
        root = CHECKPOINT_ROOT / f"{source_slug}_source"
        for seed in SEEDS:
            stem = f"prometa_{source_slug}_seed{seed}"
            gcn = root / f"{stem}.pt"
            poi = root / f"{stem}_poi.pt"
            e3 = root / f"{stem}_e3.pt"
            audit_path = root / f"{stem}.json"
            for path in (gcn, poi, e3, audit_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            expected = {
                "source_e3": source,
                "seed": seed,
                "k_shot": 2,
                "train_q": 3,
                "epochs": 100,
                "episodes_per_epoch": 100,
                "sequence_aware": True,
                "cpu_threads": 1,
            }
            for key, value in expected.items():
                if audit.get(key) != value:
                    raise ValueError(
                        f"Training audit mismatch for {audit_path}: "
                        f"{key}={audit.get(key)!r}, expected {value!r}"
                    )
            for role, path in (("gcn", gcn), ("poi", poi), ("e3", e3)):
                alpha = ""
                if role in {"poi", "e3"}:
                    state = load_state(path)
                    alpha = float(state["alpha"].detach().cpu())
                rows.append(
                    {
                        "setting": f"retrained_main_{source_slug}_source",
                        "seed": seed,
                        "role": role,
                        "status": "reproducibly_retrained",
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256(path),
                        "alpha": alpha,
                        "notes": (
                            "Direction-specific joint GCN/POI/E3 training; "
                            "K=2, train-Q=3, 100 epochs, 100 episodes/epoch"
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT.relative_to(ROOT)} rows={len(rows)}")


if __name__ == "__main__":
    main()
