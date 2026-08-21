#!/usr/bin/env python3
"""Verify retained sequence-aware source, inputs, and checkpoint coverage."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch

from prometa.evaluation import load_encoder, load_protein_encoder


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "sequence_checkpoint_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    device = torch.device("cpu")
    failures: list[str] = []
    retained = 0
    disclosed_missing = 0

    for _, row in manifest.iterrows():
        status = str(row["status"])
        label = f"{row['setting']} seed={row['seed']} role={row['role']}"
        if status == "not_retained_not_reconstructed":
            disclosed_missing += 1
            if str(row["path"]).strip() or str(row["sha256"]).strip():
                failures.append(f"{label}: missing artifact must not have a path/hash")
            continue
        path = ROOT / str(row["path"])
        if not path.is_file():
            failures.append(f"{label}: missing file {path.relative_to(ROOT)}")
            continue
        retained += 1
        actual_hash = sha256(path)
        if actual_hash != str(row["sha256"]):
            failures.append(f"{label}: SHA-256 mismatch")
            continue
        role = str(row["role"])
        if role == "gcn":
            load_encoder(path, device)
        elif role in {"poi", "e3"}:
            encoder = load_protein_encoder(path, device)
            expected_alpha = float(row["alpha"])
            actual_alpha = float(encoder.alpha.detach().cpu())
            if abs(actual_alpha - expected_alpha) > 1e-12:
                failures.append(
                    f"{label}: alpha expected={expected_alpha} actual={actual_alpha}"
                )

    source_text = (ROOT / "prometa" / "training.py").read_text(encoding="utf-8")
    for required in [
        "*poi_encoder.parameters()",
        "*e3_encoder.parameters()",
        "poi_encoder.alpha * poi",
        "e3_encoder.alpha * e3",
    ]:
        if required not in source_text:
            failures.append(f"integrated training source is missing: {required}")

    main_gcn = ROOT / "checkpoints" / "sequence_aware" / "main_seed42" / "protonet_encoder_seed42.pt"
    numerical_gcn = ROOT / "checkpoints" / "prometa_global" / "protonet_encoder_seed42.pt"
    if sha256(main_gcn) != sha256(numerical_gcn):
        failures.append("main seed-42 GCN provenance copy is not byte-identical")

    print(
        f"sequence provenance: retained={retained}; "
        f"disclosed_missing={disclosed_missing}; failures={len(failures)}"
    )
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
