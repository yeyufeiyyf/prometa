"""Resolve released weights by study, condition and seed (no network access)."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    study: str
    condition: str
    seed: int
    checkpoint: str
    sha256: str
    namespace: str = "main"
    direction: str | None = None
    k: int | None = None
    q: str | None = None
    source_e3: str = "CRBN"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_catalog(artifacts: str | Path = "artifacts") -> list[Checkpoint]:
    """List the 45 current transfer weights and 57 named study weight copies."""
    root = Path(artifacts)
    fresh = root / "transfer_setting_fresh_20261015"
    mapping = root / "study_checkpoints/checkpoint_mapping.json"
    if not (fresh / "jobs.json").is_file() or not mapping.is_file():
        raise FileNotFoundError(
            f"Checkpoint metadata missing under {root.resolve()}. "
            "Extract the matching prometa_artifacts_processed_v2.1.0.zip in the repository root.")
    entries = []
    for job in json.loads((fresh / "jobs.json").read_text()):
        manifest = json.loads((fresh / "runs" / job["id"] / "run_manifest.json").read_text())
        direction = f"{job['source']}_to_{job['target']}"
        q = "all" if job["q"] == 0 else str(job["q"])
        entries.append(Checkpoint(
            study="transfer", condition=f"k{job['k']}_q{q}", seed=job["seed"],
            checkpoint=f"transfer_setting_fresh_20261015/checkpoints/{job['id']}/model.pt",
            sha256=manifest["artifact_sha256"]["checkpoint"],
            direction=direction, k=job["k"], q=q, source_e3=job["source"]))
    for record in json.loads(mapping.read_text())["checkpoints"]:
        direction = record["condition"] if "_to_" in record["condition"] else None
        source = direction.split("_to_")[0] if direction else "VHL"
        rare = record["study"] == "rare_e3"
        entries.append(Checkpoint(
            study=record["study"], condition=record["condition"], seed=record["seed"],
            checkpoint=record["checkpoint"], sha256=record["sha256"],
            namespace="rare" if rare else "main", direction=direction,
            source_e3="CRBN" if rare else source))
    return entries


def select_checkpoint(artifacts: str | Path = "artifacts", *, study: str,
                      seed: int, direction: str | None = None,
                      condition: str | None = None, k: int | None = None,
                      q: str | int | None = None) -> Checkpoint:
    """Select exactly one released checkpoint; never silently fall back to another run."""
    study = "rare_e3" if study == "rare" else study
    if direction:
        direction = direction.upper().replace("->", "_TO_").replace("_TO_", "_to_")
    q = "all" if str(q).lower() in ("0", "all") else None if q is None else str(q)
    if study == "transfer" and (direction is None or k is None or q is None):
        raise ValueError("Transfer selection requires direction, k and q (use q='all' for all queries).")
    if study != "transfer" and (k is not None or q is not None):
        raise ValueError("K/Q select transfer checkpoints only; other studies use condition/direction and seed.")
    entries = [item for item in checkpoint_catalog(artifacts)
               if item.study == study and item.seed == seed
               and (direction is None or item.direction == direction)
               and (condition is None or item.condition == condition)
               and (k is None or item.k == k) and (q is None or item.q == q)]
    if len(entries) != 1:
        raise ValueError(f"Expected one checkpoint, found {len(entries)}. "
                         "List valid choices with python -m prometa.checkpoints --study " + study)
    return entries[0]


def resolve_checkpoint(artifacts: str | Path = "artifacts", **selection) -> Path:
    """Return a checked local path, verifying its bytes against the archived SHA-256."""
    entry = select_checkpoint(artifacts, **selection)
    path = Path(artifacts).resolve() / entry.checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Extract the artifact archive first.")
    if _digest(path) != entry.sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    return path


def load_checkpoint(artifacts: str | Path = "artifacts", *, device: str = "cpu", **selection):
    """Load a frozen model with the architecture version used by the selected study."""
    from .release_io import load_model
    entry = select_checkpoint(artifacts, **selection)
    namespace = None
    if entry.namespace == "rare":
        from .rare import model as namespace
    return load_model(resolve_checkpoint(artifacts, **selection), device, namespace=namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--study", help="Optional filter: transfer/kcurve/poi/linker/families/cdk6/efficiency/rare_e3")
    parser.add_argument("--verify", action="store_true", help="Verify every listed file's SHA-256")
    args = parser.parse_args()
    try:
        entries = checkpoint_catalog(args.artifacts)
        if args.study:
            study = "rare_e3" if args.study == "rare" else args.study
            entries = [entry for entry in entries if entry.study == study]
            if not entries:
                raise ValueError(f"No checkpoint entries for study {args.study!r}.")
        if args.verify:
            for entry in entries:
                path = args.artifacts / entry.checkpoint
                if not path.is_file() or _digest(path) != entry.sha256:
                    raise ValueError(f"Missing or modified checkpoint: {path}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps({"count": len(entries), "verified": args.verify,
                      "checkpoints": [asdict(entry) for entry in entries]}, indent=2))


if __name__ == "__main__":
    main()
