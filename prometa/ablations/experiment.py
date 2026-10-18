"""Training and evaluation engine for the isolated E3-v2 experiment matrix."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import re
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import Tensor
from torch_geometric.data import Batch, Data

from .data import (
    ATOM_FEATURE_NAMES,
    BOND_FEATURE_NAMES,
    ContextColumns,
    attach_context_components,
    build_context_component_index,
    build_context_graph_index,
    canonicalize_e3,
    context_key,
    prepare_contexts,
    smiles_to_graph,
    stable_context_id,
)
from .model import E3V2Config, E3V2Model, VALID_VARIANTS


E3_UNIPROT_FALLBACK = {
    "CRBN": "Q96SW2",
    "VHL": "P40337",
    "cIAP1": "Q13490",
    "FEM1B": "Q9UK73",
    "MDM2": "Q00987",
    "XIAP": "P98170",
}


@dataclass(frozen=True)
class EpisodeSpec:
    task: str
    e3: str
    support_ids: tuple[str, ...]
    support_labels: tuple[int, ...]
    query_ids: tuple[str, ...]
    query_labels: tuple[int, ...]


@dataclass
class ExperimentData:
    strict_contexts: pd.DataFrame
    training_contexts: pd.DataFrame
    graph_index: dict[str, Data]
    graph_audit: pd.DataFrame
    row_lookup: pd.DataFrame
    protein_bank: dict[str, Tensor]
    e3_uniprot: dict[str, str]
    protein_dim: int
    audit: dict[str, Any]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software_versions() -> dict[str, Any]:
    """Return the numerical software stack recorded in every run signature."""

    def distribution(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "torch_geometric": distribution("torch-geometric"),
        "scikit_learn": distribution("scikit-learn"),
        "rdkit": distribution("rdkit"),
    }


def experiment_code_hashes() -> dict[str, str]:
    """Hash every local module that defines data, model, or experiment semantics."""

    project_root = Path(__file__).resolve().parents[2]
    module_paths = [
        Path(__file__),
        Path(__file__).with_name("model.py"),
        Path(__file__).with_name("data.py"),
        project_root / "experiments" / "ablation_variant.py",
        project_root / "reproduce.py",
    ]
    return {path.name: sha256_file(path) for path in module_paths}


def stable_int(*parts: Any) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def normalize_accession(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    if "|" in text:
        fields = text.split("|")
        if len(fields) >= 2:
            text = fields[1]
    return text.split(";")[0].split(",")[0].strip()


def load_protein_bank(path: Path) -> tuple[dict[str, Tensor], int]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"Protein embedding bank is not a non-empty mapping: {path}")
    bank: dict[str, Tensor] = {}
    dimensions: set[int] = set()
    for key, value in raw.items():
        accession = normalize_accession(key)
        tensor = torch.as_tensor(value, dtype=torch.float32).flatten().cpu()
        if accession is None or tensor.numel() == 0:
            continue
        bank[accession] = tensor
        dimensions.add(int(tensor.numel()))
    if len(dimensions) != 1:
        raise ValueError(f"Protein embedding dimensions are inconsistent: {dimensions}")
    return bank, dimensions.pop()


def infer_e3_uniprot(component_table: pd.DataFrame) -> dict[str, str]:
    result = dict(E3_UNIPROT_FALLBACK)
    required = {"E3 ligase", "E3 ligase Uniprot"}
    if not required.issubset(component_table.columns):
        return result
    rows = component_table.dropna(subset=list(required))
    observed: dict[str, set[str]] = {}
    for _, row in rows.iterrows():
        e3 = canonicalize_e3(row["E3 ligase"])
        accession = normalize_accession(row["E3 ligase Uniprot"])
        if accession:
            observed.setdefault(e3, set()).add(accession)
    for e3, accessions in observed.items():
        if len(accessions) == 1:
            result[e3] = next(iter(accessions))
    return result


def deduplicate_labeled_contexts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = ContextColumns()
    needed = [
        columns.compound_id,
        columns.target,
        columns.e3,
        columns.smiles,
        columns.label,
    ]
    filtered = frame.dropna(subset=needed).copy()
    filtered[columns.label] = pd.to_numeric(filtered[columns.label], errors="coerce")
    filtered = filtered[filtered[columns.label].isin((0, 1))].copy()
    prepared = prepare_contexts(filtered, require_unique=False)
    kept: list[pd.Series] = []
    conflicts = 0
    duplicates = 0
    for _, group in prepared.groupby("row_id", sort=False):
        duplicates += max(0, len(group) - 1)
        if group[columns.label].nunique() != 1 or group[columns.smiles].nunique() != 1:
            conflicts += 1
            continue
        kept.append(group.iloc[0])
    if not kept:
        raise ValueError("No unambiguous labelled contexts remain")
    output = pd.DataFrame(kept).reset_index(drop=True)
    return output, {
        "input_rows": int(len(frame)),
        "complete_binary_rows": int(len(filtered)),
        "duplicate_rows_removed": int(duplicates),
        "conflicting_contexts_dropped": int(conflicts),
        "retained_contexts": int(len(output)),
    }


def build_experiment_data(
    *,
    contexts_path: Path,
    sequence_path: Path,
    component_path: Path,
    protein_embeddings_path: Path,
    source_e3: str,
    target_e3: str,
    supervised_train_scope: str = "all_non_target_e3",
) -> ExperimentData:
    source_e3 = canonicalize_e3(source_e3)
    target_e3 = canonicalize_e3(target_e3)
    strict_raw = pd.read_csv(contexts_path, low_memory=False)
    required_context_fields = ["Compound ID", "Target", "E3 ligase", "Smiles", "Label_bin"]
    strict_complete = strict_raw.dropna(subset=required_context_fields).copy()
    strict_invalid_key_rows = int(len(strict_raw) - len(strict_complete))
    strict = prepare_contexts(strict_complete)
    sequence_raw = pd.read_csv(sequence_path, low_memory=False)
    labelled, dedup_audit = deduplicate_labeled_contexts(sequence_raw)
    component_raw = pd.read_csv(component_path, low_memory=False)
    component_index = build_context_component_index(component_raw, on_ambiguous="drop")
    strict = attach_context_components(strict, component_index)

    rare = labelled[
        ~labelled["canonical_e3"].isin({source_e3, target_e3})
    ].copy()
    if not rare.empty:
        rare = attach_context_components(rare, component_index)
    scope_aliases = {
        "all_non_target_e3_with_source_only_validation": "all_non_target_e3",
        "all_non_target_e3": "all_non_target_e3",
        "source_e3_only": "source_e3_only",
    }
    try:
        resolved_train_scope = scope_aliases[str(supervised_train_scope)]
    except KeyError as error:
        raise ValueError(
            "supervised_train_scope must be source_e3_only or all_non_target_e3; "
            f"got {supervised_train_scope!r}"
        ) from error
    source_training = strict[strict["canonical_e3"].eq(source_e3)]
    if resolved_train_scope == "source_e3_only":
        training = source_training.copy()
    else:
        training = pd.concat(
            [source_training, rare],
            ignore_index=True,
            sort=False,
        )
    if training["canonical_e3"].eq(target_e3).any():
        raise AssertionError("Target E3 leaked into supervised training rows")
    if training["row_id"].duplicated().any():
        raise AssertionError("Training row IDs are not unique")

    all_rows = pd.concat([strict, rare], ignore_index=True, sort=False)
    all_rows = all_rows.drop_duplicates("row_id", keep="first").reset_index(drop=True)
    graph_bundle = build_context_graph_index(all_rows, invalid_molecule="drop")
    graph_ids = set(graph_bundle.graphs)
    all_rows = all_rows[all_rows["row_id"].isin(graph_ids)].copy()
    training = training[training["row_id"].isin(graph_ids)].copy()
    strict = strict[strict["row_id"].isin(graph_ids)].copy()
    bank, protein_dim = load_protein_bank(protein_embeddings_path)
    e3_uniprot = infer_e3_uniprot(component_raw)
    e3_uniprot = {
        e3: accession for e3, accession in e3_uniprot.items() if accession in bank
    }
    if source_e3 not in e3_uniprot or target_e3 not in e3_uniprot:
        raise ValueError(
            f"Missing frozen E3 embedding: source={source_e3 in e3_uniprot}, "
            f"target={target_e3 in e3_uniprot}"
        )

    row_lookup = all_rows.set_index("row_id", drop=False)
    poi_available = all_rows["Uniprot"].map(normalize_accession).isin(bank)
    training_poi = training["Uniprot"].map(normalize_accession).isin(bank)
    graph_audit = graph_bundle.audit
    match_count = int(
        graph_audit["e3_ligand_match_status"].astype(str).str.startswith("matched_").sum()
    )
    audit = {
        "strict_rows": int(len(strict)),
        "strict_invalid_key_rows_excluded": strict_invalid_key_rows,
        "supervised_training_rows": int(len(training)),
        "training_e3_counts": training["canonical_e3"].value_counts().to_dict(),
        "supervised_train_scope": resolved_train_scope,
        "non_source_non_target_rows_available": int(len(rare)),
        "non_source_non_target_rows_used": int(
            len(rare) if resolved_train_scope == "all_non_target_e3" else 0
        ),
        "rare_labelled_deduplication": dedup_audit,
        "graph_rows": int(len(graph_bundle.graphs)),
        "invalid_graph_rows": int(len(graph_bundle.audit) - len(graph_bundle.graphs)),
        "e3_ligand_mask_matches": match_count,
        "e3_ligand_mask_coverage": float(match_count / max(len(graph_audit), 1)),
        "protein_embedding_entries": int(len(bank)),
        "protein_embedding_dim": int(protein_dim),
        "poi_embedding_coverage_all": float(poi_available.mean()),
        "poi_embedding_coverage_training": float(training_poi.mean()),
        "e3_embedding_map": e3_uniprot,
        "target_e3_excluded_from_supervised_training": True,
    }
    return ExperimentData(
        strict_contexts=strict.reset_index(drop=True),
        training_contexts=training.reset_index(drop=True),
        graph_index=graph_bundle.graphs,
        graph_audit=graph_audit,
        row_lookup=row_lookup,
        protein_bank=bank,
        e3_uniprot=e3_uniprot,
        protein_dim=protein_dim,
        audit=audit,
    )


def eligible_training_tasks(
    frame: pd.DataFrame,
    *,
    source_e3: str,
    k_shot: int,
    q_query: int,
) -> dict[str, pd.DataFrame]:
    required = int(k_shot) + int(q_query)
    source_e3 = canonicalize_e3(source_e3)
    tasks: dict[str, pd.DataFrame] = {}
    primary = frame[frame["canonical_e3"].eq(source_e3)]
    for target, group in primary.groupby("Target", sort=True):
        counts = group["Label_bin"].value_counts()
        if int(counts.get(0, 0)) >= required and int(counts.get(1, 0)) >= required:
            tasks[f"{source_e3}::{target}"] = group.reset_index(drop=True)
    for e3, group in frame[~frame["canonical_e3"].eq(source_e3)].groupby(
        "canonical_e3", sort=True
    ):
        counts = group["Label_bin"].value_counts()
        if int(counts.get(0, 0)) >= required and int(counts.get(1, 0)) >= required:
            tasks[f"{e3}::__aggregate__"] = group.reset_index(drop=True)
    if not tasks:
        raise RuntimeError("No K+Q eligible supervised training task")
    return tasks


def split_source_validation_tasks(
    tasks: Mapping[str, pd.DataFrame],
    *,
    source_e3: str,
    seed: int,
    fraction: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    """Reserve deterministic source-E3 target tasks for source-only selection.

    Rare-E3 aggregate tasks remain training-only: they are too small and mix
    targets, so they must not determine the selected epoch.  When fewer than
    two source tasks are available, validation is disabled rather than leaking
    target-E3 information into model selection.
    """

    if not 0.0 <= fraction < 1.0:
        raise ValueError("source validation fraction must be in [0, 1)")
    source_name = canonicalize_e3(source_e3)
    source_prefix = f"{source_name}::"
    source_tasks = sorted(name for name in tasks if name.startswith(source_prefix))
    if fraction == 0.0 or len(source_tasks) < 2:
        return dict(tasks), {}, {
            "enabled": False,
            "reason": "disabled_or_fewer_than_two_source_tasks",
            "train_tasks": sorted(tasks),
            "validation_tasks": [],
        }
    count = max(1, int(round(len(source_tasks) * fraction)))
    count = min(count, len(source_tasks) - 1)
    ranked = sorted(
        source_tasks,
        key=lambda name: (stable_int("source_validation_task", seed, name), name),
    )
    validation_names = set(ranked[:count])
    training = {name: frame for name, frame in tasks.items() if name not in validation_names}
    validation = {name: tasks[name] for name in sorted(validation_names)}

    def row_digest(selected: Mapping[str, pd.DataFrame]) -> str:
        payload = {
            name: sorted(frame["row_id"].astype(str).tolist())
            for name, frame in sorted(selected.items())
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    return training, validation, {
        "enabled": True,
        "fraction": float(fraction),
        "selection_scope": "source_e3_target_tasks_only",
        "train_tasks": sorted(training),
        "validation_tasks": sorted(validation),
        "train_row_sha256": row_digest(training),
        "validation_row_sha256": row_digest(validation),
        "target_e3_used_for_selection": False,
    }


def build_episode_specs(
    tasks: Mapping[str, pd.DataFrame],
    *,
    count: int,
    seed: int,
    k_shot: int,
    q_query: int,
) -> tuple[list[EpisodeSpec], str]:
    names = sorted(tasks)
    rng = random.Random(stable_int("e3_v2_training_episodes", seed, k_shot, q_query))
    order: list[str] = []
    while len(order) < count:
        block = list(names)
        rng.shuffle(block)
        order.extend(block)
    specs: list[EpisodeSpec] = []
    for index, name in enumerate(order[:count]):
        task = tasks[name]
        selected: dict[int, list[str]] = {}
        for label in (1, 0):
            candidates = sorted(task.loc[task["Label_bin"].eq(label), "row_id"])
            local = random.Random(stable_int(seed, name, index, label))
            selected[label] = local.sample(candidates, k_shot + q_query)
        support_ids = tuple(selected[1][:k_shot] + selected[0][:k_shot])
        query_ids = tuple(selected[1][k_shot:] + selected[0][k_shot:])
        specs.append(
            EpisodeSpec(
                task=name,
                e3=str(task["canonical_e3"].iloc[0]),
                support_ids=support_ids,
                support_labels=tuple([1] * k_shot + [0] * k_shot),
                query_ids=query_ids,
                query_labels=tuple([1] * q_query + [0] * q_query),
            )
        )
    payload = [asdict(spec) for spec in specs]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return specs, digest


def protein_feature(
    bank: Mapping[str, Tensor], accession: Any, dimension: int
) -> tuple[Tensor, bool]:
    key = normalize_accession(accession)
    if key is not None:
        feature = bank.get(key)
        if feature is None and "-" in key:
            feature = bank.get(key.split("-")[0])
        if feature is not None:
            return feature, True
    return torch.zeros(dimension, dtype=torch.float32), False


def shuffled_e3(
    true_e3: str,
    catalog: Sequence[str],
    seed: int,
) -> str:
    candidates = [value for value in catalog if value != true_e3]
    if not candidates:
        raise ValueError("A shuffled E3 intervention requires a donor E3")
    # One donor per biological E3 and run seed keeps every support/query row in
    # an episode under the same counterfactual context.  Row-wise donors would
    # create a biologically incoherent pseudo-episode.
    return candidates[stable_int("shuffle_e3", seed, true_e3) % len(candidates)]


def model_graph(graph: Data) -> Data:
    """Strip heterogeneous audit strings before PyG collation."""

    match_ok = bool(getattr(graph, "e3_ligand_match_ok", False))
    ligand_mask = getattr(graph, "e3_ligand_mask", None)
    if not match_ok:
        # Absence of a validated component annotation means atom attention,
        # not an explicitly empty ligand.  Use every atom while retaining the
        # true component-coverage audit separately in graph_audit.csv.
        ligand_mask = torch.ones(graph.x.size(0), dtype=torch.bool)
    return Data(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=getattr(graph, "edge_attr", None),
        e3_ligand_mask=ligand_mask,
        # Keep the fallback and the annotation audit separate.  The grounded
        # component branch is allowed to use a mask only when this flag is true.
        e3_ligand_match_ok=torch.tensor([match_ok], dtype=torch.bool),
    )


def make_batch(
    row_ids: Sequence[str],
    data: ExperimentData,
    *,
    device: torch.device,
    intervention: str,
    source_e3: str,
    seed: int,
    e3_catalog: Sequence[str] | None = None,
) -> tuple[Batch, Tensor, Tensor, dict[str, Any]]:
    graphs: list[Data] = []
    poi_features: list[Tensor] = []
    e3_features: list[Tensor] = []
    poi_present = 0
    biological_e3s: list[str] = []
    model_e3s: list[str] = []
    catalog = sorted(
        {
            canonicalize_e3(value)
            for value in (
                data.e3_uniprot if e3_catalog is None else e3_catalog
            )
            if canonicalize_e3(value) in data.e3_uniprot
        }
    )
    if not catalog:
        raise ValueError("The E3 intervention catalog is empty")
    for row_id in row_ids:
        row = data.row_lookup.loc[str(row_id)]
        graphs.append(model_graph(data.graph_index[str(row_id)]))
        poi, present = protein_feature(
            data.protein_bank, row.get("Uniprot"), data.protein_dim
        )
        poi_features.append(poi)
        poi_present += int(present)
        true_e3 = str(row["canonical_e3"])
        if intervention == "correct":
            model_e3 = true_e3
        elif intervention == "wrong_e3":
            model_e3 = canonicalize_e3(source_e3)
            if model_e3 == true_e3:
                model_e3 = shuffled_e3(true_e3, catalog, seed)
        elif intervention == "shuffled_e3":
            model_e3 = shuffled_e3(true_e3, catalog, seed)
        else:
            raise ValueError(f"Unknown E3 intervention: {intervention}")
        accession = data.e3_uniprot.get(model_e3)
        e3, e3_present = protein_feature(data.protein_bank, accession, data.protein_dim)
        if not e3_present:
            raise ValueError(f"No E3 feature for intervention donor {model_e3}")
        e3_features.append(e3)
        biological_e3s.append(true_e3)
        model_e3s.append(model_e3)
    graph_batch = Batch.from_data_list(graphs).to(device)
    return (
        graph_batch,
        torch.stack(poi_features).to(device),
        torch.stack(e3_features).to(device),
        {
            "poi_present": poi_present,
            "n": len(row_ids),
            "biological_e3": biological_e3s,
            "model_e3": model_e3s,
        },
    )


def model_forward(
    model: E3V2Model,
    row_ids: Sequence[str],
    data: ExperimentData,
    *,
    device: torch.device,
    intervention: str,
    source_e3: str,
    seed: int,
    diagnostics: bool = False,
    e3_catalog: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph_batch, poi, e3, batch_audit = make_batch(
        row_ids,
        data,
        device=device,
        intervention=intervention,
        source_e3=source_e3,
        seed=seed,
        e3_catalog=e3_catalog,
    )
    kwargs: dict[str, Any] = {
        "poi_features": poi,
        "return_diagnostics": diagnostics,
    }
    if model.config.variant in {"grounded_constant_e3", "grounded_poi_control"}:
        # Keep the full grounded architecture active without exposing E3
        # identity.  The source-E3 feature is a real, training-side protein
        # representation and is held constant for every train/eval row.
        fixed_e3 = canonicalize_e3(source_e3)
        accession = data.e3_uniprot.get(fixed_e3)
        constant, present = protein_feature(
            data.protein_bank,
            accession,
            data.protein_dim,
        )
        if not present:
            raise ValueError(
                f"No source-E3 feature for grounded control {fixed_e3!r}"
            )
        kwargs["e3_features"] = constant.to(device).unsqueeze(0).expand(
            len(row_ids), -1
        )
        batch_audit["model_e3_input_policy"] = "fixed_source_e3_control"
        batch_audit["effective_model_e3"] = [fixed_e3] * len(row_ids)
    elif model.config.variant != "no_e3":
        kwargs["e3_features"] = e3
    return model(graph_batch, **kwargs), batch_audit


def e3_gradient_norms(model: E3V2Model) -> dict[str, float]:
    # Prefixes are matched on module boundaries.  Plain substring matching
    # would count ``grounded_e3_low_rank`` in the legacy ``e3_low_rank`` group
    # and would double-count the shared global interaction projection.
    groups = {
        "e3_projection": ("e3_projection", "e3_scale"),
        "film": ("film",),
        "conditional_metric": ("metric_conditioner",),
        "atom_attention": ("atom_attention",),
        "interaction": ("e3_low_rank",),
        "grounded_e3_adapter": (
            "e3_precomputed_protein_adapter",
            "e3_protein_encoder",
        ),
        "grounded_interaction": (
            "grounded_e3_input_norm",
            "grounded_e3_low_rank",
            "grounded_e3_code_norm",
            "grounded_recruiter_e3_projection",
            "grounded_compatibility_head",
        ),
    }
    output: dict[str, float] = {}
    for group, needles in groups.items():
        squared = 0.0
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in needles
            ):
                squared += float(parameter.grad.detach().square().sum().cpu())
        output[group] = math.sqrt(squared)
    output["e3_total"] = math.sqrt(sum(value * value for value in output.values()))
    return output


@torch.no_grad()
def validation_episode_loss(
    model: E3V2Model,
    data: ExperimentData,
    specs: Sequence[EpisodeSpec],
    *,
    device: torch.device,
    source_e3: str,
    seed: int,
) -> float:
    if not specs:
        return float("nan")
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    losses: list[float] = []
    for spec in specs:
        support, _ = model_forward(
            model,
            spec.support_ids,
            data,
            device=device,
            intervention="correct",
            source_e3=source_e3,
            seed=seed,
        )
        query, _ = model_forward(
            model,
            spec.query_ids,
            data,
            device=device,
            intervention="correct",
            source_e3=source_e3,
            seed=seed,
        )
        support_labels = torch.tensor(spec.support_labels, device=device)
        query_labels = torch.tensor(spec.query_labels, device=device)
        logits = model.score_episode(support, support_labels, query)["logits"]
        losses.append(float(loss_function(logits, query_labels).cpu()))
    return float(np.mean(losses))


def train_episodic(
    model: E3V2Model,
    data: ExperimentData,
    specs: Sequence[EpisodeSpec],
    *,
    epochs: int,
    episodes_per_epoch: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    source_e3: str,
    seed: int,
    compatibility_loss_weight: float = 0.0,
    compatibility_e3_catalog: Sequence[str] | None = None,
    validation_specs: Sequence[EpisodeSpec] = (),
    early_stopping_patience: int = 0,
    validation_min_delta: float = 0.0,
) -> dict[str, Any]:
    if compatibility_loss_weight < 0.0:
        raise ValueError("compatibility_loss_weight cannot be negative")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    first_gradients: dict[str, float] | None = None
    compatibility_catalog = sorted(
        {canonicalize_e3(value) for value in (compatibility_e3_catalog or [])}
    )
    compatibility_enabled = (
        compatibility_loss_weight > 0.0
        and model.config.variant == "grounded_interaction"
        and len(compatibility_catalog) >= 2
    )
    best_state: dict[str, Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch: int | None = None
    stale_epochs = 0
    set_seed(seed)
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        activity_losses: list[float] = []
        compatibility_losses: list[float] = []
        compatibility_rows = 0
        start = epoch * episodes_per_epoch
        for spec in specs[start : start + episodes_per_epoch]:
            support, _ = model_forward(
                model,
                spec.support_ids,
                data,
                device=device,
                intervention="correct",
                source_e3=source_e3,
                seed=seed,
            )
            query, _ = model_forward(
                model,
                spec.query_ids,
                data,
                device=device,
                intervention="correct",
                source_e3=source_e3,
                seed=seed,
            )
            support_labels = torch.tensor(spec.support_labels, device=device)
            query_labels = torch.tensor(spec.query_labels, device=device)
            scored = model.score_episode(support, support_labels, query)
            activity_loss = loss_function(scored["logits"], query_labels)
            compatibility_loss = activity_loss.new_zeros(())
            if compatibility_enabled:
                negative_query, _ = model_forward(
                    model,
                    spec.query_ids,
                    data,
                    device=device,
                    intervention="wrong_e3",
                    source_e3=source_e3,
                    seed=seed,
                    e3_catalog=compatibility_catalog,
                )
                positive_logit = query.get("compatibility_logit")
                negative_logit = negative_query.get("compatibility_logit")
                positive_valid = query.get("component_valid")
                negative_valid = negative_query.get("component_valid")
                if (
                    isinstance(positive_logit, Tensor)
                    and isinstance(negative_logit, Tensor)
                    and isinstance(positive_valid, Tensor)
                    and isinstance(negative_valid, Tensor)
                ):
                    valid = positive_valid.bool() & negative_valid.bool()
                    if valid.any():
                        n_valid = int(valid.sum().item())
                        logits = torch.cat(
                            [positive_logit[valid], negative_logit[valid]], dim=0
                        )
                        labels = torch.cat(
                            [
                                torch.ones(n_valid, device=device),
                                torch.zeros(n_valid, device=device),
                            ],
                            dim=0,
                        )
                        compatibility_loss = F.binary_cross_entropy_with_logits(
                            logits, labels
                        )
                        compatibility_rows += n_valid
                        compatibility_losses.append(
                            float(compatibility_loss.detach().cpu())
                        )
            loss = activity_loss + compatibility_loss_weight * compatibility_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if first_gradients is None:
                first_gradients = e3_gradient_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            activity_losses.append(float(activity_loss.detach().cpu()))
        validation_loss = validation_episode_loss(
            model,
            data,
            validation_specs,
            device=device,
            source_e3=source_e3,
            seed=seed,
        )
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "activity_loss": float(np.mean(activity_losses)),
            "compatibility_loss": (
                float(np.mean(compatibility_losses))
                if compatibility_losses
                else None
            ),
            "compatibility_rows": int(compatibility_rows),
            "source_validation_loss": (
                None if not math.isfinite(validation_loss) else validation_loss
            ),
        }
        history.append(record)
        if math.isfinite(validation_loss):
            if validation_loss < best_validation_loss - validation_min_delta:
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            print(json.dumps({"phase": "episodic", **record}), flush=True)
        if (
            validation_specs
            and early_stopping_patience > 0
            and stale_epochs >= early_stopping_patience
        ):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "first_step_gradient_norms": first_gradients or {},
        "final_temperature": float(model.prototype_head.distance.temperature.detach().cpu()),
        "compatibility_loss_weight": float(compatibility_loss_weight),
        "compatibility_enabled": bool(compatibility_enabled),
        "compatibility_e3_catalog": compatibility_catalog,
        "best_epoch": best_epoch,
        "best_source_validation_loss": (
            None if best_epoch is None else best_validation_loss
        ),
        "epochs_completed": len(history),
        "selection_used_target_e3": False,
    }


def load_eval_episodes(
    split_path: Path,
    data: ExperimentData,
) -> tuple[dict[str, Any], list[EpisodeSpec]]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    episodes: list[EpisodeSpec] = []
    for task, values in payload.get("meta_test", {}).items():
        for index, episode in enumerate(values):
            converted: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {}
            for role in ("support", "query"):
                row_ids: list[str] = []
                labels: list[int] = []
                for example in episode[role]:
                    key = context_key(
                        example["Compound ID"],
                        example.get("Task", task),
                        example["E3 ligase"],
                    )
                    row_id = stable_context_id(key)
                    if row_id not in data.graph_index:
                        raise KeyError(f"Frozen split row has no graph: {key.as_tuple()}")
                    row_ids.append(row_id)
                    labels.append(int(example["Label_bin"]))
                converted[role] = (tuple(row_ids), tuple(labels))
            episodes.append(
                EpisodeSpec(
                    task=f"{task}#{index}",
                    e3=str(example["E3 ligase"]),
                    support_ids=converted["support"][0],
                    support_labels=converted["support"][1],
                    query_ids=converted["query"][0],
                    query_labels=converted["query"][1],
                )
            )
    return payload.get("meta", {}), episodes


def binary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    truth = frame["y_true"].to_numpy(dtype=int)
    score = frame["y_score"].to_numpy(dtype=float)
    prediction = (score >= 0.5).astype(int)
    return {
        "n_query": int(len(frame)),
        "n_positive": int(truth.sum()),
        "n_negative": int((1 - truth).sum()),
        "auroc": float(roc_auc_score(truth, score)) if len(np.unique(truth)) == 2 else np.nan,
        "auprc": float(average_precision_score(truth, score)) if truth.sum() else np.nan,
        "accuracy": float(accuracy_score(truth, prediction)),
    }


def intervention_roles(condition: str) -> tuple[str, str]:
    """Map a named diagnostic to independent support/query interventions."""

    aliases = {
        "correct": ("correct", "correct"),
        "wrong_e3": ("wrong_e3", "wrong_e3"),
        "both_wrong": ("wrong_e3", "wrong_e3"),
        "shuffled_e3": ("shuffled_e3", "shuffled_e3"),
        "both_shuffled": ("shuffled_e3", "shuffled_e3"),
        "support_wrong": ("wrong_e3", "correct"),
        "query_wrong": ("correct", "wrong_e3"),
        "support_shuffled": ("shuffled_e3", "correct"),
        "query_shuffled": ("correct", "shuffled_e3"),
    }
    try:
        return aliases[condition]
    except KeyError as error:
        raise ValueError(f"Unknown evaluation E3 condition: {condition}") from error


@torch.no_grad()
def evaluate_splits(
    model: E3V2Model,
    data: ExperimentData,
    *,
    split_paths: Sequence[Path],
    interventions: Sequence[str],
    device: torch.device,
    source_e3: str,
    seed: int,
    variant_name: str,
    intervention_e3_catalog: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    support_embedding_deltas: dict[str, list[float]] = {
        name: [] for name in interventions if name != "correct"
    }
    query_embedding_deltas: dict[str, list[float]] = {
        name: [] for name in interventions if name != "correct"
    }
    prototype_deltas: dict[str, list[float]] = {
        name: [] for name in interventions if name != "correct"
    }
    e3_code_deltas: dict[str, list[float]] = {
        name: [] for name in interventions if name != "correct"
    }
    mask_coverages: list[float] = []
    component_validity: list[bool] = []
    grounded_e3_raw_abs: list[float] = []
    compatibility_margins: dict[str, list[float]] = {
        name: [] for name in interventions if name != "correct"
    }
    for split_path in split_paths:
        meta, episodes = load_eval_episodes(split_path, data)
        for episode_index, episode in enumerate(episodes):
            reference_support, _ = model_forward(
                model,
                episode.support_ids,
                data,
                device=device,
                intervention="correct",
                source_e3=source_e3,
                seed=seed,
                diagnostics=True,
                e3_catalog=intervention_e3_catalog,
            )
            reference_query, _ = model_forward(
                model,
                episode.query_ids,
                data,
                device=device,
                intervention="correct",
                source_e3=source_e3,
                seed=seed,
                diagnostics=True,
                e3_catalog=intervention_e3_catalog,
            )
            diagnostics = reference_query.get("diagnostics", {})
            coverage = diagnostics.get("e3_ligand_mask_coverage")
            if isinstance(coverage, Tensor):
                mask_coverages.extend(coverage.detach().cpu().tolist())
            validity = reference_query.get("component_valid")
            if isinstance(validity, Tensor):
                component_validity.extend(validity.detach().cpu().bool().tolist())
            raw_e3 = diagnostics.get("grounded_e3_raw")
            if isinstance(raw_e3, Tensor):
                grounded_e3_raw_abs.extend(raw_e3.detach().abs().cpu().reshape(-1).tolist())
            support_labels = torch.tensor(episode.support_labels, device=device)
            reference_scored = model.score_episode(
                reference_support, support_labels, reference_query
            )
            for intervention in interventions:
                support_action, query_action = intervention_roles(intervention)
                if support_action == "correct":
                    support = reference_support
                else:
                    support, _ = model_forward(
                        model,
                        episode.support_ids,
                        data,
                        device=device,
                        intervention=support_action,
                        source_e3=source_e3,
                        seed=seed,
                        diagnostics=True,
                        e3_catalog=intervention_e3_catalog,
                    )
                if query_action == "correct":
                    query = reference_query
                else:
                    query, _ = model_forward(
                        model,
                        episode.query_ids,
                        data,
                        device=device,
                        intervention=query_action,
                        source_e3=source_e3,
                        seed=seed,
                        diagnostics=True,
                        e3_catalog=intervention_e3_catalog,
                    )
                if intervention != "correct":
                    support_delta = torch.norm(
                        support["embedding"] - reference_support["embedding"], dim=1
                    )
                    query_delta = torch.norm(
                        query["embedding"] - reference_query["embedding"], dim=1
                    )
                    support_embedding_deltas[intervention].extend(
                        support_delta.detach().cpu().tolist()
                    )
                    query_embedding_deltas[intervention].extend(
                        query_delta.detach().cpu().tolist()
                    )
                scored = model.score_episode(support, support_labels, query)
                if intervention != "correct":
                    prototype_delta = torch.norm(
                        scored["prototypes"] - reference_scored["prototypes"], dim=1
                    )
                    prototype_deltas[intervention].extend(
                        prototype_delta.detach().cpu().tolist()
                    )
                    reference_code = diagnostics.get("grounded_e3_code")
                    altered_code = query.get("diagnostics", {}).get(
                        "grounded_e3_code"
                    )
                    if isinstance(reference_code, Tensor) and isinstance(
                        altered_code, Tensor
                    ):
                        code_delta = torch.norm(altered_code - reference_code, dim=1)
                        e3_code_deltas[intervention].extend(
                            code_delta.detach().cpu().tolist()
                        )
                    correct_compatibility = reference_query.get("compatibility_logit")
                    altered_compatibility = query.get("compatibility_logit")
                    altered_validity = query.get("component_valid")
                    if (
                        isinstance(correct_compatibility, Tensor)
                        and isinstance(altered_compatibility, Tensor)
                        and isinstance(validity, Tensor)
                        and isinstance(altered_validity, Tensor)
                    ):
                        valid = validity.bool() & altered_validity.bool()
                        if valid.any():
                            margin = (
                                correct_compatibility[valid]
                                - altered_compatibility[valid]
                            )
                            compatibility_margins[intervention].extend(
                                margin.detach().cpu().tolist()
                            )
                probabilities = torch.softmax(scored["logits"], dim=1)
                classes = scored["classes"].detach().cpu().tolist()
                positive_index = classes.index(1)
                scores = probabilities[:, positive_index].detach().cpu().tolist()
                for row_id, label, score in zip(
                    episode.query_ids, episode.query_labels, scores
                ):
                    row = data.row_lookup.loc[row_id]
                    prediction_rows.append(
                        {
                            "variant": variant_name,
                            "intervention": intervention,
                            "direction": meta.get("direction"),
                            "source_e3": source_e3,
                            "target_e3": meta.get("target"),
                            "seed": seed,
                            "k_shot": int(meta.get("k_shot", 0)),
                            "q_query": int(meta.get("q_query", 0)),
                            "protocol_version": meta.get(
                                "protocol_version", "legacy"
                            ),
                            "support_intervention": support_action,
                            "query_intervention": query_action,
                            "task": episode.task,
                            "episode": episode_index,
                            "row_id": row_id,
                            "compound_id": str(row["Compound ID"]),
                            "y_true": int(label),
                            "y_score": float(score),
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    metric_rows: list[dict[str, Any]] = []
    grouping = [
        "variant",
        "intervention",
        "direction",
        "seed",
        "k_shot",
        "q_query",
        "protocol_version",
    ]
    for keys, group in predictions.groupby(grouping, dropna=False, sort=True):
        metric_rows.append({**dict(zip(grouping, keys)), **binary_metrics(group)})
    diagnostics = {
        "mean_embedding_l2_change": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in query_embedding_deltas.items()
        },
        "fraction_nonzero_embedding_change": {
            name: float(np.mean(np.asarray(values) > 1e-8)) if values else 0.0
            for name, values in query_embedding_deltas.items()
        },
        "mean_support_embedding_l2_change": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in support_embedding_deltas.items()
        },
        "mean_query_embedding_l2_change": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in query_embedding_deltas.items()
        },
        "mean_prototype_l2_change": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in prototype_deltas.items()
        },
        "mean_e3_code_l2_change": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in e3_code_deltas.items()
        },
        "mean_compatibility_margin_correct_minus_intervened": {
            name: float(np.mean(values)) if values else None
            for name, values in compatibility_margins.items()
        },
        "mean_atom_selection_fraction": (
            float(np.mean(mask_coverages)) if mask_coverages else None
        ),
        "verified_component_fraction": (
            float(np.mean(component_validity)) if component_validity else None
        ),
        "grounded_e3_raw_abs_median": (
            float(np.median(grounded_e3_raw_abs)) if grounded_e3_raw_abs else None
        ),
        "grounded_e3_raw_abs_max": (
            float(np.max(grounded_e3_raw_abs)) if grounded_e3_raw_abs else None
        ),
    }
    identity_columns = [
        "direction",
        "seed",
        "k_shot",
        "q_query",
        "task",
        "episode",
        "row_id",
    ]
    score_wide = predictions.pivot(
        index=identity_columns,
        columns="intervention",
        values="y_score",
    )
    score_changes: dict[str, float] = {}
    nonzero_score_changes: dict[str, float] = {}
    for intervention in interventions:
        if intervention == "correct" or intervention not in score_wide.columns:
            continue
        delta = (score_wide[intervention] - score_wide["correct"]).abs()
        score_changes[intervention] = float(delta.mean())
        nonzero_score_changes[intervention] = float((delta > 1e-8).mean())
    diagnostics["mean_absolute_score_change"] = score_changes
    diagnostics["fraction_nonzero_score_change"] = nonzero_score_changes
    return predictions, pd.DataFrame(metric_rows), diagnostics


def build_ctp_records(
    tuple_path: Path,
    data: ExperimentData,
    *,
    target_e3: str,
    seed: int,
    max_rows: int | None,
) -> tuple[list[tuple[Data, Tensor, str]], dict[str, Any]]:
    table = pd.read_csv(tuple_path, low_memory=False)
    required = ["Smiles", "Uniprot", "E3 ligase"]
    table = table.dropna(subset=required).copy()
    table["canonical_e3"] = table["E3 ligase"].map(canonicalize_e3)
    table["poi_accession"] = table["Uniprot"].map(normalize_accession)
    target_name = canonicalize_e3(target_e3)
    table = table[~table["canonical_e3"].eq(target_name)]
    target_rows = data.strict_contexts[
        data.strict_contexts["canonical_e3"].eq(target_name)
    ]
    forbidden_smiles = set(target_rows["Smiles"].dropna().astype(str))
    forbidden_compounds = set(target_rows["Compound ID"].astype(str))
    before_identity_filter = len(table)
    table = table[
        ~table["Smiles"].astype(str).isin(forbidden_smiles)
        & ~table["Compound ID"].astype(str).isin(forbidden_compounds)
    ]
    identity_leakage_rows_removed = int(before_identity_filter - len(table))
    table = table[table["canonical_e3"].isin(data.e3_uniprot)]
    table = table[table["poi_accession"].isin(data.protein_bank)]
    table = table.drop_duplicates(["Smiles", "poi_accession", "canonical_e3"])
    if max_rows is not None and len(table) > max_rows:
        table = table.sample(max_rows, random_state=seed).reset_index(drop=True)
    records: list[tuple[Data, Tensor, str]] = []
    invalid = 0
    for _, row in table.iterrows():
        try:
            graph, _ = smiles_to_graph(row["Smiles"])
        except Exception:
            invalid += 1
            continue
        records.append(
            (
                graph,
                data.protein_bank[str(row["poi_accession"])],
                str(row["canonical_e3"]),
            )
        )
    if len({record[2] for record in records}) < 2:
        raise RuntimeError("CTP requires at least two train-side E3 ligases")
    return records, {
        "candidate_rows": int(len(table)),
        "valid_graph_rows": int(len(records)),
        "invalid_smiles_rows": int(invalid),
        "target_identity_leakage_rows_removed": identity_leakage_rows_removed,
        "e3_counts": pd.Series([record[2] for record in records]).value_counts().to_dict(),
        "target_e3_excluded": target_name,
        "target_compound_and_smiles_excluded": True,
    }


def _ctp_e3_features(
    e3_names: Sequence[str], data: ExperimentData, device: torch.device
) -> Tensor:
    return torch.stack(
        [data.protein_bank[data.e3_uniprot[name]] for name in e3_names]
    ).to(device)


def pretrain_counterfactual_tuples(
    model: E3V2Model,
    data: ExperimentData,
    *,
    tuple_path: Path,
    target_e3: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    max_rows: int | None,
) -> dict[str, Any]:
    records, audit = build_ctp_records(
        tuple_path,
        data,
        target_e3=target_e3,
        seed=seed,
        max_rows=max_rows,
    )
    head = nn.Linear(model.config.embedding_dim, 1).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
        + list(head.parameters()),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    loss_function = nn.BCEWithLogitsLoss()
    catalog = sorted({record[2] for record in records})
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed + 7919)
    for epoch in range(epochs):
        model.train()
        head.train()
        permutation = torch.randperm(len(records), generator=generator).tolist()
        losses: list[float] = []
        negative_kind = ("e3_swap", "target_swap", "both_swap")[epoch % 3]
        for offset in range(0, len(permutation), batch_size):
            indices = permutation[offset : offset + batch_size]
            if len(indices) < 2:
                continue
            batch_records = [records[index] for index in indices]
            graphs = Batch.from_data_list(
                [model_graph(record[0]) for record in batch_records]
            ).to(device)
            poi = torch.stack([record[1] for record in batch_records]).to(device)
            e3_names = [record[2] for record in batch_records]
            e3 = _ctp_e3_features(e3_names, data, device)
            negative_poi = poi.roll(1, dims=0)
            negative_e3_names = [
                [candidate for candidate in catalog if candidate != name][
                    stable_int(seed, epoch, offset, position, name)
                    % (len(catalog) - 1)
                ]
                for position, name in enumerate(e3_names)
            ]
            negative_e3 = _ctp_e3_features(negative_e3_names, data, device)
            positive = model(graphs, poi_features=poi, e3_features=e3)["embedding"]
            negative = model(
                graphs,
                poi_features=negative_poi if negative_kind != "e3_swap" else poi,
                e3_features=negative_e3 if negative_kind != "target_swap" else e3,
            )["embedding"]
            logits = torch.cat([head(positive).view(-1), head(negative).view(-1)])
            labels = torch.cat(
                [torch.ones(len(indices), device=device), torch.zeros(len(indices), device=device)]
            )
            loss = loss_function(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {
            "epoch": epoch + 1,
            "negative_kind": negative_kind,
            "loss": float(np.mean(losses)),
        }
        history.append(record)
        print(json.dumps({"phase": "ctp", **record}), flush=True)
    return {
        **audit,
        "epochs": epochs,
        "batch_size": batch_size,
        "history": history,
        "counterfactual_schedule": "epoch-cycled e3_swap/target_swap/both_swap",
        "target_e3_seen": False,
    }


def resolve_split_paths(
    split_root: Path,
    source_e3: str,
    target_e3: str,
    seed: int,
    q_values: Iterable[int] | int,
    *,
    protocol: str = "legacy_s3_q",
    k_values: Iterable[int] | int | None = None,
) -> list[Path]:
    if isinstance(q_values, int):
        q_list = [int(q_values)]
    else:
        q_list = [int(value) for value in q_values]
    paths: list[Path] = []
    if protocol == "legacy_s3_q":
        for q_query in q_list:
            experiment = f"s3_{source_e3.lower()}_{target_e3.lower()}_k2q{q_query}"
            path = split_root / experiment / f"{experiment}_seed{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
        return paths
    if protocol != "e3_v3_nested_k_v1":
        raise ValueError(f"Unknown split protocol: {protocol}")
    if len(q_list) != 1:
        raise ValueError("e3_v3_nested_k_v1 requires one fixed q_query value")
    if k_values is None:
        raise ValueError("e3_v3_nested_k_v1 requires test_k")
    if isinstance(k_values, int):
        k_list = [int(k_values)]
    else:
        k_list = [int(value) for value in k_values]
    metadata: list[dict[str, Any]] = []
    for k_shot in k_list:
        path = (
            split_root
            / f"{source_e3.lower()}_to_{target_e3.lower()}"
            / f"seed_{seed}"
            / f"k{k_shot}_q{q_list[0]}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
        expected = {
            "protocol_version": protocol,
            "source": canonicalize_e3(source_e3),
            "target": canonicalize_e3(target_e3),
            "seed": int(seed),
            "k_shot": int(k_shot),
            "q_query": int(q_list[0]),
        }
        mismatches = {
            key: (meta.get(key), value)
            for key, value in expected.items()
            if meta.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Split metadata mismatch for {path}: {mismatches}")
        metadata.append(meta)
        paths.append(path)
    for field in ("task_universe_hash", "query_pool_hash"):
        values = {str(meta.get(field)) for meta in metadata}
        if len(values) != 1 or "None" in values:
            raise ValueError(
                f"Nested-K splits do not share one audited {field}: {values}"
            )
    return paths


def run_signature(
    *,
    variant_name: str,
    source_e3: str,
    target_e3: str,
    seed: int,
    settings: Mapping[str, Any],
    paths: Mapping[str, Path],
    split_paths: Sequence[Path],
    device: torch.device,
    software: Mapping[str, Any],
) -> str:
    file_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if path.is_file()
    }
    file_hashes.update(
        {f"split_{index}": sha256_file(path) for index, path in enumerate(split_paths)}
    )
    payload = {
        "variant": variant_name,
        "source_e3": canonicalize_e3(source_e3),
        "target_e3": canonicalize_e3(target_e3),
        "seed": int(seed),
        "settings": dict(settings),
        "device": str(device),
        "software": dict(software),
        "file_hashes": file_hashes,
        "code_hashes": experiment_code_hashes(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=json_default).encode("utf-8")
    ).hexdigest()


def run_experiment(
    *,
    variant_name: str,
    source_e3: str,
    target_e3: str,
    seed: int,
    device: torch.device,
    paths: Mapping[str, Path],
    output_dir: Path,
    checkpoint_path: Path,
    settings: Mapping[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    started = time.time()
    split_paths = resolve_split_paths(
        paths["split_root"],
        source_e3,
        target_e3,
        seed,
        settings["test_q"],
        protocol=str(settings.get("split_protocol", "legacy_s3_q")),
        k_values=settings.get("test_k"),
    )
    software = software_versions()
    signature = run_signature(
        variant_name=variant_name,
        source_e3=source_e3,
        target_e3=target_e3,
        seed=seed,
        settings=settings,
        paths=paths,
        split_paths=split_paths,
        device=device,
        software=software,
    )
    success = output_dir / "_SUCCESS"
    if success.is_file() and not force:
        existing = json.loads(
            (output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        if existing.get("run_signature") != signature:
            raise RuntimeError(
                "Completed run signature does not match the requested code, data, "
                "or protocol. Use a new versioned output directory (preferred) or "
                "rerun explicitly with force."
            )
        return existing
    if force and success.is_file():
        # This marker is generated by this runner; removing it prevents an
        # in-progress forced rerun from being mistaken for a complete result.
        success.unlink()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"Incomplete non-empty run directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    use_ctp = variant_name.endswith("_ctp")
    use_compatibility = variant_name.endswith("_compat")
    if use_ctp and use_compatibility:
        raise ValueError("A run variant cannot combine _ctp and _compat suffixes")
    if use_ctp:
        base_variant = variant_name.removesuffix("_ctp")
    elif use_compatibility:
        base_variant = variant_name.removesuffix("_compat")
    else:
        base_variant = variant_name
    if base_variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown variant: {variant_name}")
    set_seed(seed)
    data = build_experiment_data(
        contexts_path=paths["contexts"],
        sequence_path=paths["sequence_table"],
        component_path=paths["component_table"],
        protein_embeddings_path=paths["protein_embeddings"],
        source_e3=source_e3,
        target_e3=target_e3,
        supervised_train_scope=str(
            settings.get("supervised_train_scope", "all_non_target_e3")
        ),
    )
    model_config = E3V2Config(
        variant=base_variant,
        node_dim=len(ATOM_FEATURE_NAMES),
        edge_dim=len(BOND_FEATURE_NAMES),
        pretrained_protein_dim=data.protein_dim,
        distance="squared_euclidean",
        temperature=1.0,
        learnable_temperature=True,
    )
    model = E3V2Model(model_config).to(device)
    parameter_counts = {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    # Keep episode sampling/dropout streams paired after condition-specific modules exist.
    set_seed(seed)

    ctp_audit: dict[str, Any] | None = None
    if use_ctp:
        ctp_audit = pretrain_counterfactual_tuples(
            model,
            data,
            tuple_path=paths["tuple_table"],
            target_e3=target_e3,
            epochs=int(settings["ctp_epochs"]),
            batch_size=int(settings["ctp_batch_size"]),
            learning_rate=float(settings["ctp_learning_rate"]),
            seed=seed,
            device=device,
            max_rows=settings.get("max_ctp_rows"),
        )
        set_seed(seed)

    all_tasks = eligible_training_tasks(
        data.training_contexts,
        source_e3=source_e3,
        k_shot=int(settings["k_shot"]),
        q_query=int(settings["train_q"]),
    )
    tasks, validation_tasks, selection_audit = split_source_validation_tasks(
        all_tasks,
        source_e3=source_e3,
        seed=seed,
        fraction=float(settings.get("source_validation_fraction", 0.0)),
    )
    total_episodes = int(settings["epochs"]) * int(settings["episodes_per_epoch"])
    specs, episode_hash = build_episode_specs(
        tasks,
        count=total_episodes,
        seed=seed,
        k_shot=int(settings["k_shot"]),
        q_query=int(settings["train_q"]),
    )
    validation_specs: list[EpisodeSpec] = []
    validation_episode_hash: str | None = None
    if validation_tasks:
        validation_count = max(
            len(validation_tasks),
            len(validation_tasks)
            * int(settings.get("source_validation_episodes_per_task", 1)),
        )
        validation_specs, validation_episode_hash = build_episode_specs(
            validation_tasks,
            count=validation_count,
            seed=seed + 104729,
            k_shot=int(settings["k_shot"]),
            q_query=int(settings["train_q"]),
        )
    supervised_training_e3s = sorted(
        {
            str(frame["canonical_e3"].iloc[0])
            for frame in tasks.values()
            if not frame.empty
        }
    )
    compatibility_weight = (
        float(settings.get("compatibility_loss_weight", 0.0))
        if use_compatibility
        else 0.0
    )
    training_audit = train_episodic(
        model,
        data,
        specs,
        epochs=int(settings["epochs"]),
        episodes_per_epoch=int(settings["episodes_per_epoch"]),
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        device=device,
        source_e3=source_e3,
        seed=seed,
        compatibility_loss_weight=compatibility_weight,
        compatibility_e3_catalog=supervised_training_e3s,
        validation_specs=validation_specs,
        early_stopping_patience=int(settings.get("early_stopping_patience", 0)),
        validation_min_delta=float(settings.get("validation_min_delta", 0.0)),
    )

    predictions, metrics, evaluation_audit = evaluate_splits(
        model,
        data,
        split_paths=split_paths,
        interventions=settings["evaluation_e3_conditions"],
        device=device,
        source_e3=source_e3,
        seed=seed,
        variant_name=variant_name,
        intervention_e3_catalog=supervised_training_e3s,
    )
    predictions.to_csv(output_dir / "query_predictions.csv", index=False)
    metrics.to_csv(output_dir / "split_metrics.csv", index=False)
    data.graph_audit.to_csv(output_dir / "graph_and_ligand_mask_audit.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "variant_name": variant_name,
            "source_e3": source_e3,
            "target_e3": target_e3,
            "seed": seed,
        },
        checkpoint_path,
    )
    artifact_sha256 = {
        "checkpoint": sha256_file(checkpoint_path),
        "query_predictions": sha256_file(output_dir / "query_predictions.csv"),
        "split_metrics": sha256_file(output_dir / "split_metrics.csv"),
        "graph_and_ligand_mask_audit": sha256_file(
            output_dir / "graph_and_ligand_mask_audit.csv"
        ),
    }
    control_e3_policy: dict[str, Any] | None = None
    if base_variant in {"grounded_constant_e3", "grounded_poi_control"}:
        fixed_e3 = canonicalize_e3(source_e3)
        fixed_accession = data.e3_uniprot.get(fixed_e3)
        fixed_feature, fixed_present = protein_feature(
            data.protein_bank,
            fixed_accession,
            data.protein_dim,
        )
        if not fixed_present:  # pragma: no cover - guarded during every forward
            raise ValueError(f"No source-E3 control feature for {fixed_e3!r}")
        control_e3_policy = {
            "policy": "fixed_source_e3_feature_for_all_train_and_eval_rows",
            "e3": fixed_e3,
            "uniprot": fixed_accession,
            "feature_sha256": hashlib.sha256(
                fixed_feature.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
            "target_e3_feature_used": False,
            "full_grounded_forward_path_used": True,
        }
    manifest = {
        "format": "prometa_e3_v3_run_v1",
        "run_signature": signature,
        "variant": variant_name,
        "base_model_variant": base_variant,
        "source_e3": source_e3,
        "target_e3": target_e3,
        "seed": seed,
        "device": str(device),
        "software": software,
        "code_sha256": experiment_code_hashes(),
        "artifact_sha256": artifact_sha256,
        "settings": dict(settings),
        "model_config": asdict(model_config),
        "parameter_counts": parameter_counts,
        "control_e3_policy": control_e3_policy,
        "data_audit": data.audit,
        "eligible_training_tasks": {
            name: int(len(frame)) for name, frame in all_tasks.items()
        },
        "training_tasks": {name: int(len(frame)) for name, frame in tasks.items()},
        "source_validation_tasks": {
            name: int(len(frame)) for name, frame in validation_tasks.items()
        },
        "source_validation_selection": selection_audit,
        "training_episode_count": total_episodes,
        "training_episode_sha256": episode_hash,
        "source_validation_episode_sha256": validation_episode_hash,
        "training": training_audit,
        "counterfactual_pretraining": ctp_audit,
        "evaluation": evaluation_audit,
        "checkpoint": str(checkpoint_path),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
            if path.is_file()
        },
        "split_paths": [str(path) for path in split_paths],
        "elapsed_seconds": time.time() - started,
        "completed": True,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    success.write_text("ok\n", encoding="utf-8")
    return manifest


__all__ = [
    "ExperimentData",
    "EpisodeSpec",
    "build_experiment_data",
    "build_episode_specs",
    "evaluate_splits",
    "run_experiment",
]
