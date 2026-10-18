"""Portable I/O for the three separately versioned ProMeta checkpoint families."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import torch
from .experiment import build_experiment_data, model_forward
from .model import E3V2Config, E3V2Model

SEEDS = (42, 101, 202, 303, 404, 505, 606, 707, 2025, 3407)
PAPER_SEEDS = (42, 2025, 3407)
DIRECTIONS = (("CRBN", "VHL"), ("VHL", "CRBN"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):h.update(block)
    return h.hexdigest()


def training_data(artifacts, source, target):
    base = Path(artifacts) / "current_transfer/prometa_as/data-prometa"
    return build_experiment_data(
        contexts_path=base / "tables/strict_crbn_vhl_contexts_fair_esm_v1.csv",
        sequence_path=base / "tables/protac_with_seq_fair_esm_v1.csv",
        component_path=base / "tables/protac-fine_esm_complete_v1.csv",
        protein_embeddings_path=base / "embeddings/protein_embeddings_fair_v1.pt",
        source_e3=source, target_e3=target, supervised_train_scope="source_e3_only")


def load_model(checkpoint, device="cpu", namespace=None):
    if namespace is None:config_type, model_type = E3V2Config, E3V2Model
    else:config_type, model_type = namespace.E3V2Config, namespace.E3V2Model
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model_type(config_type(**saved["model_config"]))
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.to(device).eval()


def context_checkpoint(artifacts, source, target, seed):
    return Path(artifacts) / f"context_weights/checkpoints/{source}_to_{target}/seed_{seed}/grounded_interaction.pt"


def study_checkpoint(artifacts, study, condition, seed):
    """Resolve the named file used by a publication study."""
    return Path(artifacts) / "study_checkpoints" / study / condition / f"seed_{seed}" / "model.pt"


def source_reference(artifacts, data, source, target, seed):
    from .experiment import eligible_training_tasks, split_source_validation_tasks, stable_int
    path = Path(artifacts) / f"context_weights/training_records/{source}_to_{target}/seed_{seed}/run_manifest.json"
    record = json.loads(path.read_text())
    assert sha256(context_checkpoint(artifacts, source, target, seed)) == record["artifact_sha256"]["checkpoint"]
    tasks = eligible_training_tasks(data.training_contexts, source_e3=source, k_shot=2, q_query=3)
    tasks, _, selection = split_source_validation_tasks(tasks, source_e3=source, seed=seed,
        fraction=float(record["settings"]["source_validation_fraction"]))
    assert selection == record["source_validation_selection"]
    ids, labels = [], []
    for task, frame in sorted(tasks.items()):
        for label in (1, 0):
            candidates = sorted(frame.loc[frame.Label_bin.eq(label), "row_id"].astype(str),
                key=lambda rid:(stable_int("k0_source_reference",seed,task,label,rid),rid))
            assert len(candidates) >= 5
            ids.extend(candidates[:5]);labels.extend([label] * 5)
    assert set(data.row_lookup.loc[ids,"canonical_e3"]) == {source}
    return ids, labels


@torch.no_grad()
def encode(model, ids, data, source, seed, device="cpu"):
    return model_forward(model, ids, data, device=torch.device(device), intervention="correct",
                         source_e3=source, seed=seed, diagnostics=False)[0]


def select(features, indices):
    return {k:features[k][indices] for k in ("embedding","metric_scale") if features.get(k) is not None}


@torch.no_grad()
def predict_episode(model, support, labels, query):
    y = torch.as_tensor(labels,dtype=torch.long,device=support["embedding"].device)
    result = model.score_episode(support,y,query)
    return result["logits"].softmax(1)[:,result["classes"].tolist().index(1)].cpu().numpy()


def attach_family_contexts(data, cohort):
    from .data import prepare_contexts, build_context_graph_index
    frame = pd.DataFrame({"Compound ID":"CASE_" + cohort.compound_id.astype(str),
        "Uniprot":cohort.uniprot,"Target":cohort.target,"E3 ligase":cohort.e3,
        "Smiles":cohort.smiles,"Label_bin":cohort.label,
        "e3 ligand smiles":cohort.e3_ligand_smiles})
    prepared = prepare_contexts(frame)
    graphs = build_context_graph_index(prepared,invalid_molecule="raise")
    assert not set(graphs.graphs) & set(data.graph_index)
    data.graph_index.update(graphs.graphs)
    data.row_lookup = pd.concat([data.row_lookup.reset_index(drop=True),prepared],ignore_index=True).set_index("row_id",drop=False)
    cohort = cohort.copy();cohort["row_id"] = prepared.row_id.tolist()
    return cohort


def linker_data(artifacts, frame):
    from .data import smiles_to_graph
    from .experiment import ExperimentData
    bank = torch.load(Path(artifacts)/"current_transfer/prometa_as/data-prometa/embeddings/protein_embeddings_fair_v1.pt",map_location="cpu",weights_only=False)
    assert not set(frame.Uniprot)-set(bank)
    graphs, audits = {}, []
    for row in frame.itertuples():
        graph, audit = smiles_to_graph(row.Smiles,e3_ligand_smiles=row.canonical_e3_ligand)
        graph.row_id = row.row_id;graph.y = torch.tensor([int(row.label)])
        graphs[row.row_id] = graph
        audits.append(dict(row_id=row.row_id,matched=bool(audit.matched)))
    return ExperimentData(strict_contexts=frame,training_contexts=frame,graph_index=graphs,
        graph_audit=pd.DataFrame(audits),row_lookup=frame.set_index("row_id"),protein_bank=bank,
        e3_uniprot={"CRBN":"Q96SW2","VHL":"P40337"},protein_dim=1280,
        audit={"external_linker_benchmark":True})
