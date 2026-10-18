"""ProMeta CRBN+VHL multi-source training and frozen-weight rare-E3 inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from prometa.rare.data import ATOM_FEATURE_NAMES, BOND_FEATURE_NAMES, canonicalize_e3
from prometa.rare.experiment import EpisodeSpec, build_experiment_data, model_forward, set_seed, software_versions, train_episodic
from prometa.rare.model import E3V2Config, E3V2Model
SEEDS = (42, 101, 202, 303, 404, 505, 606, 707, 2025, 3407)
RARE_E3S = ('CIAP1', 'FEM1B', 'MDM2', 'XIAP')
ALL_E3S = tuple((canonicalize_e3(value) for value in ('CRBN', 'VHL', *RARE_E3S)))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def stable_int(*parts: Any) -> int:
    text = '||'.join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'big')

def context_map(data: Any) -> dict[str, str]:
    column = 'Context ID' if 'Context ID' in data.row_lookup else 'context_key_compound_e3_target'
    frame = data.row_lookup.dropna(subset=[column]).copy()
    frame[column] = frame[column].astype(str)
    if frame[column].duplicated().any():
        raise AssertionError('Context ID is not unique')
    return frame.set_index(column)['row_id'].astype(str).to_dict()

def all_context_rows(data: Any) -> pd.DataFrame:
    frame = data.row_lookup.copy()
    if frame.index.name == 'row_id' and 'row_id' not in frame.columns:
        frame = frame.reset_index()
    return frame.reset_index(drop=True)

def training_rows(data: Any, allowed_e3s: Sequence[str], target_e3: str | None) -> pd.DataFrame:
    allowed = {canonicalize_e3(value) for value in allowed_e3s}
    target = None if target_e3 is None else canonicalize_e3(target_e3)
    frame = all_context_rows(data)
    selected = frame.loc[frame['canonical_e3'].isin(allowed)].copy()
    if target is not None and selected['canonical_e3'].eq(target).any():
        raise AssertionError('Held-out target E3 leaked into supervised training')
    if selected['row_id'].duplicated().any():
        raise AssertionError('Training row IDs are not unique')
    return selected.reset_index(drop=True)

def eligible_multisource_tasks(frame: pd.DataFrame, *, k_shot: int, q_query: int) -> dict[str, dict[str, pd.DataFrame]]:
    required = k_shot + q_query
    output: dict[str, dict[str, pd.DataFrame]] = {}
    for e3, e3_frame in frame.groupby('canonical_e3', sort=True):
        tasks: dict[str, pd.DataFrame] = {}
        for target, group in e3_frame.groupby('Target', sort=True):
            counts = group['Label_bin'].value_counts()
            if int(counts.get(0, 0)) >= required and int(counts.get(1, 0)) >= required:
                tasks[f'{e3}::{target}'] = group.reset_index(drop=True)
        if not tasks:
            counts = e3_frame['Label_bin'].value_counts()
            if int(counts.get(0, 0)) >= required and int(counts.get(1, 0)) >= required:
                tasks[f'{e3}::__aggregate__'] = e3_frame.reset_index(drop=True)
        if tasks:
            output[str(e3)] = tasks
    if not output:
        raise RuntimeError('No eligible multi-source training tasks')
    return output

def split_validation_tasks(grouped: Mapping[str, Mapping[str, pd.DataFrame]], *, seed: int, fraction: float) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    training: dict[str, dict[str, pd.DataFrame]] = {}
    validation: dict[str, dict[str, pd.DataFrame]] = {}
    for e3, tasks in sorted(grouped.items()):
        names = sorted(tasks)
        splittable = [name for name in names if not name.endswith('::__aggregate__')]
        if e3 in {'CRBN', 'VHL'} and len(splittable) >= 2 and (fraction > 0):
            count = min(len(splittable) - 1, max(1, int(round(len(splittable) * fraction))))
            ranked = sorted(splittable, key=lambda name: (stable_int('validation', seed, name), name))
            held = set(ranked[:count])
        else:
            held = set()
        training[e3] = {name: frame for name, frame in tasks.items() if name not in held}
        if held:
            validation[e3] = {name: tasks[name] for name in sorted(held)}
    audit = {'train_tasks': {e3: sorted(tasks) for e3, tasks in training.items()}, 'validation_tasks': {e3: sorted(tasks) for e3, tasks in validation.items()}, 'validation_scope': 'held-out CRBN/VHL target tasks only', 'rare_e3_used_for_epoch_selection': False}
    return (training, validation, audit)

def build_balanced_specs(grouped: Mapping[str, Mapping[str, pd.DataFrame]], *, count: int, seed: int, k_shot: int, q_query: int) -> tuple[list[EpisodeSpec], str]:
    e3s = sorted((e3 for e3, tasks in grouped.items() if tasks))
    if not e3s:
        return ([], hashlib.sha256(b'[]').hexdigest())
    rng = random.Random(stable_int('balanced_e3_order', seed, k_shot, q_query))
    e3_order: list[str] = []
    while len(e3_order) < count:
        block = list(e3s)
        rng.shuffle(block)
        e3_order.extend(block)
    task_counts = {e3: 0 for e3 in e3s}
    specs: list[EpisodeSpec] = []
    for index, e3 in enumerate(e3_order[:count]):
        names = sorted(grouped[e3])
        local_task_rng = random.Random(stable_int('task', seed, e3, task_counts[e3]))
        name = names[local_task_rng.randrange(len(names))]
        task_counts[e3] += 1
        task = grouped[e3][name]
        chosen: dict[int, list[str]] = {}
        for label in (1, 0):
            candidates = sorted(task.loc[task['Label_bin'].eq(label), 'row_id'].astype(str))
            local = random.Random(stable_int('rows', seed, name, index, label))
            chosen[label] = local.sample(candidates, k_shot + q_query)
        specs.append(EpisodeSpec(task=name, e3=e3, support_ids=tuple(chosen[1][:k_shot] + chosen[0][:k_shot]), support_labels=tuple([1] * k_shot + [0] * k_shot), query_ids=tuple(chosen[1][k_shot:] + chosen[0][k_shot:]), query_labels=tuple([1] * q_query + [0] * q_query)))
    payload = [asdict(spec) for spec in specs]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return (specs, digest)

def select_output(output: Mapping[str, Any], indices: list[int]) -> dict[str, Any]:
    scale = output.get('metric_scale')
    return {'embedding': output['embedding'][indices], 'metric_scale': None if scale is None else scale[indices]}

def compound_support(output: Mapping[str, Any], episode: Mapping[str, Any], index_by_context: Mapping[str, int]) -> tuple[dict[str, Any], torch.Tensor]:
    embeddings, scales, labels = ([], [], [])
    metric_scale = output.get('metric_scale')
    for compound in map(str, episode['support_compound_ids']):
        items = [item for item in episode['support'] if str(item['Compound ID']) == compound]
        indices = [index_by_context[str(item['Context ID'])] for item in items]
        observed_labels = {int(item['Label_bin']) for item in items}
        if len(observed_labels) != 1:
            raise AssertionError('Conflicting compound support labels')
        embeddings.append(output['embedding'][indices].mean(0))
        if metric_scale is not None:
            scales.append(metric_scale[indices].mean(0))
        labels.append(observed_labels.pop())
    if sorted(labels) != [0, 1]:
        raise AssertionError('K=1 support must contain one compound per class')
    pooled = {'embedding': torch.stack(embeddings), 'metric_scale': None if metric_scale is None else torch.stack(scales)}
    return (pooled, torch.tensor(labels, dtype=torch.long, device=pooled['embedding'].device))

@torch.no_grad()
def evaluate_rare(model: E3V2Model, data: Any, table: pd.DataFrame, split: Mapping[str, Any], *, selected_tasks: Sequence[str], seed: int, device: torch.device) -> pd.DataFrame:
    row_by_context = context_map(data)
    rows = []
    for task in selected_tasks:
        task_table = table.loc[table['Canonical E3'].eq(task)].sort_values('Context ID')
        context_ids = task_table['Context ID'].astype(str).tolist()
        output, batch_audit = model_forward(model, [row_by_context[value] for value in context_ids], data, device=device, intervention='correct', source_e3='CRBN', seed=seed)
        index_by_context = {value: index for index, value in enumerate(context_ids)}
        for episode in split['tasks'][task]['episodes']:
            support, support_y = compound_support(output, episode, index_by_context)
            query_items = episode['query']
            indices = [index_by_context[str(item['Context ID'])] for item in query_items]
            scored = model.score_episode(support, support_y, select_output(output, indices))
            classes = scored['classes'].detach().cpu().tolist()
            probabilities = torch.softmax(scored['logits'], dim=1)[:, classes.index(1)].cpu().tolist()
            for item, probability in zip(query_items, probabilities, strict=True):
                rows.append({'seed': seed, 'task': task, 'episode_id': str(episode['Episode ID']), 'support_compound_ids': '|'.join(map(str, episode['support_compound_ids'])), 'context_id': str(item['Context ID']), 'compound_id': str(item['Compound ID']), 'target': str(item['Target']), 'y_true': int(item['Label_bin']), 'y_score': float(probability), 'poi_present_in_batch': int(batch_audit['poi_present'])})
    return pd.DataFrame(rows)

def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame['y_true'].to_numpy(int)
    score = frame['y_score'].to_numpy(float)
    pred = score >= 0.5
    recalls = [float((pred[truth == label] == label).mean()) for label in (0, 1)]
    return {'n': int(len(frame)), 'auroc': float(roc_auc_score(truth, score)), 'auprc': float(average_precision_score(truth, score)), 'balanced_accuracy': float(np.mean(recalls))}

def summarize_worker(predictions: pd.DataFrame, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows = []
    groups = ['seed', 'task', 'episode_id', 'support_compound_ids']
    for keys, frame in predictions.groupby(groups, sort=False):
        episode_rows.append({'protocol': protocol, **dict(zip(groups, keys, strict=True)), **metrics(frame)})
    episode = pd.DataFrame(episode_rows)
    task_seed = episode.groupby(['protocol', 'seed', 'task'], as_index=False)[['auroc', 'auprc', 'balanced_accuracy']].mean()
    return (episode, task_seed)

def protocol_definition(name: str) -> dict[str, Any]:
    if name == 'common_multisource':
        return {'allowed_e3s': ['CRBN', 'VHL'], 'held_out_e3': None, 'test_tasks': list(RARE_E3S), 'train_k': 2, 'train_q': 3}
    if name.startswith('loeo_'):
        target_task = name.removeprefix('loeo_')
        if target_task not in RARE_E3S:
            raise ValueError(f'Unknown rare E3 target: {target_task}')
        target = canonicalize_e3(target_task)
        return {'allowed_e3s': [e3 for e3 in ALL_E3S if e3 != target], 'held_out_e3': target, 'test_tasks': [target_task], 'train_k': 1, 'train_q': 1}
    raise ValueError(f'Unknown protocol: {name}')

def worker(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    set_seed(args.seed)
    device = torch.device(args.device)
    definition = protocol_definition(args.protocol)
    data = build_experiment_data(contexts_path=args.source_contexts, sequence_path=args.rare_contexts, component_path=args.component_table, protein_embeddings_path=args.protein_embeddings, source_e3='CRBN', target_e3='VHL', supervised_train_scope='source_e3_only')
    data.training_contexts = training_rows(data, definition['allowed_e3s'], definition['held_out_e3'])
    observed_e3s = sorted(data.training_contexts['canonical_e3'].unique())
    if observed_e3s != sorted(definition['allowed_e3s']):
        raise AssertionError(f"Training E3 mismatch: {observed_e3s} != {definition['allowed_e3s']}")
    grouped = eligible_multisource_tasks(data.training_contexts, k_shot=definition['train_k'], q_query=definition['train_q'])
    train_groups, validation_groups, selection = split_validation_tasks(grouped, seed=args.seed, fraction=args.validation_fraction)
    train_specs, train_hash = build_balanced_specs(train_groups, count=args.epochs * args.episodes_per_epoch, seed=args.seed, k_shot=definition['train_k'], q_query=definition['train_q'])
    validation_count = max(1, sum((len(tasks) for tasks in validation_groups.values())) * 2)
    validation_specs, validation_hash = build_balanced_specs(validation_groups, count=validation_count, seed=args.seed + 104729, k_shot=definition['train_k'], q_query=definition['train_q'])
    config = E3V2Config(variant='grounded_interaction', node_dim=len(ATOM_FEATURE_NAMES), edge_dim=len(BOND_FEATURE_NAMES), pretrained_protein_dim=data.protein_dim, distance='squared_euclidean', temperature=1.0, learnable_temperature=True)
    model = E3V2Model(config).to(device)
    set_seed(args.seed)
    training = train_episodic(model, data, train_specs, epochs=args.epochs, episodes_per_epoch=args.episodes_per_epoch, learning_rate=args.learning_rate, weight_decay=args.weight_decay, device=device, source_e3='CRBN', seed=args.seed, validation_specs=validation_specs, early_stopping_patience=args.patience, validation_min_delta=args.min_delta)
    table = pd.read_csv(args.rare_contexts, dtype={'Compound ID': str}, low_memory=False)
    split = json.loads(args.split.read_text())
    predictions = evaluate_rare(model, data, table, split, selected_tasks=definition['test_tasks'], seed=args.seed, device=device)
    episode, task_seed = summarize_worker(predictions, args.protocol)
    output = args.output / args.protocol / f'seed_{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'grounded_interaction.pt'
    torch.save({'model_state_dict': model.state_dict(), 'model_config': asdict(config), 'protocol': args.protocol, 'seed': args.seed, 'allowed_training_e3s': definition['allowed_e3s'], 'held_out_e3': definition['held_out_e3']}, checkpoint)
    predictions.assign(protocol=args.protocol).to_csv(output / 'query_predictions.csv', index=False)
    episode.to_csv(output / 'episode_metrics.csv', index=False)
    task_seed.to_csv(output / 'task_seed_metrics.csv', index=False)
    audit = {'format': 'prometa2_multisource_raree3_loeo_v1', 'protocol': args.protocol, 'seed': args.seed, 'device': str(device), 'definition': definition, 'training_rows_by_e3': data.training_contexts['canonical_e3'].value_counts().sort_index().to_dict(), 'eligible_tasks_by_e3': {e3: sorted(tasks) for e3, tasks in grouped.items()}, 'episode_sampling': 'E3-balanced, then uniform eligible task within E3', 'selection': selection, 'train_episode_sha256': train_hash, 'validation_episode_sha256': validation_hash, 'training': training, 'target_e3_in_supervised_training': False, 'test_support_used_for_gradient_training': False, 'elapsed_seconds': time.time() - started, 'software': software_versions(), 'input_sha256': {'source_contexts': sha256(args.source_contexts), 'rare_contexts': sha256(args.rare_contexts), 'component_table': sha256(args.component_table), 'protein_embeddings': sha256(args.protein_embeddings), 'split': sha256(args.split)}, 'artifacts': {'checkpoint': sha256(checkpoint), 'query_predictions': sha256(output / 'query_predictions.csv'), 'episode_metrics': sha256(output / 'episode_metrics.csv'), 'task_seed_metrics': sha256(output / 'task_seed_metrics.csv')}}
    (output / 'run_audit.json').write_text(json.dumps(audit, indent=2, default=str) + '\n')
    (output / '_SUCCESS').write_text('ok\n')
    return {'protocol': args.protocol, 'seed': args.seed, 'output': str(output), 'elapsed_seconds': audit['elapsed_seconds']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["infer", "train"])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    args = parser.parse_args()
    base = args.artifacts.resolve() / "rare_e3"
    args.protocol = "common_multisource"
    args.source_contexts = base / "tables/strict_crbn_vhl_contexts_fair_esm_v1.csv"
    args.rare_contexts = base / "tables/rare_e3_fair_v1.csv"
    args.component_table = base / "tables/protac-fine_esm_complete_v1.csv"
    args.protein_embeddings = base / "embeddings/protein_embeddings_fair_v1.pt"
    args.split = base / "splits/rare_e3_fair_v1_k1_exhaustive/episodes.json"
    args.learning_rate = 1e-3
    args.weight_decay = 1e-5
    args.validation_fraction = .2
    args.patience = 15
    args.min_delta = 1e-4
    if args.output.exists():parser.error("Use a new output directory.")
    if args.action == "train":
        if args.seed is None:parser.error("Training requires --seed; run each seed in its own process.")
        worker(args)
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data = build_experiment_data(contexts_path=args.source_contexts, sequence_path=args.rare_contexts,
        component_path=args.component_table, protein_embeddings_path=args.protein_embeddings,
        source_e3="CRBN", target_e3="VHL", supervised_train_scope="source_e3_only")
    table = pd.read_csv(args.rare_contexts, dtype={"Compound ID": str}, low_memory=False)
    split = json.loads(args.split.read_text())
    rows = []
    for seed in ([args.seed] if args.seed is not None else SEEDS):
        path = args.artifacts.resolve() / f"study_checkpoints/rare_e3/common_multisource/seed_{seed}/model.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["protocol"] == "common_multisource"
        assert set(saved["allowed_training_e3s"]) == {"CRBN", "VHL"}
        model = E3V2Model(E3V2Config(**saved["model_config"]))
        model.load_state_dict(saved["model_state_dict"], strict=True)
        model.to(args.device).eval()
        predictions = evaluate_rare(model, data, table, split, selected_tasks=RARE_E3S,
                                    seed=seed, device=torch.device(args.device))
        rows.append(predictions.assign(protocol="common_multisource"))
        print(f"Rare-E3 seed={seed}", flush=True)
    args.output.mkdir(parents=True)
    pd.concat(rows, ignore_index=True).to_csv(args.output / "query_predictions.csv", index=False)


if __name__ == "__main__":
    main()
