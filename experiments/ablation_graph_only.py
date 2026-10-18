"""Molecule-only GNN training with support-based prototype classification."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from prometa.ablations.data import ATOM_FEATURE_NAMES, BOND_FEATURE_NAMES
from prometa.ablations.experiment import build_episode_specs, build_experiment_data, eligible_training_tasks, load_eval_episodes, model_graph, set_seed, split_source_validation_tasks
from prometa.ablations.model import MoleculeEncoder

def make_graph_batch(data, row_ids, device):
    return Batch.from_data_list([model_graph(data.graph_index[str(r)]) for r in row_ids]).to(device)

def encode(model, data, row_ids, device):
    batch = make_graph_batch(data, row_ids, device)
    return model(batch.x, batch.edge_index, batch.batch, getattr(batch, 'edge_attr', None)).graph_embedding

def logits(s, sy, q):
    labels = torch.as_tensor(sy, dtype=torch.long, device=s.device)
    classes = torch.tensor([0, 1], dtype=torch.long, device=s.device)
    protos = torch.stack([s[labels == c].mean(0) for c in classes])
    return -torch.cdist(q, protos).pow(2)

def episode_loss(model, data, spec, device):
    z = encode(model, data, list(spec.support_ids) + list(spec.query_ids), device)
    ns = len(spec.support_ids)
    out = logits(z[:ns], spec.support_labels, z[ns:])
    qy = torch.as_tensor(spec.query_labels, dtype=torch.long, device=device)
    return F.cross_entropy(out, qy)

def train(data, source, seed, manifest, device, run_dir):
    settings = manifest['settings']
    tasks = eligible_training_tasks(data.training_contexts, source_e3=source, k_shot=2, q_query=3)
    train_tasks, val_tasks, selection = split_source_validation_tasks(tasks, source_e3=source, seed=seed, fraction=float(settings['source_validation_fraction']))
    total_epochs = int(manifest['training']['epochs_completed'])
    train_specs, train_hash = build_episode_specs(train_tasks, count=int(settings['epochs']) * int(settings['episodes_per_epoch']), seed=seed, k_shot=2, q_query=3)
    val_count = max(len(val_tasks), len(val_tasks) * int(settings['source_validation_episodes_per_task']))
    val_specs, val_hash = build_episode_specs(val_tasks, count=val_count, seed=seed + 104729, k_shot=2, q_query=3)
    if selection != manifest['source_validation_selection']:
        raise RuntimeError('Source validation task split differs from the main checkpoint')
    if train_hash != manifest['training_episode_sha256']:
        raise RuntimeError('Source training episode stream differs from the main checkpoint')
    if val_hash != manifest['source_validation_episode_sha256']:
        raise RuntimeError('Source validation episode stream differs from the main checkpoint')
    model = MoleculeEncoder(len(ATOM_FEATURE_NAMES), len(BOND_FEATURE_NAMES), 128, 3, 0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings['learning_rate']), weight_decay=float(settings['weight_decay']))
    best, best_state, best_epoch = (float('inf'), None, 0)
    history = []
    offset = 0
    for epoch in range(total_epochs):
        model.train()
        losses = []
        for spec in train_specs[offset:offset + int(settings['episodes_per_epoch'])]:
            optimizer.zero_grad(set_to_none=True)
            loss = episode_loss(model, data, spec, device)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        offset += int(settings['episodes_per_epoch'])
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([episode_loss(model, data, x, device).item() for x in val_specs]))
        train_loss = float(np.mean(losses))
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'validation_loss': val_loss})
        if val_loss < best - float(settings.get('validation_min_delta', 0.0001)):
            best, best_epoch = (val_loss, epoch + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError('No best graph checkpoint was selected')
    model.load_state_dict(best_state)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'training_episode_sha256': train_hash, 'validation_episode_sha256': val_hash, 'best_epoch': best_epoch, 'epochs_completed': len(history)}, run_dir / 'episodic_gnn.pt')
    pd.DataFrame(history).to_csv(run_dir / 'training_history.csv', index=False)
    (run_dir / 'training_audit.json').write_text(json.dumps({'status': 'PASS', 'source_e3': source, 'seed': seed, 'training_episode_sha256': train_hash, 'source_validation_episode_sha256': val_hash, 'source_validation_selection': selection, 'best_epoch': best_epoch, 'epochs_completed': len(history), 'target_e3_used_for_training_or_selection': False, 'input_manifest': str((MAIN / f"training_records/{source}_to_{('VHL' if source == 'CRBN' else 'CRBN')}/seed_{seed}/run_manifest.json").resolve())}, indent=2) + '\n')
    return model


def main():
    from prometa.release_io import training_data
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--direction',choices=['CRBN_to_VHL','VHL_to_CRBN'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory.')
    global MAIN
    MAIN=a.artifacts.resolve()/'context_weights'
    source,target=a.direction.split('_to_')
    manifest=json.loads((MAIN/f'training_records/{a.direction}/seed_{a.seed}/run_manifest.json').read_text())
    data=training_data(a.artifacts,source,target)
    split=a.artifacts/f'current_transfer/table_s4_prometa_rerun/splits/{a.direction.lower()}/seed_{a.seed}/k2_q3.json'
    device=torch.device(a.device);torch.set_num_threads(2);set_seed(a.seed)
    model=train(data,source,a.seed,manifest,device,a.output)
    _,episodes=load_eval_episodes(split,data);rows=[]
    with torch.no_grad():
        for ep in episodes:
            z=encode(model,data,list(ep.support_ids)+list(ep.query_ids),device)
            n=len(ep.support_ids);prob=logits(z[:n],ep.support_labels,z[n:]).softmax(1)[:,1].cpu().numpy()
            for rid,y,value in zip(ep.query_ids,ep.query_labels,prob):
                rows.append(dict(direction=f'{source}->{target}',seed=a.seed,task=ep.task,row_id=rid,y_true=int(y),y_score=float(value)))
    pd.DataFrame(rows).to_csv(a.output/'query_predictions.csv',index=False)
    print(f'Trained graph-only prototype control: {a.output}')


if __name__=='__main__':main()
