# ProMeta ablation code

| Paper component/control | Code variant / operation |
|---|---|
| Complete ProMeta | `grounded_interaction` |
| No POI input | `grounded_no_poi_context` |
| No E3 input | `grounded_no_e3_context` |
| No global interaction | `grounded_no_global` |
| No local recruiter–E3 residual | `grounded_no_component` |
| Legacy additive fusion | `additive` |
| Concat. MLP | `grounded_additive`: nonlinear concatenation projection replacing the two interaction residuals |
| Source linear | `grounded_source_linear` |
| Support linear | Frozen full encoder; L2 logistic regression with C=1, liblinear, balanced classes, max_iter=1000, random_state=0 |
| No episodic training | Ordinary source-supervised classification; target support prototypes retained at inference |
| GNN + prototype | Molecule-only `MoleculeEncoder`, source episodic training and target-support prototypes |

`additive` and `grounded_additive` are different implementations. The historical second name denotes the concatenation MLP, not the legacy additive control. No POI/E3 ablations zero the corresponding context in the encoder; source labels/target support availability follow the unchanged protocol.

Train one structural/input variant with frozen current inputs/splits:

```bash
python -m experiments.ablation_variant --config configs/ablation_variants.json \
  --source-e3 CRBN --target-e3 VHL --seed 42 --variant grounded_no_global --device cuda:0
```

Check data loading, optimization, source validation and target prediction with a short run:

```bash
python -m experiments.ablation_variant --config configs/ablation_variants.json \
  --source-e3 CRBN --target-e3 VHL --seed 42 --variant grounded_no_global --device cpu \
  --smoke --output-root reproduced/ablation_smoke/results --checkpoint-root reproduced/ablation_smoke/checkpoints
```

Compare prototype/support-linear heads on the same archived full encoder without retraining:

```bash
python -m experiments.ablation_training_head --config configs/ablation_training_head.json \
  --source-e3 CRBN --target-e3 VHL --seed 42 --device cpu --head-only
```

Omit `--head-only` to train the ordinary supervised source control. It uses the archived full model's source-only task split, validation episode identities and source-example exposure budget. Its target-query labels do not select model parameters.

Train the molecule-only prototype control:

```bash
python -m experiments.ablation_graph_only --artifacts artifacts --direction CRBN_to_VHL \
  --seed 42 --device cuda:0 --output reproduced/graph_only_seed42
```

The current paper uses the separate `ablation_matched_source_20261015` batch in the artifact archive. It covers all eleven conditions above at CRBN→VHL, $K=2,Q=3$, seeds 42, 2025 and 3407. Each seed has four tasks and 24 fixed query rows. The 21 fresh architecture checkpoints (including `w/o POI`), three fresh no-episodic encoders and three fresh molecule-only GNNs were trained from random initialization on the full model's fixed source pool. Support linear uses the paired full encoder. The source task partition, training and validation episode identities, and target query IDs were audited against the paired full run; `summary.json` records the result. The archive contains per-seed predictions and the pooled-query AUROC/AUPRC summary with between-seed SD. Source snapshots of the exact study runners are under `provenance/`; their original workspace paths need adaptation for an independent retraining environment. The generic commands above document the underlying controls but use their own configured source split, so they should not be used to reproduce the current paper table without the matched-source protocol.
