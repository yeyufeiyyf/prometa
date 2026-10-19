# ProMeta

ProMeta predicts binary PROTAC degradation activity in a new E3-ligase context from a small labeled support set. A molecular GNN with bond features and fixed protein features supplies the representation; global and recruiter–E3 interaction branches combine the inputs. Active/inactive class prototypes classify query compounds without updating the encoder at target inference.

This repository contains the ProMeta method and its publication experiments. Comparator implementations and weights are excluded. The code is MIT-licensed; frozen molecular data retain their original source attribution.

## Install

The tested environment is Python 3.10, PyTorch 2.5.1, PyG 2.8.0 and the versions in `requirements.txt`. The local validation used the CUDA 12.1 build of PyTorch, including CPU inference runs.

```bash
conda env create -f environment.yml
conda activate prometa
python -m pip install -e . --no-deps
python verify_release.py --code-only
```

Use a PyTorch build compatible with the available GPU for GPU training/profiling. All archived prediction/metric reproduction commands below run on CPU. ESM-2 feature-producing weights are not needed when using the provided frozen protein bank.

## Download the ProMeta artifacts

The code archive and the frozen artifact archive are separate. Extract `prometa_artifacts_20261019_cdk6.zip` **inside the repository root**; it creates `artifacts/`. The current bidirectional comparison has 45 independently trained direction–setting–seed checkpoints, its 45 fixed splits, and individual query predictions under `artifacts/transfer_setting_fresh_20261015/`. The bundle also retains the earlier shared-checkpoint transfer run for the separate K-curve analysis, plus frozen contexts/components/protein features, other study checkpoints, and measured timings. The artifact bundle is ignored by git and belongs in a GitHub Release asset or the linked data deposit.

```bash
curl -fL -o prometa_artifacts_20261019_cdk6.zip \
  https://github.com/yeyufeiyyf/prometa/releases/download/v2.0.0/prometa_artifacts_20261019_cdk6.zip
unzip prometa_artifacts_20261019_cdk6.zip
python verify_release.py
```

Download the [complete checkpoint, split and input bundle](https://github.com/yeyufeiyyf/prometa/releases/download/v2.0.0/prometa_artifacts_20261019_cdk6.zip) from the [v2.0.0 publication release](https://github.com/yeyufeiyyf/prometa/releases/tag/v2.0.0). The release also supplies the code ZIP and a SHA-256 download manifest. See [checkpoint API documentation](docs/CHECKPOINT_API.md) for individual experiments, including CDK6.

Permanent archive: [Zenodo v2.0.0](https://doi.org/10.5281/zenodo.23474832). The archive contains the same code snapshot, checkpoints, fixed splits and frozen inputs as this publication release. The series DOI is [10.5281/zenodo.21371598](https://doi.org/10.5281/zenodo.21371598).

The Zenodo copy stores the artifact ZIP in 30 volumes. Run `python restore_zenodo_artifacts.py --download` to download, verify and restore the original ZIP, then extract it inside this repository. The restored file is byte-identical to the single GitHub artifact ZIP.

For direct loading, the artifact bundle includes `study_checkpoints/<study>/<condition>/seed_<seed>/model.pt` for K curve, POI, linker, families, CDK6, efficiency and rare E3. Use the direction as `condition` for K curve/POI/linker, the assay name for families, `CDK6` for the CDK6 case, `FLT3` for efficiency and `common_multisource` for rare E3. These files are copies of the original trained weights; `study_checkpoints/checkpoint_mapping.json` records the exact source and SHA-256 of each copy. To regenerate them from the original weights, run `python -m experiments.materialize_study_checkpoints --artifacts artifacts`.

## Reproduce the paper studies

`K` is the support count **per class**, and numeric `Q` is the query count **per class**. Thus K2Q3 uses four labeled references and six queries. An internal `Q=0` filename means all eligible queries remaining after support selection.

Recompute every ProMeta result from the frozen query predictions or measured timings:

```bash
python reproduce.py --experiment all --mode metrics --output reproduced/paper_metrics
```

Regenerate predictions for transfer, K learning curves, rare E3, POI, linker, HPK1/FLT3 and CDK6 from their saved weights:

```bash
python reproduce.py --experiment all --mode infer --output reproduced/paper_inference --workers 2
```

Generate ProMeta-only study plots from either output directory:

```bash
python -m experiments.plot_results --run reproduced/paper_inference --artifacts artifacts --output reproduced/prometa_plots
```

The all-study inference command now reloads CDK6's three checkpoints and regenerates all 1,188 query scores under its 33 fixed support pairs. Original efficiency timings are recomputed; GPU timing is not rerun by this command. Each study's level is recorded in `reproduction_status.json`. `comparison.json` checks identities, labels and all paper-rounded metric columns, while recording unrounded score/metric differences.

| Study | Paper protocol | Entry point / evidence |
|---|---|---|
| Bidirectional transfer | 15 direction–setting comparisons, seeds 42/2025/3407 | `reproduce.py --experiment transfer`; 45 separately initialized and source-trained checkpoints, 45 matched evaluation cells; pooled-query and task-mean AUROC |
| K curve | Fixed K5 common-query cohort, K=0–5; K=1–7 extension archived | `reproduce.py --experiment kcurve`; source prototypes at K0 |
| Rare E3 | CRBN+VHL multi-source; four rare ligases; exhaustive compound-level one-shot supports | `reproduce.py --experiment rare`; ten independently trained weights |
| POI | 13 POI–direction pairs; 10 seeds; K0/K1/K2 | `reproduce.py --experiment poi`; four K1 support subsets averaged per query before metrics |
| Linker | Nine fixed-end series, 85 molecules, three seeds | `reproduce.py --experiment linker`; disjoint support/query linkers, equal series weighting |
| HPK1/FLT3 | All active–inactive support pairs; three seeds | `reproduce.py --experiment families`; FLT3_10nM is a separately labeled secondary endpoint |
| CDK6 | 14 compounds, 33 support pairs, 1,188 original queries | `reproduce.py --experiment cdk6 --mode infer`; three checkpoints, fixed split; 0.886 ± 0.008, median 0.933 |
| Efficiency | 33-candidate/FLT3 workload; 1,380 original ProMeta timing samples | `reproduce.py --experiment efficiency --mode metrics` |

Always choose a new output directory. Reproduction never overwrites the reference artifact bundle. See [protocol details](docs/PROTOCOLS.md) for checkpoint selection, aggregation and numerical precision.

## Train ProMeta

The current S2/S3 transfer results can also be checked directly, without running the other studies:

```bash
python -m experiments.transfer_setting_fresh --artifacts artifacts --output reproduced/transfer_current
python -m experiments.transfer_setting_fresh --artifacts artifacts --mode infer --output reproduced/transfer_current_infer
```

The first command recomputes both AUROC aggregations from frozen query predictions; the second reloads each of the 45 distinct checkpoints and evaluates the same saved splits. Training provenance, per-run manifests, configuration, and source-partition hashes accompany the weights. The older `experiments/transfer.py` driver and its 12 source checkpoints belong to the archived shared-checkpoint transfer/K-curve run and do not reproduce the current S2/S3 training protocol.

Train one CRBN+VHL multi-source seed for rare E3:

```bash
python -m experiments.rare_e3 train --artifacts artifacts --seed 42 --device cuda:0 --output reproduced/rare_train_seed42
```

Only `common_multisource` is exposed by this publication entry point. The original rare-model training and inference code version is preserved under `prometa/rare/`.

## Predict a new query set

Choose a checkpoint by experiment, direction, K/Q and seed:

```bash
python -m prometa.checkpoints --study transfer --verify
python predict.py --study transfer --direction VHL_to_CRBN --k 2 --q 3 --seed 42 \
  --support examples/support.csv --query examples/query.csv \
  --output reproduced/example_selected_checkpoint.csv
```

The source E3 and frozen protein bank are selected automatically. Each prediction
writes a matching `.metadata.json` recording weight and input hashes. The catalog
also exposes 57 named checkpoints for the other studies. See
[checkpoint selection, Python API and individual-setting reproduction](docs/CHECKPOINT_API.md).

Explicit paths are also supported:

```bash
python predict.py \
  --checkpoint artifacts/context_weights/checkpoints/VHL_to_CRBN/seed_42/grounded_interaction.pt \
  --protein-bank artifacts/current_transfer/prometa_as/data-prometa/embeddings/protein_embeddings_fair_v1.pt \
  --support examples/support.csv --query examples/query.csv \
  --source-e3 VHL --output reproduced/example_predictions.csv
```

The included example is a frozen CRBN target episode. Both CSVs need `Compound ID`, `Uniprot`, `Target`, `E3 ligase` and `Smiles`. The support also needs binary `Label_bin`; query labels do not enter scoring. `e3 ligand smiles` supplies recruiter localization. New proteins need their matching 1,280-dimensional frozen features; `--e3-map` optionally supplies E3-name-to-feature-key mappings. The active-class score concerns binary activity, not DC50/Dmax regression.

## Ablations and profiling

The current CRBN→VHL $K=2,Q=3$ input and model-design ablations are released with three-seed checkpoints, per-query predictions, run audits, and summaries under `artifacts/ablation_matched_source_20261015/`. The older general ablation entry points remain available; see [ablation protocols and results](docs/ABLATIONS.md).

The ProMeta-only profiler checks its archived CPU scoring path without GPU timing:

```bash
python -m experiments.efficiency --artifacts artifacts --seed 42 --validate-only --output reproduced/efficiency_input_check
```

For a fresh GPU measurement, omit `--validate-only` and choose `--device cuda:0`. The profiler excludes model loading, graph/ESM preprocessing and disk I/O from timed calls, and updates only a disposable training copy.

## Files and versions

- `prometa/`: main GNN, protein adapters, interaction branches, prototype classifier and episodic engine.
- `prometa/ablations/`: exact extended architecture version implementing removals, concatenation fusion and source-linear scoring.
- `prometa/rare/`: rare-E3 campaign code version.
- `experiments/`: ProMeta-only study, ablation and profiling entry points.
- `configs/`: portable source/ablation settings.
- `results/`: small ProMeta-only reference summaries; older S4/S5 summaries are explicitly separated.
- `artifacts/MANIFEST.json`: frozen inputs/checkpoint/prediction hashes.
- `artifacts/study_checkpoints/`: checkpoint copies named by study, condition/direction and seed; `checkpoint_mapping.json` records their original paths and hashes.
- `docs/`: protocols, version boundaries, data attribution and publication instructions.

Older supplementary UMAP/scaffold pictures and legacy 3D/intersection implementations are not replaced with current-model outputs. The legacy S4/S5 numerical references are identified in [data and legacy scope](docs/DATA_AND_LEGACY.md).
