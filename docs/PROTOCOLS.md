# ProMeta publication protocols

## Checkpoint families

Keep three families separate: (1) the supplied current transfer rerun, (2) the context-source weights used by POI/linker/HPK1/FLT3/efficiency, and (3) CRBN+VHL multi-source rare-E3 weights. All use the bond-aware GNN/ESM/global-and-local interaction design, but they are separate trained runs. Never substitute a shared `best_model.pt` for these experiment-specific references.

For convenient loading, `artifacts/study_checkpoints/<study>/<condition>/seed_<seed>/model.pt` contains named, byte-identical copies for K curve, POI, linker, families, CDK6, efficiency and rare E3. The condition is the direction for K curve/POI/linker, the assay family for HPK1/FLT3, `CDK6` for the CDK6 case, `FLT3` for efficiency, and `common_multisource` for rare E3. The artifact download's `study_checkpoints/checkpoint_mapping.json` records each original path and SHA-256. Naming a copy for a study does not imply separate training.

The main, rare and ablation module copies preserve the corresponding source algorithms separately. The architecture's internal class name `E3V2Model` is retained for checkpoint compatibility.

## Current 15-setting transfer and archived K curve

- Seeds: 42, 2025, 3407. Six settings per direction plus three additional reverse Q10 settings total 15.
- Source and target E3s are disjoint in training/evaluation. Support/query context identities and labels are fixed in saved JSONs.
- Current S2/S3 and Figure 1 a,b use `artifacts/transfer_setting_fresh_20261015/`: 45 randomly initialized, independently source-trained models, one for each direction–K/Q–seed combination. The source-training molecule pool and source-validation tasks are fixed within each direction/seed and match the designated baseline source partition. Source validation selects an epoch; target support constructs prototypes without encoder updates, and the held-out target queries are never used for training. Each `runs/<direction>/k<k>_q<q>/seed_<seed>/` directory contains a run manifest, source-partition audit, and query predictions; `checkpoints/` and `splits/` use the same setting identifiers. There are 45 distinct checkpoint SHA-256 values, 510 task-seed episodes, and 4,695 query rows. S2 pools query scores within each seed before AUROC; S3 averages task-level AUROCs equally within each seed. The two metrics are then averaged across seeds.
- The older `artifacts/current_transfer/table_s4_prometa_rerun/` contains 45 evaluation-named copies of 12 source weights. These are preserved as provenance for the archived shared-checkpoint run and K curve; they are not current S2/S3 transfer weights.
- The final K curve uses those K2Q3 weights and the fixed K5-evaluable task/query cohort: 33 forward and 170 reverse query rows per seed. K0 constructs prototypes from five deterministically selected source-training rows per class per source task, excluding source validation tasks.
- K0→K1 changes prototype provenance, not only support budget. K≥1 compares larger target support sets with a fixed encoder.
- Final K-curve whiskers are three-seed min–max ranges; reported CSV SDs are sample standard deviations. Main-figure compact transfer bars do not show SD.
- Eligible task/query pools can change between transfer settings. Comparisons across methods belong within each setting.

## Rare E3

Ten seeds: 42, 101, 202, 303, 404, 505, 606, 707, 2025, 3407. Train on CRBN+VHL, evaluate FEM1B/MDM2/XIAP/cIAP1. The target pool contains 40 contexts from 38 compounds. A support compound's multiple context embeddings are averaged before constructing its class reference; all contexts of a compound stay on the same support/query side. First average support-pair metrics within each ligase/seed, then weight four ligases equally. Exhaustive support pairs share compounds and are not independent replicates.

## POI and linker

POI uses ten source checkpoint seeds and fixed six-query target tasks. Four K1 support subsets produce four scores per query: average those scores **before** computing task AUROC, then average tasks equally within each direction/seed.

Linker uses seeds 42/2025/3407, nine fixed-end series and 85 compounds. Enumerate active–inactive support pairs within each series; average episode metrics within series before equal weighting of series. Both molecular and canonical-linker identities are disjoint between supports and queries. Rank percentiles are computed among queries in each episode, then averaged per compound. The frozen `compound_display_mapping.csv` maps Compound 1–11 to the actual BTK series identities; measured labels determine display grouping, and score-blind hashing determines within-group order.

## Family cases and efficiency

HPK1/FLT3 use all active–inactive support pairs, source-VHL checkpoints and three seeds. Average pair metrics within each seed, then report seed mean/sample SD. Keep the raw database-label FLT3 cohort separate from the secondary FLT3_10nM assay endpoint.

CDK6 supplies three original checkpoints (seeds 42/2025/3407), its 14-compound cohort, all 33 support pairs, the fixed evaluation split and reference predictions. `reproduce.py --experiment cdk6 --mode infer` reloads the checkpoints without training, regenerates 1,188 queries and verifies their identities, labels and scores before recovering AUROC 0.8858586 ± 0.0076261. The maximum CPU score difference in the October 19 validation was 1.85e-6, within the recorded 1e-5 tolerance. Per-seed episode AUROCs are averaged before computing the three-seed mean/sample SD. The main model implementation and supplied frozen protein/component features are used.

Efficiency summaries average each seed's median measured duration, rather than pooling raw timing calls. B16 processes 16+16+1 candidates and divides full-pool time by 33. GPU memory is peak allocated memory; online timings exclude input preparation. A fresh GPU profile can vary by device and system load.

## Numerical comparisons

Frozen files are checked by SHA256. Regenerated query identities and labels must match exactly. Validation separately records maximum unrounded score differences and compares mean/SD columns at the paper's three-decimal precision. CPU/GPU floating-point reductions can alter near-tied scores and AUPRC ranks; raw-score equality is not asserted. This release does not silently broaden the original ZIP's extremely strict tolerance and call it exact reproduction: it exposes both paper-precision agreement and the unrounded differences.

Full source retraining commands are supplied, but publication preparation validation used existing checkpoints. Byte-identical fresh training is not claimed.
