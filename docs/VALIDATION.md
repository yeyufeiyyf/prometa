# Preparation validation — 2026-10-13

## CDK6 checkpoint addition — 2026-10-19

The updated author bundle supplies all three CDK6 checkpoints and its original
33-pair split. Checkpoint inference regenerates 1,188 query rows, with identical
query identities and labels and a maximum score difference of 1.847743988e-6
(tolerance 1e-5). The three-seed AUROC is 0.8858585859 ± 0.0076260954, and the
median support-pair AUROC after averaging seeds is 0.9333333333. Both the
experiment-specific runner and the unified `reproduce.py --experiment cdk6
--mode infer` entry point are provided. The checkpoint catalog now exposes 102
entries including three CDK6 entries. See `cdk6_checkpoint_validation_20261019.json`.
Earlier statements below about missing CDK6 weights describe previous release stages.

## Checkpoint interface update — 2026-10-19

All 99 main/study catalog entries resolve to files with the archived SHA-256 and
load with strict state-dictionary matching and the appropriate main/rare model
version. The experiment selector and explicit-path prediction produced identical
scores for the supplied six-query example. Rare-model CSV prediction also ran.
Individual-setting inference (all three seeds) and individual-seed inference
passed archived score, query-identity, label and AUROC checks. The default full
45-checkpoint metric audit still passes. The complete 45-checkpoint inference
check earlier on October 19 passed with maximum score difference
`1.1102230246251565e-16`. Details of the new interface checks are in
`checkpoint_interface_validation_20261019.json`.

At the initial interface-update stage CDK6 had no supplied original checkpoint.
The later CDK6 addition above supersedes that limitation.

## Original preparation checks

The publication entry points were exercised locally on the selected ProMeta-only files. No remote upload was performed.

| Check | Result |
|---|---|
| Offline manifest | All frozen artifact hashes verified; 42 original checkpoints, 45 named transfer copies and 54 named study copies |
| All-study metric reconstruction | Passed: transfer/K curve/rare/POI/linker/families/CDK6 and original timing aggregation |
| All-study checkpoint inference | Passed for transfer, K curve, rare, POI, linker and HPK1/FLT3; 300 seconds in the preparation environment |
| ProMeta paper reference checks | All 90 transfer mean/SD values and the other stated study means match paper precision |
| CDK6 original-score reconstruction | 1,188 queries; 0.8858586 ± 0.0076261; support-pair median 0.9333333 |
| New-query API example | Four labeled supports and six label-free queries processed successfully |
| Prototype/support-linear control | Frozen-encoder head comparison executed on one source/seed split |
| Structural training smoke | Two epochs × three episodes ran, including source validation, checkpoint writing and target prediction |
| Efficiency scoring path | All three source checkpoints reproduce the archived FLT3 task to at most 5.96e-8 difference on the tested CPU path |
| Packaging | Python sources parse and the method builds as a wheel without fetching dependencies |
| Isolation | No baseline implementation imports or original server paths remain in executable code/configuration |

Inference comparisons preserve molecule/task identities and labels. The largest unrounded score differences and paper-rounded mean/SD agreement are in `inference_comparison.json`; they are not mislabeled as exact bitwise equality. Complete source retraining and fresh GPU timing were not run during publication preparation. The ablation package supplies runnable implementations/configs rather than paper-matched historical numeric checkpoints.

The code and artifact downloads are separate. Temporary validation outputs, model copies, cached plots and build products are not part of the public code archive.

## Named checkpoint validation — 2026-10-14

The 45 transfer setting copies and 54 study copies were checked byte-for-byte against their recorded source weights. Full all-study inference completed using the named checkpoint paths: all 62 comparison units passed at paper precision, and all six paper-reference checks passed. The three efficiency seed paths each passed CPU scoring validation. CDK6 remains an archived-prediction reconstruction because its checkpoint was not supplied.

## Current 15-setting transfer validation — 2026-10-15

The current S2/S3 transfer release has 45 distinct checkpoint hashes from 45 randomly initialized direction–K/Q–seed training runs. Frozen predictions match the saved support/query split identities and labels exactly: 510 task-seed episodes and 4,695 query rows. Recomputing pooled-query and task-mean AUROC matches the current S2/S3 ProMeta summaries at unrounded precision. Reloading all 45 weights and regenerating all 45 prediction files also passed the same identity and summary checks; the maximum absolute query-score difference was $1.11\times10^{-16}$. The older transfer-copy checks above remain a historical record for the K-curve source weights; they do not describe the current S2/S3 models.

The separate K-curve inference path also passed after allowing a $10^{-5}$ absolute score tolerance for CPU re-evaluation of archived checkpoints. Task/query identities, labels, and paper-rounded metrics matched; the maximum difference seen in the strict-check failures before this adjustment was $4.71\times10^{-6}$.

## Matched-source ablations — 2026-10-15

All eleven conditions in the current CRBN→VHL $K=2,Q=3$ paper ablation were completed for seeds 42, 2025 and 3407. The new architecture runs use independent random initialization. For each seed, source partition, source training/validation episode hashes and all 24 target query identities match the paired complete model. The no-episodic and graph-only controls pass the corresponding source-selection audits; support linear uses the paired complete checkpoint. All 18 newly trained architecture metric files match AUROC/AUPRC recomputed from their query predictions. See `artifacts/ablation_matched_source_20261015/summary.csv` and `per_seed_metrics.csv`. The earlier code-only statement above describes the October 13 preparation stage, before this new batch was generated.
