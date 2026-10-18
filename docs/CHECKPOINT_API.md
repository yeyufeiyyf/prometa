# Checkpoint selection and prediction

Extract `prometa_artifacts_20261019_cdk6.zip` inside the repository root. This update
adds the three supplied CDK6 checkpoints and their fixed split; other weights are unchanged. The code and artifact
ZIPs must both be supplied when publishing this release.

## List and verify weights

```bash
python -m prometa.checkpoints --verify
python -m prometa.checkpoints --study transfer --verify
```

The catalog exposes 102 named entries:

| Study (`--study`) | Selector | Seeds / number of files |
|---|---|---|
| `transfer` | `--direction`, `--k`, `--q` | 15 settings × 3 seeds = 45 distinct weights |
| `kcurve` | `--direction` | 2 directions × 3 seeds = 6 |
| `poi` | `--direction` | 2 directions × 10 seeds = 20 |
| `linker` | `--direction` | 2 directions × 3 seeds = 6 |
| `families` | `--condition HPK1`, `FLT3` or `FLT3_10nM` | 3 assays × 3 seeds = 9 |
| `cdk6` | `--condition CDK6` (optional) | 3 seeds = 3 |
| `efficiency` | `--condition FLT3` | 3 seeds = 3 |
| `rare_e3` (alias `rare`) | `--condition common_multisource` | 10 seeds = 10 |

The three paper seeds are 42, 2025 and 3407. Ten-seed studies additionally use
101, 202, 303, 404, 505, 606 and 707. The 57 non-transfer study entries are named
copies of their recorded original weights; some studies deliberately reuse an
encoder. The current transfer entries are 45 independently trained weights.
The catalog rejects unknown or ambiguous choices and verifies SHA-256 when resolving a path.

## Predict new queries

Select the model by its experiment identifiers:

```bash
python predict.py --study transfer --direction VHL_to_CRBN --k 2 --q 3 --seed 42 \
  --support examples/support.csv --query examples/query.csv \
  --output reproduced/new_queries.csv
```

The supplied example contains one CRBN/CDK6 target task. To choose a family-study
encoder for this same illustrative task:

```bash
python predict.py --study families --condition HPK1 --seed 42 \
  --support examples/support.csv --query examples/query.csv \
  --output reproduced/new_queries_family_encoder.csv
```

Selecting a study chooses its saved encoder; it does not replace the supplied
support/query CSVs with that study's original data. These example predictions do
not reconstruct the original HPK1 or CDK6 case study. Use the reproduction commands
below for original study results. Transfer K/Q flags identify the saved model's
setting; the new CSVs define the actual support/query compounds for a new prediction.

The protein bank and source E3 are inferred from the artifact bundle. Use
`--artifacts /path/to/artifacts` when it is elsewhere. `--protein-bank` and
`--e3-map` allow explicitly supplied features for new proteins/ligases.

Both CSVs require `Compound ID`, `Uniprot`, `Target`, `E3 ligase` and `Smiles`.
Support also requires `Label_bin` with both 0 and 1 represented. Each call must
contain a single shared POI/E3 task, with disjoint support/query contexts.
`e3 ligand smiles` supplies recruiter localization. Query labels are ignored.
The output contains `prometa_active_score` (softmax probability for dataset class
1) and a `predicted_label` using threshold 0.5. This is a model score, not a
calibrated estimate of experimental success. The adjacent `.metadata.json`
records the selected weight, seed and SHA-256 hashes of the model, feature bank,
support file and query file. Existing output files are not overwritten.

The original explicit-path interface remains available:

```bash
python predict.py \
  --checkpoint artifacts/transfer_setting_fresh_20261015/checkpoints/VHL_to_CRBN/k2_q3/seed_42/model.pt \
  --source-e3 VHL --support examples/support.csv --query examples/query.csv \
  --output reproduced/new_queries_explicit.csv
```

## Python interface

```python
from prometa.checkpoints import resolve_checkpoint, load_checkpoint

selection = dict(study="transfer", direction="VHL_to_CRBN", k=2, q=3, seed=42)
path = resolve_checkpoint("artifacts", **selection)  # verifies archived SHA-256
model = load_checkpoint("artifacts", device="cpu", **selection)
```

`load_checkpoint` selects the separately versioned rare-E3 architecture when
`study="rare_e3"`. The CSV prediction interface also selects its matching forward
implementation. Ablation weights and their distinct prediction heads are covered
by [ABLATIONS.md](ABLATIONS.md); they are not interchangeable with the main-model
prediction interface.

## Reproduce saved experiment results

One transfer setting, all three seeds:

```bash
python -m experiments.transfer_setting_fresh --mode infer \
  --direction CRBN_to_VHL --k 2 --q 3 --output reproduced/forward_k2q3
```

One transfer checkpoint / seed:

```bash
python -m experiments.transfer_setting_fresh --mode infer \
  --direction VHL_to_CRBN --k 2 --q all --seed 42 \
  --output reproduced/reverse_k2qall_seed42
```

All 45 transfer checkpoints:

```bash
python reproduce.py --experiment transfer --mode infer --output reproduced/all_transfer
```

Other study protocols (including their original support selection and aggregation):

```bash
python reproduce.py --experiment poi --mode infer --output reproduced/poi
python reproduce.py --experiment rare --mode infer --output reproduced/rare
python reproduce.py --experiment families --mode infer --output reproduced/families
```

Use `kcurve`, `linker` or `cdk6` in the same interface for those studies. CDK6 now
supports checkpoint inference:
`python reproduce.py --experiment cdk6 --mode infer --output reproduced/cdk6`.
For new query CSVs, select its encoder with `predict.py --study cdk6 --seed 42`
and the usual support/query/output arguments.
The efficiency metric workflow reconstructs archived timings; fresh GPU timing
requires the separate profiling command documented in README.

Inference verifies saved query identities, labels, score tolerance and metrics.
Minor floating-point differences across platforms are allowed; use the supplied
environment and fixed inputs to compare results. Seed-specific AUROC is checked
against that seed's reference, while a three-seed run also checks the paper mean/SD.
