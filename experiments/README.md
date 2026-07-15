# ProMeta experiment entry points

This directory is the executable mapping between the response letter and the
ProMeta-only public artifacts.

- `response_letter_experiments.csv`: split, data, checkpoint, head, and K/Q
  registry for every experiment.
- `response_letter_expected.csv`: rounded values transcribed from the response
  letter for automated verification.
- `build_response_splits.py`: original and BCR--ABL-excluded CRBN/VHL splits.
- `build_rare_common_query_splits.py`: the frozen 90-row rare-E3 common-query
  cohort.
- `train_prometa.py`: episodic ProMeta and non-episodic component-control
  training.
- `materialize_experiment_checkpoints.py`: creates one explicitly named
  checkpoint artifact per reported experiment and writes the checkpoint
  manifest.
- `materialize_experiment_splits.py`: creates one explicitly named split or
  frozen-cohort artifact per reported experiment and writes the split
  manifest.
- `evaluate_prometa.py`: checkpoint-backed prototype and linear-head inference.
- `evaluate_prometa_3d.py`: RDKit conformer-descriptor ProMeta control.
- `evaluate_frozen_checkpoints.py`: exact evaluation of the historical S7,
  compound-family, and VZ185 frozen-prediction checkpoints.
- `summarize_rare_e3.py`: pooled overall/per-ligase S4 metrics.
- `verify_experiment_checkpoints.py`: file existence, source/copy SHA-256,
  sidecar, and dedicated-path audit for all checkpoint artifacts.
- `verify_experiment_splits.py`: file existence, source/copy SHA-256, and
  dedicated-path audit for all split/cohort artifacts.
- `verify_response_letter.py`: numeric and ProMeta-only scope audit.

Typical reproduction sequence:

```bash
python experiments/materialize_experiment_checkpoints.py --force
python experiments/build_response_splits.py --profile all
python experiments/build_rare_common_query_splits.py
python experiments/materialize_experiment_splits.py --force
python experiments/evaluate_prometa.py --device cpu
python experiments/summarize_rare_e3.py
python experiments/evaluate_prometa_3d.py --device cpu
python experiments/evaluate_frozen_checkpoints.py
python experiments/verify_experiment_checkpoints.py
python experiments/verify_experiment_splits.py
python experiments/verify_response_letter.py
```

The frozen reference predictions and summaries are under `results/released/`;
fresh runs are written under `results/recomputed/`.
