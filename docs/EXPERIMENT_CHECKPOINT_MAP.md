# Experiment-specific checkpoint map

Every reported ProMeta experiment has a dedicated artifact under
`checkpoints/by_experiment/<experiment_id>/`.  The exact file-level mapping is
recorded in `experiments/experiment_checkpoint_manifest.csv`.

The same experiment ID has a dedicated split directory under
`splits/by_experiment/<experiment_id>/`, recorded in
`experiments/experiment_split_manifest.csv`.

## Artifact types

- `encoder_state_dict`: a byte-for-byte copy of the retained encoder that
  produced that experiment's released values.  The copied and original SHA-256
  hashes are both recorded in the manifest and must match.
- `frozen_prediction_checkpoint`: a serialized set of the released per-query
  or per-compound predictions.  This is used only for historical analyses for
  which a complete standalone model/input pair was not retained: S7, the R9
  compound-family audit, and the legacy VZ185 audit.
- `support_query_split`: a byte-identical, experiment-specific copy of the
  retained source split for seeds 42, 2025, and 3407.
- `frozen_evaluation_cohort`: an explicit R9/VZ185 cohort file used where no
  standalone historical support/query split exists.

The frozen artifacts are explicit about this distinction; they are not
presented as recovered neural weights.  They preserve the exact reported
values and are evaluated by `experiments/evaluate_frozen_checkpoints.py`.

## Rebuild and evaluate

```bash
python experiments/materialize_experiment_checkpoints.py --force
python experiments/materialize_experiment_splits.py --force
python experiments/evaluate_prometa.py --device cpu
python experiments/evaluate_prometa_3d.py --device cpu
python experiments/evaluate_frozen_checkpoints.py
python experiments/verify_experiment_checkpoints.py
python experiments/verify_experiment_splits.py
python experiments/verify_response_letter.py
```

`evaluate_prometa.py` and `evaluate_prometa_3d.py` load only the dedicated
checkpoint and split `by_experiment` names. They no longer select shared
historical paths at run time.
