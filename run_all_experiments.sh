#!/usr/bin/env bash
# Recompute every public ProMeta result from the experiment-specific
# checkpoints and splits, then run the release audits.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"

printf '%s\n' '[1/10] Verifying release checksums'
"$PYTHON_BIN" scripts/verify_release_checksums.py

printf '%s\n' '[2/10] Verifying sequence-aware source, inputs, and checkpoints'
"$PYTHON_BIN" experiments/verify_sequence_provenance.py

printf '%s\n' '[3/10] Verifying experiment-specific checkpoints and splits'
"$PYTHON_BIN" experiments/verify_experiment_checkpoints.py
"$PYTHON_BIN" experiments/verify_experiment_splits.py

printf '%s\n' '[4/10] Recomputing S2--S6 and S9 checkpoint-backed experiments'
"$PYTHON_BIN" experiments/evaluate_prometa.py --device "$DEVICE"

printf '%s\n' '[5/10] Pooling the rare-E3 common-query predictions'
"$PYTHON_BIN" experiments/summarize_rare_e3.py

printf '%s\n' '[6/10] Recomputing the S8 RDKit-3D control'
"$PYTHON_BIN" experiments/evaluate_prometa_3d.py --device "$DEVICE"

printf '%s\n' '[7/10] Evaluating frozen S7, compound-family, and VZ185 checkpoints'
"$PYTHON_BIN" experiments/evaluate_frozen_checkpoints.py

printf '%s\n' '[8/10] Running the retained main seed-42 sequence-aware path'
"$PYTHON_BIN" experiments/evaluate_sequence_aware.py --device "$DEVICE"

printf '%s\n' '[9/10] Auditing all released response-letter values'
"$PYTHON_BIN" experiments/verify_response_letter.py

printf '%s\n' '[10/10] Complete'
printf 'Fresh outputs: %s\n' "$ROOT/results/recomputed"
