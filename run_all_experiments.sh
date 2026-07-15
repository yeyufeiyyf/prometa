#!/usr/bin/env bash
# Recompute every public ProMeta result from the experiment-specific
# checkpoints and splits, then run the release audits.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"

printf '%s\n' '[1/8] Verifying release checksums'
"$PYTHON_BIN" scripts/verify_release_checksums.py

printf '%s\n' '[2/8] Verifying experiment-specific checkpoints and splits'
"$PYTHON_BIN" experiments/verify_experiment_checkpoints.py
"$PYTHON_BIN" experiments/verify_experiment_splits.py

printf '%s\n' '[3/8] Recomputing S2--S6 and S9 checkpoint-backed experiments'
"$PYTHON_BIN" experiments/evaluate_prometa.py --device "$DEVICE"

printf '%s\n' '[4/8] Pooling the rare-E3 common-query predictions'
"$PYTHON_BIN" experiments/summarize_rare_e3.py

printf '%s\n' '[5/8] Recomputing the S8 RDKit-3D control'
"$PYTHON_BIN" experiments/evaluate_prometa_3d.py --device "$DEVICE"

printf '%s\n' '[6/8] Evaluating frozen S7, compound-family, and VZ185 checkpoints'
"$PYTHON_BIN" experiments/evaluate_frozen_checkpoints.py

printf '%s\n' '[7/8] Auditing all released response-letter values'
"$PYTHON_BIN" experiments/verify_response_letter.py

printf '%s\n' '[8/8] Complete'
printf 'Fresh outputs: %s\n' "$ROOT/results/recomputed"
