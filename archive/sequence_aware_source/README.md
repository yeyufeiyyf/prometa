# Sequence-aware source snapshot

`run_baselines_vhl.py` is the exact retained self-contained experimental
source snapshot (SHA-256
`fbb4f31d6a5d7c871a6cfd637bc28f74803d0e91f0d449c1ef0b056b9f66eece`).
It defines separate POI and E3 `ProteinEncoder` modules, computes

```text
z = z_mol + alpha_poi * z_poi + alpha_e3 * z_e3
```

and registers the molecular, POI, and E3 parameter sets in the same Adam
optimizer. The implementation has also been integrated into `prometa/model.py`,
`prometa/training.py`, and `prometa/evaluation.py` for a shorter public entry
point.

The retained artifact coverage, including explicitly disclosed missing main
seed-2025/3407 dual-sequence checkpoints, is recorded in
`experiments/sequence_checkpoint_manifest.csv`. Missing original checkpoints
are not reconstructed or substituted.
