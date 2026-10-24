# ProMeta processed-data release, version 2.1.0

This is a distribution-content update of the published v2.0.0 experiments.
It omits the raw PROTAC-DB export and raw activity-measurement/assay columns.
Final dataset labels, retained model inputs, molecular structures, protein
features, evaluation splits, predictions and checkpoints are preserved.
No experiments were retrained and no labels were regenerated for this release.

The main sequence-audited dataset contains 1,325 contexts (808 CRBN and 517 VHL;
678 positive and 647 negative). The rare-E3 pool contains 40 contexts.
The parallel sequence table has 1,453 records and is not the parent of the
strict 1,325-context dataset.

These are the historical publication labels, not a newly DegradeMaster-relabeled
dataset. The archived scalar rule assigned 1 for DC50 <100 nM OR Dmax >80%, and
0 otherwise. The reported preprocessing excluded scalar DC50 in [100,500] nM
OR Dmax in [60,80]%, followed by per-E3 balancing with single-class pools retained.
Negative labels include records without quantitative evidence and do not imply
experimentally confirmed inactivity. Qualitative activity calls did not enter
this scalar rule. Subsequent context conflict removal and sequence auditing
can change class ratios. See retained context/sequence audits.

This package supports reproduction starting from the released processed inputs.
It does not provide a complete reconstruction from the raw database export.
The early import-cleanup and random-sampling implementation is not included.
An audit of the former raw export found that applying the stated scalar rule,
gray-zone filter and per-E3 balancing without that early cleanup yields 1,602
records rather than the archived 1,598; the missing early-stage provenance
therefore remains unresolved. This packaging change does not resolve it.

Molecular source identities and DOI references are retained for attribution.
The upstream source is PROTAC-DB, https://cadd.zju.edu.cn/protacdb/ .
The source snapshot was the local 9,384-row export distributed in v2.0.0;
its original retrieval date is not established here. Upstream data rights
remain with their respective providers; the repository MIT license applies
to code and does not relicense upstream molecular data.

MANIFEST.json records hashes of files distributed in this revision.
Historical training/audit manifests preserve hashes of the original training
inputs, including tables that formerly contained unused measurement columns.
Those hashes describe historical provenance, not the stripped files.
PROCESSING_RELEASE_CHANGES.json identifies changed tables and both hashes.
Sequence feature coverage and model-input values are unchanged; historical
measurement-backfill checks are not independently rerunnable without the
source measurements. Sequence and literature mappings remain documented.

Code and checkpoints use the same modeling implementation as v2.0.0.
Removing measurement columns does not establish a different labeling policy.
