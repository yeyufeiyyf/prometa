# Publication release

Repository: https://github.com/yeyufeiyyf/prometa

Current release: https://github.com/yeyufeiyyf/prometa/releases/tag/v2.1.0

Version 2.1.0 updates distribution contents and download instructions. GitHub hosts the source, two processed label tables, the restoration script and its checksum manifest. The complete processed artifact bundle is archived at https://doi.org/10.5281/zenodo.23618575 as 30 ZIP volumes. See the repository README for restoration and verification commands.

Labels, retained model inputs, protein features, splits, predictions and weights are unchanged from the previous publication artifacts. Raw PROTAC-DB measurements and the raw export are omitted. See [data scope and limitations](PROCESSED_DATA_README.md). The artifact MANIFEST.json describes the newly distributed files; historical training hashes remain provenance for the original inputs.

The old GitHub v2.0.0 raw-data artifact and its download helpers have been withdrawn. Its code archive and source tag remain historical records. This GitHub update does not remove earlier Zenodo versions. Historical source files may refer to old archive names, but the current README and v2.1.0 release provide the processed downloads.

The bundle retains 45 transfer, 27 ablation and three CDK6 checkpoints, together with the other documented studies. The model implementation is unchanged. Raw artifacts and ZIP archives are excluded from Git source tracking.
