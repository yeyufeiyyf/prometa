# Publication release

Repository: https://github.com/yeyufeiyyf/prometa

Publication release: https://github.com/yeyufeiyyf/prometa/releases/tag/v2.0.0

The repository contains the ProMeta source code and documentation. Frozen weights,
data, support/query splits, reference predictions and provenance are distributed
separately in `prometa_artifacts_20261019_cdk6.zip` as a Release asset. The publication
release also provides `prometa_code_20261019_cdk6.zip` and `download_manifest.json`.
Raw artifacts and ZIP downloads remain excluded from the source Git tree.

The release contains 45 independently trained transfer checkpoints for the 15
settings, 27 ablation weights, and study-specific named checkpoint copies, including
all three CDK6 checkpoints and its 33-pair split. Existing transfer weights are
unchanged by the CDK6 addition. No comparator implementations or weights are included.

The main branch contains the current publication version. No separate archive
branch is maintained. The tagged release identifies the source and artifact
versions used for this publication.

Download verification and experiment invocation are documented in README and
[CHECKPOINT_API.md](CHECKPOINT_API.md). The SHA-256 manifest accompanies the downloads.
