#!/usr/bin/env python3
"""Recompute the manuscript CDK6 ProMeta result from original frozen predictions."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def calculate(inputs, output, predictions=None):
    ROOT = Path(inputs).resolve()
    members = pd.read_csv(ROOT / "cdk6_crbn_members.csv", dtype={"Compound ID": str})
    supports = pd.read_csv(ROOT / "cdk6_crbn_supports.csv",
                          dtype={"positive_support": str, "negative_support": str})
    cohort = pd.read_csv(ROOT / "cdk6_contexts.csv", dtype={"Compound ID": str})
    assert len(members) == 14 and members["Compound ID"].nunique() == 14
    assert members.Label_bin.sum() == 11
    assert len(cohort) == 14 and cohort.Smiles.notna().all()
    assert set(cohort["Compound ID"]) == set(members["Compound ID"])
    labels = members.set_index("Compound ID").Label_bin.to_dict()
    assert np.array_equal(cohort.Label_bin, cohort["Compound ID"].map(labels))
    assert len(supports) == 33 and not supports.episode.duplicated().any()
    pos = members.loc[members.Label_bin.eq(1), "Compound ID"]
    neg = members.loc[members.Label_bin.eq(0), "Compound ID"]
    assert set(zip(supports.positive_support, supports.negative_support)) == set(itertools.product(pos, neg))
    prediction_root = ROOT / "predictions" if predictions is None else Path(predictions)
    predictions = pd.concat([pd.read_csv(prediction_root / f"seed_{seed}.csv",
                              dtype={"compound_id": str}) for seed in (42, 2025, 3407)], ignore_index=True)
    assert len(predictions) == 1188 and set(predictions.seed) == {42, 2025, 3407}
    assert not predictions.duplicated(["seed", "episode", "compound_id"]).any()
    assert np.isfinite(predictions.y_score).all()
    rows = []
    for (seed, episode), block in predictions.groupby(["seed", "episode"]):
        pair = supports.set_index("episode").loc[episode]
        expected = set(members["Compound ID"]) - {pair.positive_support, pair.negative_support}
        assert len(block) == 12 and set(block.compound_id) == expected
        assert np.array_equal(block.y_true, block.compound_id.map(labels))
        assert block.y_true.sum() == 10
        rows.append(dict(seed=int(seed), episode=int(episode),
                         support_positive=pair.positive_support, support_negative=pair.negative_support,
                         AUROC=roc_auc_score(block.y_true, block.y_score),
                         AUPRC=average_precision_score(block.y_true, block.y_score)))
    pairs = pd.DataFrame(rows)
    assert len(pairs) == 99
    per_seed = pairs.groupby("seed")[["AUROC", "AUPRC"]].mean()
    per_pair = pairs.groupby("episode")[["AUROC", "AUPRC"]].mean()
    result = dict(status="PASS",mean_query_auroc=float(per_seed.AUROC.mean()),
        sd_across_seed_means=float(per_seed.AUROC.std(ddof=1)),
        median_support_pair_auroc_averaged_over_seeds=float(per_pair.AUROC.median()),
        seeds=3, compounds=14, support_pairs=33, query_rows=1188,
        reproduction_level="metrics recomputed from original predictions" if prediction_root == ROOT / "predictions"
                           else "metrics recomputed from regenerated checkpoint predictions")
    assert round(result["mean_query_auroc"], 3) == .886
    assert round(result["sd_across_seed_means"], 3) == .008
    assert round(result["median_support_pair_auroc_averaged_over_seeds"], 3) == .933
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output / "support_pair_seed_metrics.csv", index=False)
    per_seed.to_csv(output / "per_seed_summary.csv")
    per_pair.to_csv(output / "support_pair_seed_averaged.csv")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    result = calculate(args.inputs, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
