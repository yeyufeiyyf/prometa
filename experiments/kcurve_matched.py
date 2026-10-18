#!/usr/bin/env python3
"""Matched-query K=1..5 analysis from the completed seven-K evaluation."""
from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


ROOT = Path(os.environ["PROMETA_STUDY_WORKSPACE"]).resolve() / "k1_to_k7_20261007"
KS = (1, 2, 3, 4, 5)
SEEDS = (42, 2025, 3407)
DIRECTIONS = ("CRBN_to_VHL", "VHL_to_CRBN")
KEY = ["task", "row_id"]
COLOR = "#B64A51"


def metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for direction in DIRECTIONS:
        for seed in SEEDS:
            source = ROOT / "evaluation" / direction / f"seed_{seed}" / "query_predictions.csv"
            frame = pd.read_csv(source, dtype={"row_id": str})
            reference = frame.loc[frame.k_shot.eq(5), KEY + ["y_true"]].rename(
                columns={"y_true": "reference_label"})
            assert reference.reference_label.nunique() == 2
            assert len(reference) == len(reference.drop_duplicates(KEY))
            for k in KS:
                candidate = frame.loc[frame.k_shot.eq(k), KEY + ["y_true", "y_score"]]
                paired = reference.merge(candidate, on=KEY, validate="one_to_one")
                assert len(paired) == len(reference)
                assert paired.reference_label.eq(paired.y_true).all()
                truth, scores = paired.y_true, paired.y_score
                rows.append({
                    "direction": direction.replace("_to_", "->"),
                    "seed": seed, "k": k,
                    "tasks": paired.task.nunique(),
                    "query_rows": len(paired),
                    "positive_queries": int(truth.sum()),
                    "negative_queries": int(len(truth) - truth.sum()),
                    "auroc": roc_auc_score(truth, scores),
                    "auprc": average_precision_score(truth, scores),
                    "balacc": balanced_accuracy_score(truth, scores >= 0.5),
                })
    per_seed = pd.DataFrame(rows).sort_values(["direction", "seed", "k"])
    grouped = per_seed.groupby(["direction", "k"], sort=False)
    assert grouped.query_rows.nunique().eq(1).all()
    assert grouped.tasks.nunique().eq(1).all()
    summary = grouped.agg(
        seeds=("seed", "nunique"), tasks=("tasks", "first"),
        query_rows=("query_rows", "first"),
        positive_queries=("positive_queries", "first"),
        negative_queries=("negative_queries", "first"),
        auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc=("auprc", "mean"), auprc_sd=("auprc", "std"),
        balacc=("balacc", "mean"), balacc_sd=("balacc", "std"),
    ).reset_index()
    assert len(per_seed) == 30 and len(summary) == 10
    return per_seed, summary


def figure(per_seed: pd.DataFrame, summary: pd.DataFrame) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(178 / 25.4, 84 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.19,
                        top=0.80, wspace=0.22)
    for ax, direction, letter in zip(axes, DIRECTIONS, "ab"):
        name = direction.replace("_to_", "->")
        local = per_seed.loc[per_seed.direction.eq(name)]
        means = summary.loc[summary.direction.eq(name)].sort_values("k")
        for seed in SEEDS:
            points = local.loc[local.seed.eq(seed)].sort_values("k")
            ax.plot(points.k, points.auroc, color="#AEB8BD",
                    linewidth=0.9, marker="o", markersize=3.2, zorder=2)
        ax.plot(means.k, means.auroc, color=COLOR,
                linewidth=2.0, marker="D", markersize=5.0, zorder=3)
        ax.axhline(0.5, color="#ABB6BD", linewidth=0.8,
                   linestyle=(0, (3, 3)), zorder=0)
        ax.set_xlim(0.7, 5.3)
        ax.set_ylim(0.35, 1.02)
        ax.set_xticks(KS)
        ax.set_yticks(np.arange(0.4, 1.01, 0.2))
        ax.set_xlabel("Support examples per class (K)")
        ax.set_title(f"{letter}  {name.replace('->', ' → ')}", loc="left",
                     fontsize=10.5, fontweight="bold", pad=18)
        tasks, queries = int(means.tasks.iloc[0]), int(means.query_rows.iloc[0])
        ax.text(0, 1.045, f"{tasks} POIs · {queries} queries per seed",
                transform=ax.transAxes, ha="left", va="bottom",
                color="#61717C", fontsize=8.5)
        ax.tick_params(axis="both", labelsize=9, length=3)
        ax.spines["left"].set_color("#9EACB5")
        ax.spines["bottom"].set_color("#9EACB5")
    axes[0].set_ylabel("AUROC")
    for ext, kwargs in (("pdf", {}), ("eps", {}), ("svg", {}),
                        ("png", {"dpi": 300})):
        fig.savefig(ROOT / f"k1_to_k5_common_query_auroc.{ext}",
                    facecolor="white", **kwargs)
    plt.close(fig)


def main() -> None:
    per_seed, summary = metrics()
    per_seed.to_csv(ROOT / "k1_to_k5_paired_per_seed.csv", index=False)
    summary.to_csv(ROOT / "k1_to_k5_paired_summary.csv", index=False)
    figure(per_seed, summary)
    print(summary[["direction", "k", "tasks", "query_rows",
                   "auroc", "auroc_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
