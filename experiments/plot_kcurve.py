#!/usr/bin/env python3
"""Compare source-reference K0 with matched-query target-support K1..5."""
from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(os.environ["PROMETA_STUDY_WORKSPACE"]).resolve() / "k1_to_k7_20261007"
RED = "#B64A51"
BLUE = "#456D83"
GRAY = "#B3BDC3"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    zero = pd.read_csv(ROOT / "k0_per_seed_metrics.csv")
    zero["prototype_source"] = "source_E3_training"
    few = pd.read_csv(ROOT / "k1_to_k5_paired_per_seed.csv")
    few["prototype_source"] = "target_E3_support"
    per_seed = pd.concat([zero, few], ignore_index=True)
    summary = per_seed.groupby(["direction", "k", "prototype_source"],
                               sort=False).agg(
        seeds=("seed", "nunique"), tasks=("tasks", "first"),
        query_rows=("query_rows", "first"),
        auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc=("auprc", "mean"), auprc_sd=("auprc", "std"),
        balacc=("balacc", "mean"), balacc_sd=("balacc", "std"),
    ).reset_index()
    assert len(per_seed) == 36 and len(summary) == 12
    assert per_seed.groupby(["direction", "seed"]).query_rows.nunique().eq(1).all()
    assert per_seed.groupby(["direction", "seed"]).tasks.nunique().eq(1).all()
    return per_seed, summary


def main() -> None:
    per_seed, summary = load()
    per_seed.to_csv(ROOT / "k0_to_k5_paired_per_seed.csv", index=False)
    summary.to_csv(ROOT / "k0_to_k5_paired_summary.csv", index=False)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(178 / 25.4, 88 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.19,
                        top=0.77, wspace=0.22)
    for ax, name, letter in zip(axes, ["CRBN->VHL", "VHL->CRBN"], "ab"):
        local = per_seed.loc[per_seed.direction.eq(name)]
        means = summary.loc[summary.direction.eq(name)].sort_values("k")
        zero_mean = means.loc[means.k.eq(0)].iloc[0]
        for seed in (42, 2025, 3407):
            points = local.loc[(local.seed.eq(seed)) & (local.k.ge(1))].sort_values("k")
            ax.plot(points.k, points.auroc, color=GRAY,
                    linewidth=0.9, marker="o", markersize=3.1, zorder=2)
        ax.scatter(local.loc[local.k.eq(0)].k,
                   local.loc[local.k.eq(0)].auroc,
                   s=19, facecolor="white", edgecolor=BLUE,
                   linewidth=0.9, zorder=3)
        ax.scatter([0], [zero_mean.auroc], color=BLUE,
                   marker="s", s=45, edgecolor="white", linewidth=0.6,
                   zorder=5)
        target_mean = means.loc[means.k.ge(1)]
        ax.plot(target_mean.k, target_mean.auroc, color=RED,
                linewidth=1.9, marker="D", markersize=5.0, zorder=4)
        ax.axhline(0.5, color="#ABB6BD", linewidth=0.8,
                   linestyle=(0, (3, 3)), zorder=0)
        ax.set_xlim(-0.32, 5.3)
        ax.set_ylim(0.35, 1.02)
        ax.set_xticks(range(0, 6))
        ax.set_yticks(np.arange(0.4, 1.01, 0.2))
        ax.set_xlabel("Support examples per class (K)")
        ax.set_title(f"{letter}  {name.replace('->', ' → ')}",
                     loc="left", fontsize=10.5, fontweight="bold", pad=18)
        tasks, queries = int(means.tasks.iloc[0]), int(means.query_rows.iloc[0])
        ax.text(0, 1.045, f"{tasks} POIs · {queries} queries per seed",
                transform=ax.transAxes, ha="left", va="bottom",
                color="#61717C", fontsize=8.5)
        ax.tick_params(axis="both", labelsize=9, length=3)
        ax.spines["left"].set_color("#9EACB5")
        ax.spines["bottom"].set_color("#9EACB5")
    axes[0].set_ylabel("AUROC")
    handles = [
        Line2D([0], [0], color=BLUE, marker="s", linewidth=0,
               markersize=6, label="K=0: source references"),
        Line2D([0], [0], color=RED, marker="D", linewidth=1.7,
               markersize=5, label="K=1–5: target support"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.52, 0.99), frameon=False,
               fontsize=8.3, handletextpad=0.4, columnspacing=2.2)
    for ext, kwargs in (("pdf", {}), ("eps", {}), ("svg", {}),
                        ("png", {"dpi": 300})):
        fig.savefig(ROOT / f"k0_to_k5_common_query_auroc.{ext}",
                    facecolor="white", **kwargs)
    plt.close(fig)
    print(summary[["direction", "k", "auroc", "auroc_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
