#!/usr/bin/env python3
"""One-panel bidirectional K curve using the Plot Is All You Need direct-label style.

Visual reference: gallery/oa-geo-essd-17-965-2025-f05 in
https://github.com/liouhai/plot-is-all-you-need (direct line labels and
restrained grid). The 0-to-1 link is dashed because prototype provenance
switches from source-E3 references to target-E3 supports.
"""

from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb


ROOT = Path(os.environ["PROMETA_STUDY_WORKSPACE"]).resolve() / "k1_to_k7_20261007"
SUMMARY = ROOT / "k0_to_k5_paired_summary.csv"
PER_SEED = ROOT / "k0_to_k5_paired_per_seed.csv"
OUT = ROOT / "k0_to_k5_bidirectional_single_panel"

SERIES = (
    # Match the bidirectional Table S4 figure: ProMeta red and its dark ink.
    # Its blue encodes RF+ECFP, so reusing it for a transfer direction here
    # would give the same color two meanings in the manuscript.
    ("CRBN->VHL", "CRBN → VHL", "#B64A51", 0.958),
    ("VHL->CRBN", "VHL → CRBN", "#263845", 0.709),
)


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    per_seed = pd.read_csv(PER_SEED)
    assert len(summary) == 12 and len(per_seed) == 36
    assert set(summary.k) == set(range(6))
    assert per_seed.groupby(["direction", "k"]).seed.nunique().eq(3).all()
    assert per_seed.groupby(["direction", "seed"]).query_rows.nunique().eq(1).all()

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(118 / 25.4, 70 / 25.4))
    fig.subplots_adjust(left=0.13, right=0.84, bottom=0.22, top=0.94)

    # The light guides and an explicit chance line follow the reference's
    # grid-first, direct-label treatment without copying its complex layout.
    for y in (0.6, 0.7, 0.8, 0.9, 1.0):
        ax.axhline(y, color="#E8ECED", linewidth=0.52, zorder=0)
    ax.axhline(0.5, color="#A9B2B7", linewidth=0.75,
               linestyle=(0, (3, 2.5)), zorder=0)

    for key, label, color, label_y in SERIES:
        d = summary.loc[summary.direction.eq(key)].sort_values("k")
        raw = per_seed.loc[per_seed.direction.eq(key)]
        assert d.k.tolist() == list(range(6))
        k = d.k.to_numpy(dtype=float)
        mean = d.auroc.to_numpy(dtype=float)
        extrema = raw.groupby("k").auroc.agg(["min", "max"]).reindex(range(6))
        lower = mean - extrema["min"].to_numpy()
        upper = extrema["max"].to_numpy() - mean
        assert np.all(lower >= -1e-12) and np.all(upper >= -1e-12)

        whisker_color = tuple(0.55 * c + 0.45 for c in to_rgb(color))
        ax.errorbar(k, mean, yerr=[lower, upper], fmt="none", color=whisker_color,
                    linewidth=0.75, elinewidth=0.75, capsize=1.7,
                    capthick=0.75, zorder=2)
        ax.plot(k[:2], mean[:2], color=color, linewidth=1.35,
                linestyle=(0, (2.2, 2.1)), zorder=3)
        ax.plot(k[1:], mean[1:], color=color, linewidth=1.65,
                solid_capstyle="round", zorder=3)
        ax.scatter(k[1:], mean[1:], color=color, s=20,
                   edgecolor="white", linewidth=0.55, zorder=4)
        ax.scatter([k[0]], [mean[0]], s=27, facecolor="white",
                   edgecolor=color, linewidth=1.2, zorder=5)
        ax.text(5.18, label_y, label, color=color, fontsize=7.8,
                fontweight="medium", ha="left", va="center")

    ax.set_xlim(-0.23, 6.02)
    ax.set_ylim(0.41, 1.012)
    ax.set_xticks(range(6))
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.set_xlabel("Target-E3 support examples per class (K)", labelpad=7)
    ax.set_ylabel("AUROC", labelpad=6)
    ax.tick_params(axis="both", length=2.6, width=0.65,
                   color="#6B7780", pad=3.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#76858E")

    for ext, kwargs in (
        ("pdf", {}), ("eps", {}), ("png", {"dpi": 450}),
    ):
        fig.savefig(OUT.with_suffix(f".{ext}"), facecolor="white", **kwargs)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
