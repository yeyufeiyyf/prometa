"""Plot ProMeta-only study summaries; comparator panels are outside this release."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLOR="#B64A51"


def save(fig,out,name):
    fig.tight_layout()
    for ext in ["pdf","png"]:fig.savefig(out/(name+"."+ext),dpi=300,bbox_inches="tight")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True,help="Output directory of reproduce.py")
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    summary=a.run/("regenerated_summary" if (a.run/"regenerated_summary").exists() else "summary")
    if (summary/"kcurve_summary.csv").exists():
        means=pd.read_csv(summary/"kcurve_summary.csv");raw=pd.read_csv(summary/"kcurve_per_seed.csv")
        fig,axes=plt.subplots(1,2,figsize=(7,2.8),sharey=True)
        for ax,direction in zip(axes,["CRBN->VHL","VHL->CRBN"]):
            b=means[means.direction.eq(direction)].sort_values("k")
            ranges=raw[raw.direction.eq(direction)].groupby("k").auroc.agg(["min","max"]).reindex(b.k)
            ax.errorbar(b.k,b.auroc_mean,yerr=[b.auroc_mean.to_numpy()-ranges["min"].to_numpy(),ranges["max"].to_numpy()-b.auroc_mean.to_numpy()],fmt="none",color=COLOR,capsize=2)
            ax.plot(b.k.iloc[:2],b.auroc_mean.iloc[:2],"--",color=COLOR)
            ax.plot(b.k.iloc[1:],b.auroc_mean.iloc[1:],"o-",color=COLOR)
            ax.scatter(b.k.iloc[0],b.auroc_mean.iloc[0],facecolors="white",edgecolors=COLOR)
            ax.set(title=direction.replace("->"," → "),xlabel="Support compounds per class (K)",ylim=(.35,1.02))
            ax.axhline(.5,ls="--",color="#AAAAAA",lw=.7)
        axes[0].set_ylabel("AUROC");save(fig,a.output,"prometa_kcurve")
    if (summary/"families_summary.csv").exists():
        b=pd.read_csv(summary/"families_summary.csv");b=b[b.family.isin(["HPK1","FLT3"])]
        fig,ax=plt.subplots(figsize=(3.2,2.5));ax.bar(b.family,b.auroc_mean,yerr=b.auroc_std,color=COLOR,capsize=3)
        ax.set(ylabel="Mean query AUROC",ylim=(0,1));ax.axhline(.5,ls="--",color="#AAAAAA",lw=.7)
        save(fig,a.output,"prometa_families")
    if (a.run/"cdk6/support_pair_seed_averaged.csv").exists():
        b=pd.read_csv(a.run/"cdk6/support_pair_seed_averaged.csv")
        fig,ax=plt.subplots(figsize=(2.8,2.8));v=ax.violinplot([b.AUROC],showmedians=True)
        for body in v["bodies"]:body.set_facecolor(COLOR);body.set_alpha(.8)
        ax.set(xticks=[1],xticklabels=["ProMeta"],ylabel="Query AUROC",ylim=(0,1.03))
        ax.axhline(.5,ls="--",color="#AAAAAA",lw=.7);save(fig,a.output,"prometa_cdk6_support_pairs")
    if (summary/"linker_BTK_query_rank_percentiles.csv").exists():
        b=pd.read_csv(summary/"linker_BTK_query_rank_percentiles.csv")
        order=pd.read_csv(a.artifacts/"linker/compound_display_mapping.csv")
        b=order[["query_id","display_position","display_label"]].merge(b,on="query_id",validate="one_to_one").sort_values("display_position")
        fig,ax=plt.subplots(figsize=(6.5,2.8));ax.plot(b.display_position,b.rank_percentile*100,"o-",color=COLOR)
        ax.set(xticks=b.display_position,xticklabels=b.display_label,ylabel="Mean query rank percentile (%)",xlabel="Compound",ylim=(0,105))
        ax.tick_params(axis="x",labelrotation=45);ax.axvline(7.5,ls="--",color="#AAAAAA",lw=.7)
        save(fig,a.output,"prometa_btk_query_ranks")
    print(a.output)


if __name__=="__main__":main()
