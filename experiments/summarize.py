"""Aggregate only ProMeta predictions using each paper study's exact weighting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score


def binary(frame, label="y_true", score="y_score"):
    y=frame[label].to_numpy(int);s=frame[score].to_numpy(float)
    if set(y)!={0,1}:raise ValueError("AUROC requires both query classes.")
    return dict(auroc=roc_auc_score(y,s),auprc=average_precision_score(y,s),
                balacc=balanced_accuracy_score(y,s>=.5),query_rows=len(frame))


def grouped(frame, keys, **kwargs):
    return pd.DataFrame([dict(zip(keys,k if isinstance(k,tuple) else (k,)),**binary(b,**kwargs))
                        for k,b in frame.groupby(keys,sort=True)])


def write(frame, keys, output, stem):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/f"{stem}_per_seed.csv",index=False)
    values=[x for x in ["auroc","auprc","balacc"] if x in frame]
    summary=frame.groupby(keys,as_index=False).agg(
        **{f"{v}_{stat}":(v,stat) for v in values for stat in ["mean","std"]},
        seeds=("seed","nunique"))
    summary.to_csv(output/f"{stem}_summary.csv",index=False)
    return summary


def transfer(study, output):
    rows=[]
    for p in sorted((Path(study)/"table_s4_prometa_rerun/evaluation").rglob("query_predictions.csv")):
        direction,seed,setting=p.parts[-4:-1]
        k,q=setting.split("_q");frame=pd.read_csv(p)
        rows.append(dict(direction=direction.replace("_to_","->"),seed=int(seed[5:]),k=int(k[1:]),q=int(q),**binary(frame)))
    result=pd.DataFrame(rows)
    assert len(result)==45
    return write(result,["direction","k","q"],output,"transfer")


def kcurve(study, output):
    base=Path(study)/"k1_to_k7_20261007";rows=[]
    zero=pd.read_csv(base/"k0_query_predictions.csv")
    for (direction,seed),b in zero.groupby(["direction","seed"]):
        rows.append(dict(direction=direction,seed=int(seed),k=0,**binary(b)))
    for p in sorted((base/"evaluation").rglob("query_predictions.csv")):
        direction,seed=p.parts[-3:-1];frame=pd.read_csv(p)
        reference=frame[frame.k_shot.eq(5)][["task","row_id","y_true"]].rename(columns={"y_true":"reference_label"})
        for k in range(1,6):
            pair=reference.merge(frame[frame.k_shot.eq(k)][["task","row_id","y_true","y_score"]],
                                 on=["task","row_id"],validate="one_to_one")
            assert len(pair)==len(reference) and pair.reference_label.eq(pair.y_true).all()
            rows.append(dict(direction=direction.replace("_to_","->"),seed=int(seed[5:]),k=k,**binary(pair)))
    per=pd.DataFrame(rows)
    assert len(per)==36 and per.groupby(["direction","seed"]).query_rows.nunique().eq(1).all()
    return write(per,["direction","k"],output,"kcurve")


def poi(frame, output):
    keys=["direction","seed","task","k","row_id"]
    # Average the four K=1 scores for each fixed query before computing AUROC.
    assert frame.method.eq("ProMeta+prototype").all()
    pooled=frame.groupby(keys,as_index=False).agg(y_true=("y_true","first"),y_score=("y_score","mean"))
    per_task=grouped(pooled,["direction","seed","task","k"])
    Path(output).mkdir(parents=True,exist_ok=True);per_task.to_csv(Path(output)/"poi_per_task_seed.csv",index=False)
    per=per_task.groupby(["direction","seed","k"],as_index=False)[["auroc","auprc","balacc"]].mean()
    return write(per,["direction","k"],output,"poi")


def linker(frame, output):
    keys=["direction","seed","group_id","support_positive_id","support_negative_id"]
    episodes=grouped(frame,keys,label="label",score="prometa_score")
    series=episodes.groupby(["direction","seed","group_id"],as_index=False)[["auroc","auprc","balacc"]].mean()
    per=series.groupby("seed",as_index=False)[["auroc","auprc","balacc"]].mean()
    per["scope"]="nine_series_equal"
    summary=write(per,["scope"],output,"linker")
    episodes.to_csv(Path(output)/"linker_support_pair_metrics.csv",index=False)
    series.to_csv(Path(output)/"linker_per_series_seed.csv",index=False)
    if "k0_score" in frame:
        zero=grouped(frame,keys,label="label",score="k0_score")
        zero_series=zero.groupby(["seed","group_id"])[["auroc","auprc","balacc"]].mean()
        zero_per=zero_series.groupby("seed").mean().reset_index();zero_per["scope"]="nine_series_equal"
        write(zero_per,["scope"],output,"linker_source_prototypes")
    btk=frame[frame.group_id.eq("ls_764bc1ed5a0ed5")].copy()
    btk["rank_percentile"]=btk.groupby(keys).prometa_score.rank(method="average",pct=True)
    ranks=btk.groupby(["query_id","label"],as_index=False).rank_percentile.mean()
    assert len(ranks)==11 and ranks[ranks.label.eq(1)].rank_percentile.min()>ranks[ranks.label.eq(0)].rank_percentile.max()
    ranks.to_csv(Path(output)/"linker_BTK_query_rank_percentiles.csv",index=False)
    return summary


def families(frame, output):
    keys=["family","seed","support_positive","support_negative"]
    pairs=grouped(frame,keys,score="prometa_score")
    per=pairs.groupby(["family","seed"],as_index=False)[["auroc","auprc","balacc"]].mean()
    return write(per,["family"],output,"families")


def rare(frame, output):
    episodes=grouped(frame,["seed","task","episode_id"])
    per_task=episodes.groupby(["seed","task"],as_index=False)[["auroc","auprc","balacc"]].mean()
    per=per_task.groupby("seed",as_index=False)[["auroc","auprc","balacc"]].mean();per["scope"]="four_ligases_equal"
    per_task.to_csv(Path(output)/"rare_per_ligase_seed.csv",index=False)
    write(per_task,["task"],output,"rare_per_ligase")
    return write(per,["scope"],output,"rare")


def efficiency(artifacts, output):
    base=Path(artifacts)/"efficiency"
    raw=pd.read_csv(base/"all_raw_times.csv")
    assert raw.method.eq("ProMeta").all() and len(raw)==1380
    per=raw.groupby(["seed","measurement"],as_index=False).elapsed_ms.median()
    ref=pd.read_csv(base/"per_seed_metrics.csv")
    matched=per.merge(ref,on=["seed","measurement"],validate="one_to_one")
    np.testing.assert_allclose(matched.elapsed_ms,matched.median_ms,atol=1e-10,rtol=0)
    result=per.groupby("measurement",as_index=False).elapsed_ms.agg(["mean","std"]).reset_index()
    result.to_csv(Path(output)/"efficiency_recomputed.csv",index=False)
    return result


def archived(artifacts, output, studies):
    artifacts=Path(artifacts);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results={}
    if "transfer" in studies:results["transfer"]=transfer(artifacts/"current_transfer",output)
    if "kcurve" in studies:results["kcurve"]=kcurve(artifacts/"current_transfer",output)
    if "rare" in studies:
        pred=pd.concat([pd.read_csv(p) for p in sorted((artifacts/"rare_e3/results/prometa2_multisource_loeo_v1/common_multisource").glob("seed_*/query_predictions.csv"))])
        results["rare"]=rare(pred,output)
    if "poi" in studies:
        pred=pd.concat([pd.read_csv(p) for p in sorted((artifacts/"poi").glob("*/seed_*/query_predictions.csv"))])
        results["poi"]=poi(pred,output)
    if "linker" in studies:results["linker"]=linker(pd.read_csv(artifacts/"linker/k0_k1_predictions.csv"),output)
    if "families" in studies:results["families"]=families(pd.read_csv(artifacts/"families/query_predictions.csv"),output)
    if "efficiency" in studies:results["efficiency"]=efficiency(artifacts,output)
    return results
