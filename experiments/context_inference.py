"""ProMeta-only inference for POI, matched linkers and HPK1/FLT3 families."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from prometa.release_io import (DIRECTIONS,PAPER_SEEDS,SEEDS,training_data,study_checkpoint,
    source_reference,load_model,encode,select,predict_episode,attach_family_contexts,linker_data)


def poi(artifacts, output, device):
    rows=[]
    for source,target in DIRECTIONS:
        data=training_data(artifacts,source,target)
        for seed in SEEDS:
            path=artifacts/f"poi/{source}_to_{target}/seed_{seed}/episodes.json"
            episodes=json.loads(path.read_text())
            model=load_model(study_checkpoint(artifacts,"poi",f"{source}_to_{target}",seed),device)
            ref,ref_y=source_reference(artifacts,data,source,target,seed)
            for ep in episodes:
                ids=ep["k2_support_row_ids"]
                pos=[rid for rid in ids if data.row_lookup.loc[rid,"Label_bin"]==1]
                neg=[rid for rid in ids if data.row_lookup.loc[rid,"Label_bin"]==0]
                choices=[(0,ref,"source_only")]
                choices += [(1,[p,n],f"k1_subset_{i}") for i,(p,n) in enumerate(itertools.product(pos,neg))]
                choices += [(2,ids,"k2")]
                qs=ep["query_row_ids"];q=encode(model,qs,data,source,seed,device)
                for k,si,tag in choices:
                    assert k==0 or not set(si)&set(qs)
                    sy=ref_y if k==0 else [int(data.row_lookup.loc[rid,"Label_bin"]) for rid in si]
                    prob=predict_episode(model,encode(model,si,data,source,seed,device),sy,q)
                    for rid,p in zip(qs,prob):rows.append(dict(direction=f"{source}->{target}",seed=seed,task=ep["task"],
                        k=k,support_variant=tag,method="ProMeta+prototype",row_id=rid,
                        y_true=int(data.row_lookup.loc[rid,"Label_bin"]),y_score=float(p)))
            print(f"POI {source}->{target} seed={seed}",flush=True)
    pd.DataFrame(rows).to_csv(output/"query_predictions.csv",index=False)


def families(artifacts, output, device):
    data=training_data(artifacts,"VHL","CRBN")
    cohort=pd.read_csv(artifacts/"families/cohort.csv",dtype={"compound_id":str})
    cohort=attach_family_contexts(data,cohort)
    rows=[]
    for seed in PAPER_SEEDS:
        for family,b in cohort.groupby("family",sort=False):
            model=load_model(study_checkpoint(artifacts,"families",family,seed),device)
            b=b.reset_index(drop=True);labels=b.label.to_numpy(int)
            z=encode(model,b.row_id.tolist(),data,"VHL",seed,device)
            for p,n in itertools.product(np.flatnonzero(labels==1),np.flatnonzero(labels==0)):
                qs=[i for i in range(len(b)) if i not in [p,n]]
                prob=predict_episode(model,select(z,[p,n]),[1,0],select(z,qs))
                for i,v in zip(qs,prob):rows.append(dict(family=family,seed=seed,support_positive=str(b.loc[p,"compound_id"]),
                    support_negative=str(b.loc[n,"compound_id"]),query=str(b.loc[i,"compound_id"]),y_true=int(labels[i]),prometa_score=float(v)))
        print(f"Families seed={seed}",flush=True)
    pd.DataFrame(rows).to_csv(output/"query_predictions.csv",index=False)


def linker(artifacts, output, device):
    frame=pd.read_csv(artifacts/"linker/cohort.csv")
    old=pd.read_csv(artifacts/"linker/query_predictions.csv")
    assert old.group_id.nunique()==9 and frame.row_id.nunique()==85
    rows=[]
    for source,target in DIRECTIONS:
        source_data=training_data(artifacts,source,target)
        external=linker_data(artifacts,frame[frame.canonical_e3.eq(target)].copy())
        ids=external.row_lookup.index.astype(str).tolist();index={rid:i for i,rid in enumerate(ids)}
        for seed in PAPER_SEEDS:
            model=load_model(study_checkpoint(artifacts,"linker",f"{source}_to_{target}",seed),device)
            chunks=[encode(model,ids[i:i+32],external,source,seed,device) for i in range(0,len(ids),32)]
            z={key:torch.cat([c[key] for c in chunks]) for key in ["embedding","metric_scale"] if chunks[0].get(key) is not None}
            ref,ref_y=source_reference(artifacts,source_data,source,target,seed)
            ref_chunks=[encode(model,ref[i:i+32],source_data,source,seed,device) for i in range(0,len(ref),32)]
            rz={key:torch.cat([c[key] for c in ref_chunks]) for key in ["embedding","metric_scale"] if ref_chunks[0].get(key) is not None}
            zero=dict(zip(ids,predict_episode(model,rz,ref_y,z)))
            local=old[old.direction.eq(f"{source}->{target}") & old.seed.eq(seed)]
            for (group,p,n),b in local.groupby(["group_id","support_positive_id","support_negative_id"]):
                qs=b.query_id.tolist()
                assert not set(qs)&{p,n}
                lookup=frame.set_index("row_id")
                assert not set(lookup.loc[qs,"canonical_linker"]) & set(lookup.loc[[p,n],"canonical_linker"])
                probs=predict_episode(model,select(z,[index[p],index[n]]),[1,0],select(z,[index[q] for q in qs]))
                for q,y,v in zip(qs,b.label,probs):rows.append(dict(direction=f"{source}->{target}",seed=seed,group_id=group,
                    support_positive_id=p,support_negative_id=n,query_id=q,label=int(y),prometa_score=float(v),k0_score=float(zero[q])))
            print(f"Linker {source}->{target} seed={seed}",flush=True)
    pd.DataFrame(rows).to_csv(output/"query_predictions.csv",index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment",choices=["poi","linker","families"])
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    args=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if args.output.exists():p.error("Use a new output directory.")
    args.output.mkdir(parents=True)
    globals()[args.experiment](args.artifacts.resolve(),args.output.resolve(),args.device)


if __name__=="__main__":main()
