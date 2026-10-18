#!/usr/bin/env python3
"""Classify query PROTACs with a frozen ProMeta encoder and labeled supports."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from prometa.data import prepare_contexts,build_context_graph_index
from prometa.experiment import ExperimentData,E3_UNIPROT_FALLBACK
from prometa.release_io import load_model,encode,predict_episode
from prometa.checkpoints import select_checkpoint,resolve_checkpoint,load_checkpoint,_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint",type=Path,help="Explicit checkpoint path (legacy interface)")
    selection.add_argument("--study",choices=["transfer","kcurve","poi","linker","families","cdk6","efficiency","rare_e3","rare"])
    p.add_argument("--artifacts",type=Path,default=Path(__file__).resolve().parent/"artifacts")
    p.add_argument("--direction",help="CRBN_to_VHL or VHL_to_CRBN")
    p.add_argument("--condition",help="Study condition, e.g. HPK1, FLT3 or common_multisource")
    p.add_argument("--k",type=int,help="Transfer support count per class")
    p.add_argument("--q",help="Transfer query count per class, or all")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--protein-bank",type=Path,help="Defaults to the frozen bank in --artifacts")
    p.add_argument("--support",type=Path,required=True)
    p.add_argument("--query",type=Path,required=True)
    p.add_argument("--source-e3",help="Required with --checkpoint; inferred with --study")
    p.add_argument("--e3-map",type=Path,help="Optional JSON: canonical E3 name to frozen protein-feature key")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    if a.output.exists():p.error("Use a new output file.")
    metadata_path=Path(str(a.output)+".metadata.json")
    if metadata_path.exists():p.error("Output metadata already exists; use a new output file.")
    selected=None
    if a.study:
        options=dict(study=a.study,seed=a.seed,direction=a.direction,
                     condition=a.condition,k=a.k,q=a.q)
        try:
            selected=select_checkpoint(a.artifacts,**options)
            a.checkpoint=resolve_checkpoint(a.artifacts,**options)
        except (ValueError,FileNotFoundError) as error:p.error(str(error))
        if a.source_e3 and a.source_e3.upper()!=selected.source_e3:
            p.error("--source-e3 conflicts with the selected checkpoint.")
        a.source_e3=selected.source_e3
    else:
        if not a.source_e3:p.error("--source-e3 is required with an explicit --checkpoint.")
        if any(value is not None for value in (a.direction,a.condition,a.k,a.q)):
            p.error("--direction/--condition/--k/--q require --study.")
    if a.protein_bank is None:
        a.protein_bank=a.artifacts/"current_transfer/prometa_as/data-prometa/embeddings/protein_embeddings_fair_v1.pt"
    if not a.checkpoint.is_file():p.error(f"Checkpoint not found: {a.checkpoint}")
    if not a.protein_bank.is_file():p.error(f"Protein bank not found: {a.protein_bank}. Extract the matching artifact archive.")
    torch.set_num_threads(2)
    support=pd.read_csv(a.support,dtype={"Compound ID":str})
    query=pd.read_csv(a.query,dtype={"Compound ID":str})
    if "Label_bin" not in support or set(support.Label_bin)!={0,1}:p.error("Support must contain both binary activity classes in Label_bin.")
    required=["Compound ID","Uniprot","Target","E3 ligase","Smiles"]
    if any(c not in support or c not in query for c in required):p.error("Both CSVs require: "+", ".join(required))
    # Query labels are never used for scoring; a local placeholder only satisfies the graph-table schema.
    query_input=query.copy();query_input["Label_bin"]=0
    prepared=prepare_contexts(pd.concat([support,query_input],ignore_index=True))
    sid=prepared.iloc[:len(support)].row_id.tolist();qid=prepared.iloc[len(support):].row_id.tolist()
    if set(sid)&set(qid):p.error("Support and query contexts overlap.")
    if prepared.row_id.duplicated().any():p.error("Repeated compound/E3/POI context.")
    if len(prepared[["Uniprot","canonical_e3"]].drop_duplicates())!=1:
        p.error("One prediction call must contain one shared POI/E3 task. Run different tasks separately.")
    bank=torch.load(a.protein_bank,map_location="cpu",weights_only=False)
    e3map=dict(E3_UNIPROT_FALLBACK)
    if a.e3_map:e3map.update(json.loads(a.e3_map.read_text()))
    missing=set(prepared.Uniprot)-set(bank)
    missing_e3={e for e in prepared.canonical_e3 if e not in e3map or e3map[e] not in bank}
    if missing or missing_e3:p.error(f"Missing frozen features: POI={sorted(missing)}, E3={sorted(missing_e3)}")
    graphs=build_context_graph_index(prepared,invalid_molecule="raise")
    data=ExperimentData(strict_contexts=prepared,training_contexts=prepared.iloc[:0],graph_index=graphs.graphs,
        graph_audit=graphs.audit,row_lookup=prepared.set_index("row_id",drop=False),protein_bank=bank,
        e3_uniprot=e3map,protein_dim=next(iter(bank.values())).numel(),audit={"inference_only":True})
    model=load_checkpoint(a.artifacts,device=a.device,**options) if selected else load_model(a.checkpoint,a.device)
    if selected and selected.namespace=="rare":
        from prometa.rare.experiment import model_forward
        with torch.no_grad():
            s=model_forward(model,sid,data,device=torch.device(a.device),intervention="correct",source_e3=a.source_e3,seed=a.seed)[0]
            q=model_forward(model,qid,data,device=torch.device(a.device),intervention="correct",source_e3=a.source_e3,seed=a.seed)[0]
    else:
        s=encode(model,sid,data,a.source_e3,a.seed,a.device);q=encode(model,qid,data,a.source_e3,a.seed,a.device)
    scores=predict_episode(model,s,support.Label_bin.to_numpy(int),q)
    result=query.copy();result["prometa_active_score"]=scores;result["predicted_label"]=(scores>=.5).astype(int)
    a.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.output,index=False)
    metadata=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=_digest(a.checkpoint),
        study=selected.study if selected else None,condition=selected.condition if selected else None,
        direction=selected.direction if selected else None,seed=a.seed,
        k=selected.k if selected else None,q=selected.q if selected else None,
        source_e3=a.source_e3,protein_bank_sha256=_digest(a.protein_bank),
        support_sha256=_digest(a.support),query_sha256=_digest(a.query),
        support_rows=len(support),query_rows=len(query),query_labels_used=False,
        score="softmax probability for dataset class Label_bin=1",threshold=0.5)
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Predicted {len(query)} queries from {len(support)} labeled references.")


if __name__=="__main__":main()
