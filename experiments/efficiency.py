"""ProMeta-only online profiling with the archived 33-compound/FLT3 workload."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from prometa.release_io import load_model,study_checkpoint,sha256


def measure(fn, warmup, repeats, cuda):
    for _ in range(warmup):fn()
    if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    values=[]
    for _ in range(repeats):
        if cuda:torch.cuda.synchronize()
        start=time.perf_counter_ns();fn()
        if cuda:torch.cuda.synchronize()
        values.append((time.perf_counter_ns()-start)/1e6)
    return values,torch.cuda.max_memory_allocated()/2**20 if cuda else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,choices=[42,2025,3407],default=42)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--repeats",type=int,default=100)
    p.add_argument("--task-repeats",type=int,default=30)
    p.add_argument("--validate-only",action="store_true",help="CPU input/scoring validation without timing or training")
    a=p.parse_args()
    if a.output.exists():p.error("Use a new output directory.")
    if min(a.repeats,a.task_repeats)<1 or a.warmup<0:p.error("Invalid repetition count.")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device("cpu" if a.validate_only else a.device)
    if not a.validate_only and device.type!="cuda":p.error("GPU profiling requires CUDA; use --validate-only for CPU scoring checks.")
    art=a.artifacts.resolve();checkpoint=study_checkpoint(art,"efficiency","FLT3",a.seed)
    before=sha256(checkpoint);model=load_model(checkpoint,device)
    inputs=torch.load(art/"efficiency/prometa_inputs.pt",map_location="cpu",weights_only=False)
    cohort=pd.read_csv(art/"efficiency/common_cohort.csv",dtype={"compound_id":str})
    assert len(cohort)==43 and len(inputs["graphs"])==43
    flt=cohort.index[cohort.family.eq("FLT3")].tolist()
    assert len(flt)==23
    support=[next(i for i in flt if cohort.loc[i,"label"]==1),next(i for i in flt if cohort.loc[i,"label"]==0)]
    query=[i for i in flt if i not in support]
    spos=[flt.index(i) for i in support];qpos=[flt.index(i) for i in query]
    def batch(ids,dev):
        return (Batch.from_data_list([inputs["graphs"][i] for i in ids]).to(dev),
                inputs["poi"][ids].to(dev),inputs["e3"][ids].to(dev))
    def encode(b,m=model):
        return m(b[0],poi_features=b[1],e3_features=b[2],return_diagnostics=False)["embedding"]
    def logits(s,y,q,m=model):return m.prototype_head(s,y,q)["logits"]
    y=inputs["labels"][support].to(device)
    # Verify the archived CPU path, including the same 16/16/1 chunking.
    cpu_model=copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        all_z=torch.cat([encode(batch(list(range(i,min(i+16,33))),"cpu"),cpu_model) for i in range(0,33,16)])
        z_cpu=all_z[flt]
        observed=logits(z_cpu[spos],y.cpu(),z_cpu[qpos],cpu_model).softmax(1)[:,1].numpy()
    frozen=pd.read_csv(art/"families/query_predictions.csv",dtype={"support_positive":str,"support_negative":str,"query":str})
    frozen=frozen[frozen.family.eq("FLT3") & frozen.seed.eq(a.seed) & frozen.support_positive.eq(cohort.loc[support[0],"compound_id"]) & frozen.support_negative.eq(cohort.loc[support[1],"compound_id"])]
    lookup=dict(zip(frozen["query"],frozen.prometa_score));expected=np.array([lookup[cohort.loc[i,"compound_id"]] for i in query])
    delta=float(abs(observed-expected).max());np.testing.assert_allclose(observed,expected,atol=1e-5,rtol=0)
    a.output.mkdir(parents=True)
    record=dict(seed=a.seed,checkpoint_sha256=before,archived_query_max_difference=delta,
                dtype="float32",tf32=False,cpu_threads=2,pool_compounds=33,support_compounds=2,query_compounds=21,
                preprocessing_and_IO_in_timer=False)
    if a.validate_only:
        record["status"]="CPU_SCORING_PASS";(a.output/"validation.json").write_text(json.dumps(record,indent=2)+"\n");print(record);return
    raw=[];metrics=[]
    def timed(name,fn,warmup,repeats,cuda=True,compounds=None):
        values,memory=measure(fn,warmup,repeats,cuda)
        raw.extend(dict(method="ProMeta",seed=a.seed,measurement=name,repetition=i,elapsed_ms=v) for i,v in enumerate(values))
        metrics.append(dict(method="ProMeta",seed=a.seed,measurement=name,median_ms=float(np.median(values)),peak_allocated_mib=memory,
            median_ms_per_compound=float(np.median(values)/compounds) if compounds else None,repeats=len(values)))
    with torch.no_grad():sz=encode(batch(support,device))
    for size in [1,16,33]:
        batches=[batch(list(range(i,min(i+size,33))),device) for i in range(0,33,size)]
        @torch.no_grad()
        def predict_pool():
            for b in batches:logits(sz,y,encode(b))
        timed(f"prediction_pool_batch{size}",predict_pool,a.warmup,a.repeats,compounds=33)
        del batches;torch.cuda.empty_cache()
    fb=[batch(flt[i:i+16],device) for i in range(0,23,16)]
    @torch.no_grad()
    def full_task():
        z=torch.cat([encode(b) for b in fb]);return logits(z[spos],y,z[qpos])
    timed("one_shot_task_gpu",full_task,3,a.task_repeats)
    @torch.no_grad()
    def cached_task():return logits(z_cpu[spos],y.cpu(),z_cpu[qpos],cpu_model)
    timed("cached_one_shot_task_cpu",cached_task,3,a.task_repeats,cuda=False)
    del fb,sz;torch.cuda.empty_cache()
    train=copy.deepcopy(model).train();optimizer=torch.optim.AdamW(train.parameters(),lr=1e-3,weight_decay=1e-5)
    tb=batch(list(range(33,43)),device);ty=inputs["labels"][33:43].to(device)
    def step():
        optimizer.zero_grad(set_to_none=True)
        z=encode(tb,train);loss=torch.nn.functional.cross_entropy(logits(z[:4],ty[:4],z[4:],train),ty[4:])
        loss.backward();torch.nn.utils.clip_grad_norm_(train.parameters(),5);optimizer.step()
    model.cpu();torch.cuda.empty_cache()
    timed("source_training_step",step,a.warmup,a.repeats)
    assert sha256(checkpoint)==before
    record.update(status="PROFILE_COMPLETE",device_name=torch.cuda.get_device_name(device),torch=torch.__version__,
                  warmup=a.warmup,repeats=a.repeats,task_repeats=a.task_repeats)
    pd.DataFrame(raw).to_csv(a.output/"raw_times.csv",index=False)
    pd.DataFrame(metrics).to_csv(a.output/"metrics.csv",index=False)
    (a.output/"manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))


if __name__=="__main__":main()
