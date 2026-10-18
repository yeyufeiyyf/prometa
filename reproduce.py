#!/usr/bin/env python3
"""Recompute paper metrics or regenerate ProMeta predictions in a fresh directory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
STUDIES=("transfer","kcurve","rare","poi","linker","families","cdk6","efficiency")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",choices=("all",)+STUDIES,default="all")
    p.add_argument("--mode",choices=["metrics","infer","train"],default="metrics")
    p.add_argument("--artifacts",type=Path,default=ROOT/"artifacts")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args()
    studies=list(STUDIES) if args.experiment=="all" else [args.experiment]
    art=args.artifacts.resolve();out=args.output.resolve()
    if args.workers<1:p.error("--workers must be positive.")
    if out.exists():p.error("--output must be a new directory; existing runs are never overwritten.")
    if out==art or art in out.parents:p.error("Do not write inside the reference artifact bundle.")
    if not (art/"MANIFEST.json").is_file():p.error("Extract the ProMeta artifact download into artifacts/ first.")
    if args.mode=="train" and args.experiment not in ["rare"]:
        p.error("The current 15-setting transfer release supports frozen-weight inference and metric recomputation; use --experiment rare for source training.")
    if args.mode=="infer" and args.experiment=="efficiency":
        p.error("Efficiency supplies measured timings. Use --mode metrics or the dedicated GPU profiler.")
    subprocess.run([sys.executable,str(ROOT/"verify_release.py"),"--artifacts",str(art)],check=True)
    out.mkdir(parents=True)
    env=os.environ.copy()
    env.update(PYTHONPATH=str(ROOT),MPLBACKEND="Agg",MPLCONFIGDIR=str(out/".matplotlib"),
               PYTHONDONTWRITEBYTECODE="1",PROMETA_ARTIFACTS_ROOT=str(art))
    for key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:env[key]="1"
    started=time.time();levels={};logpath=out/"reproduction.log"
    with logpath.open("w") as log:
        def run(script,*arguments):
            cmd=[sys.executable,str(ROOT/"experiments"/script),*map(str,arguments)]
            print("RUN",script,flush=True)
            log.write("RUN "+script+" "+" ".join(map(str,arguments))+"\n");log.flush()
            subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if args.mode=="infer" and "transfer" in studies:
            run("transfer_setting_fresh.py","--artifacts",art,"--mode","infer",
                "--output",out/"transfer_setting_fresh")
            levels["transfer"]="regenerated_setting_specific_checkpoint_predictions"
        if args.mode in ["infer","train"] and "kcurve" in studies:
            work=out/"current_transfer"
            shutil.copytree(art/"current_transfer",work)
            env["PROMETA_STUDY_WORKSPACE"]=str(work)
            if "kcurve" in studies:
                for flag in (work/"k1_to_k7_20261007/evaluation").rglob("_SUCCESS"):flag.unlink()
                run("kcurve.py","all","--workers",args.workers)
                run("kcurve_zero.py")
                run("kcurve_matched.py")
                run("plot_kcurve.py")
                run("plot_kcurve_single.py")
                levels["kcurve"]="regenerated_checkpoint_predictions"
        if args.mode=="train" and "rare" in studies:
            from prometa.release_io import SEEDS
            for seed in SEEDS:
                run("rare_e3.py","train","--artifacts",art,"--output",out/f"rare_train/seed_{seed}","--seed",seed,"--device",args.device)
            levels["rare"]="fresh_multisource_training"
        if args.mode=="infer":
            if "cdk6" in studies:
                run("cdk6_inference.py","--artifacts",art,"--output",out/"cdk6","--device",args.device)
                levels["cdk6"]="regenerated_checkpoint_predictions"
            for study in ["rare","poi","linker","families"]:
                if study not in studies:continue
                script="rare_e3.py" if study=="rare" else "context_inference.py"
                action="infer" if study=="rare" else study
                run(script,action,"--artifacts",art,"--output",out/study,"--device",args.device)
                levels[study]="regenerated_checkpoint_predictions"
        if args.mode=="train":
            record=dict(status="training_completed",levels=levels,elapsed_seconds=time.time()-started,
                        note="Fresh source training; exact equality to the archived weights is not claimed.")
        else:
            from experiments import summarize
            # All original numerical references are recomputed before comparing inference.
            summary=out/"summary";summary.mkdir()
            legacy_studies=[study for study in studies if study!="transfer"]
            source_results=summarize.archived(art,summary,legacy_studies)
            if "transfer" in studies:
                run("transfer_setting_fresh.py","--artifacts",art,"--mode","metrics",
                    "--output",out/"transfer_metric_check")
                for filename in ("transfer_setting_fresh_summary.csv","transfer_setting_fresh_per_seed.csv"):
                    shutil.copy2(out/"transfer_metric_check"/filename,summary/filename)
                if "transfer" not in levels:levels["transfer"]="recomputed_from_setting_specific_predictions"
            for study in studies:
                if study not in levels:levels[study]="recomputed_from_archived_predictions" if study!="efficiency" else "recomputed_from_archived_timing_samples"
            if args.mode=="infer":
                destination=out/"regenerated_summary";destination.mkdir()
                if "transfer" in studies:
                    shutil.copy2(out/"transfer_setting_fresh/transfer_setting_fresh_summary.csv",
                                 destination/"transfer_setting_fresh_summary.csv")
                if "kcurve" in studies:summarize.kcurve(out/"current_transfer",destination)
                for study in ["rare","poi","linker","families"]:
                    if study in studies:
                        import pandas as pd
                        getattr(summarize,study)(pd.read_csv(out/study/"query_predictions.csv"),destination)
                comparison_studies=[s for s in studies if s not in ["transfer","cdk6","efficiency"]]
                if comparison_studies:
                    run("validate_predictions.py","--artifacts",art,"--output",out,"--studies",",".join(comparison_studies))
            if "cdk6" in studies and args.mode!="infer":
                run("cdk6_metrics.py","--inputs",art/"cdk6","--output",out/"cdk6")
            run("check_paper_metrics.py","--summary",summary,"--studies",",".join(studies))
            if args.mode=="infer":
                inference_note=("Read comparison.json for the checked study predictions. " if comparison_studies else "")
                if "transfer" in studies:
                    inference_note+="The current transfer runner verifies its 45 regenerated predictions against archived split identities and S2/S3 metrics. "
            else:
                inference_note="Metrics were recomputed from frozen references; no checkpoint prediction comparison was requested. "
            record=dict(status="completed",levels=levels,elapsed_seconds=time.time()-started,
                        note=inference_note+"GPU timing was not remeasured.")
        (out/"reproduction_status.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2),flush=True)


if __name__=="__main__":main()
