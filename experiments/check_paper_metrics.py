"""Check regenerated/recomputed summaries against the stated ProMeta paper values."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--studies",required=True)
    a=p.parse_args();checks=[];studies=a.studies.split(",")
    root=Path(__file__).resolve().parents[1]
    if "transfer" in studies:
        ref=pd.read_csv(root/"artifacts/transfer_setting_fresh_20261015/full_query_summary.csv")
        got=pd.read_csv(a.summary/"transfer_setting_fresh_summary.csv")
        merged=ref.merge(got,on=["direction","k","q"],suffixes=("_paper","_recomputed"),validate="one_to_one")
        assert len(merged)==15
        for name in ("pooled_auroc_mean","macro_auroc_mean"):
            assert np.allclose(merged[f"{name}_paper"],merged[f"{name}_recomputed"],atol=1e-12,rtol=0)
        for name in ("pooled_auroc","macro_auroc"):
            assert np.allclose(merged[f"{name}_std"],merged[f"{name}_sd"],atol=1e-12,rtol=0)
        checks.append(dict(study="transfer",paper_values=60,passed=True))
    if "kcurve" in studies:
        got=pd.read_csv(a.summary/"kcurve_summary.csv")
        for direction,expected in [("CRBN->VHL",[.627,.837,.901,.929,.920,.940]),("VHL->CRBN",[.675,.602,.703,.700,.706,.707])]:
            assert got[got.direction.eq(direction)].sort_values("k").auroc_mean.round(3).tolist()==expected
        checks.append(dict(study="kcurve",paper_auroc_values=12,passed=True))
    for study,expected in [("rare",[.684,.787,.601]),("linker",[.708,.753,.642])]:
        if study in studies:
            row=pd.read_csv(a.summary/f"{study}_summary.csv").iloc[0]
            assert [round(row[x+"_mean"],3) for x in ["auroc","auprc","balacc"]]==expected
            checks.append(dict(study=study,passed=True))
    if "poi" in studies:
        got=pd.read_csv(a.summary/"poi_summary.csv")
        for direction,expected in [("CRBN->VHL",[.747,.797,.797]),("VHL->CRBN",[.690,.714,.715])]:
            assert got[got.direction.eq(direction)].sort_values("k").auroc_mean.round(3).tolist()==expected
        checks.append(dict(study="poi",passed=True))
    if "families" in studies:
        got=pd.read_csv(a.summary/"families_summary.csv").set_index("family")
        for family,value in [("HPK1",.698),("FLT3",.749),("FLT3_10nM",.662)]:assert round(got.loc[family,"auroc_mean"],3)==value
        checks.append(dict(study="families",passed=True))
    if "cdk6" in studies:
        got=json.loads((a.summary.parent/"cdk6/summary.json").read_text())
        assert round(got["mean_query_auroc"],3)==.886
        assert round(got["sd_across_seed_means"],3)==.008
        checks.append(dict(study="cdk6",passed=True))
    (a.summary/"paper_checks.json").write_text(json.dumps(dict(status="PASS",checks=checks),indent=2)+"\n")
    print("Paper reference checks PASS",len(checks))


if __name__=="__main__":main()
