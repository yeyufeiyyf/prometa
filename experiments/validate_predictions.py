"""Compare regenerated ProMeta identities, scores and paper-precision metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--studies",required=True)
    a=p.parse_args();art=a.artifacts;out=a.output;checks=[]
    studies=a.studies.split(",")
    def compare(old,new,keys,score="y_score",label="y_true",name=""):
        old=old.sort_values(keys).reset_index(drop=True);new=new.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(old[keys],new[keys],check_dtype=False)
        assert not old.duplicated(keys).any() and not new.duplicated(keys).any()
        np.testing.assert_array_equal(old[label],new[label])
        delta=float(abs(old[score].to_numpy()-new[score].to_numpy()).max())
        checks.append(dict(study=name,rows=len(old),identities_and_labels_match=True,
                           maximum_score_difference=delta,score_within_1e_5=delta<=1e-5))
    if "transfer" in studies:
        for path in sorted((art/"current_transfer/table_s4_prometa_rerun/evaluation").rglob("query_predictions.csv")):
            compare(pd.read_csv(path),pd.read_csv(out/path.relative_to(art)),["task","row_id"],name="transfer")
    if "kcurve" in studies:
        for path in sorted((art/"current_transfer/k1_to_k7_20261007/evaluation").rglob("query_predictions.csv")):
            compare(pd.read_csv(path),pd.read_csv(out/path.relative_to(art)),["k_shot","task","row_id"],name="kcurve")
        rel=Path("current_transfer/k1_to_k7_20261007/k0_query_predictions.csv")
        compare(pd.read_csv(art/rel),pd.read_csv(out/rel),["direction","seed","task","row_id"],name="kcurve_zero")
    if "poi" in studies:
        old=pd.concat([pd.read_csv(path) for path in sorted((art/"poi").glob("*/seed_*/query_predictions.csv"))])
        compare(old,pd.read_csv(out/"poi/query_predictions.csv"),["direction","seed","task","k","support_variant","row_id"],name="poi")
    if "rare" in studies:
        old=pd.concat([pd.read_csv(path) for path in sorted((art/"rare_e3/results/prometa2_multisource_loeo_v1/common_multisource").glob("seed_*/query_predictions.csv"))])
        compare(old,pd.read_csv(out/"rare/query_predictions.csv"),["seed","task","episode_id","context_id"],name="rare")
    if "linker" in studies:
        compare(pd.read_csv(art/"linker/query_predictions.csv"),pd.read_csv(out/"linker/query_predictions.csv"),
                ["direction","seed","group_id","support_positive_id","support_negative_id","query_id"],
                score="prometa_score",label="label",name="linker")
    if "families" in studies:
        compare(pd.read_csv(art/"families/query_predictions.csv"),pd.read_csv(out/"families/query_predictions.csv"),
                ["family","seed","support_positive","support_negative","query"],score="prometa_score",name="families")
    keys={"transfer":["direction","k","q"],"kcurve":["direction","k"],"poi":["direction","k"],
          "rare":["scope"],"linker":["scope"],"families":["family"]}
    for study in studies:
        path=out/f"summary/{study}_summary.csv";newpath=out/f"regenerated_summary/{study}_summary.csv"
        old=pd.read_csv(path).sort_values(keys[study]).reset_index(drop=True)
        new=pd.read_csv(newpath).sort_values(keys[study]).reset_index(drop=True)
        pd.testing.assert_frame_equal(old[keys[study]],new[keys[study]],check_dtype=False)
        fields=[f"{name}_{stat}" for name in ["auroc","auprc","balacc"] for stat in ["mean","std"]]
        diff=float(abs(old[fields].to_numpy()-new[fields].to_numpy()).max())
        assert np.array_equal(old[fields].round(3),new[fields].round(3)),f"{study}: metrics changed at paper precision"
        checks.append(dict(study=study,type="summary",maximum_metric_difference=diff,all_columns_match_at_3dec=True))
    report=dict(status="PASS_AT_PAPER_PRECISION",checks=checks,
        exact_score_equality=all(c.get("maximum_score_difference",0)==0 for c in checks),
        numerical_note="CPU/GPU reductions can change scores and near-tied AUPRC ranks. All identities/labels and rounded mean/SD columns are checked; unrounded differences are reported, not hidden.")
    (out/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report["status"],"comparison units:",len(checks))


if __name__=="__main__":main()
