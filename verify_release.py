#!/usr/bin/env python3
"""Verify code syntax and all selected ProMeta artifact hashes without network access."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts",type=Path,default=ROOT/"artifacts")
    p.add_argument("--code-only",action="store_true")
    a=p.parse_args()
    code=[*ROOT.glob("*.py"),*(ROOT/"prometa").rglob("*.py"),*(ROOT/"experiments").glob("*.py")]
    for path in code:ast.parse(path.read_text(),filename=str(path))
    if a.code_only:print(f"PASS: {len(code)} Python sources parse.");return
    art=a.artifacts.resolve();manifest=json.loads((art/"MANIFEST.json").read_text())
    paths=[]
    for rec in manifest["files"]:
        rel=Path(rec["path"])
        assert not rel.is_absolute() and ".." not in rel.parts
        path=art/rel
        assert path.is_file() and path.stat().st_size==rec["bytes"] and sha(path)==rec["sha256"],str(path)
        paths.append(rel)
    assert len(paths)==len(set(paths))
    assert sum("context_weights/checkpoints" in str(path) and path.suffix==".pt" for path in paths)==20
    assert sum("table_s4_prometa_rerun/checkpoints" in str(path) and path.suffix==".pt" for path in paths)==57
    fresh=art/"transfer_setting_fresh_20261015"
    jobs=json.loads((fresh/"jobs.json").read_text())
    assert len(jobs)==45
    assert sum(str(path).startswith("transfer_setting_fresh_20261015/checkpoints/") and path.name=="model.pt" for path in paths)==45
    assert sum(str(path).startswith("transfer_setting_fresh_20261015/runs/") and path.name=="query_predictions.csv" for path in paths)==45
    assert len({sha(fresh/"checkpoints"/job["id"]/"model.pt") for job in jobs})==45
    assert sum("rare_e3/results" in str(path) and path.suffix==".pt" for path in paths)==10
    transfer=art/"current_transfer"/"table_s4_prometa_rerun"
    mapping=json.loads((transfer/"checkpoint_mapping.json").read_text())
    assert len(mapping["checkpoints"])==45
    assert len({(r["direction"],r["seed"],r["evaluation_k"],r["evaluation_q"])
                for r in mapping["checkpoints"]})==45
    for record in mapping["checkpoints"]:
        named=transfer/record["checkpoint"]
        source=transfer/record["source_checkpoint"]
        assert named.is_file() and source.is_file()
        assert sha(named)==sha(source)==record["sha256"]
    study_mapping=json.loads((art/"study_checkpoints/checkpoint_mapping.json").read_text())
    assert len(study_mapping["checkpoints"])==57
    assert len({(r["study"],r["condition"],r["seed"])
                for r in study_mapping["checkpoints"]})==57
    for record in study_mapping["checkpoints"]:
        named=art/record["checkpoint"]
        source=art/record["source_checkpoint"]
        assert named.is_file() and source.is_file()
        assert sha(named)==sha(source)==record["sha256"]
    assert sum(str(path).startswith("study_checkpoints/") and path.suffix==".pt" for path in paths)==57
    assert sum(str(path).startswith("cdk6/checkpoints/") and path.name=="model.pt" for path in paths)==3
    assert (art/"cdk6/splits/cdk6_crbn.json").is_file()
    ablation=art/"ablation_matched_source_20261015"
    ablation_paths=[path for path in paths if str(path).startswith("ablation_matched_source_20261015/")]
    assert sum(path.name in ("model.pt","episodic_gnn.pt") for path in ablation_paths)==27
    assert sum(path.name=="query_predictions.csv" for path in ablation_paths)==30
    audit=json.loads((ablation/"summary.json").read_text())
    assert audit["query_alignment"]==audit["source_partition_and_episode_audit"]=="PASS"
    assert len(audit["results"])==11
    print(f"PASS: {len(code)} code files; {len(paths)} artifact hashes; 45 transfer, 27 ablation and 3 CDK6 weights.")


if __name__=="__main__":main()
