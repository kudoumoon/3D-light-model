#!/usr/bin/env python3
"""诊断联合训练后 TUM global top-k 精度变化来自权重还是 motion normalization。"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from latent_geometry_head_v2 import LatentGeometryHeadV2
from latent_motion_confidence import LatentMotionConfidence
from tools.train_reuse_probability_confidence import Pairs, collect, probability_metrics

def evaluate(name,weight_checkpoint,normalization_checkpoint,geometry,cache,rows,device,threshold):
    weights=torch.load(weight_checkpoint,map_location="cpu",weights_only=False); norm=torch.load(normalization_checkpoint,map_location="cpu",weights_only=False); model=LatentMotionConfidence().to(device).eval(); model.load_state_dict(weights["model"]); parts=[]
    for scene in ("tum_xyz_test","tum_rpy_test"):
        subset=[row for row in rows if row["scene"]==scene]; loader=DataLoader(Pairs(cache,subset,norm["motion_mean"],norm["motion_std"]),batch_size=8)
        logits,labels,errors,total=collect(geometry,model,loader,device,threshold); probability=torch.sigmoid(logits).numpy(); parts.append({"scene":scene,"probability":probability,"label":labels.numpy(),"error":errors.numpy(),"total":total})
    probability=np.concatenate([part["probability"] for part in parts]); label=np.concatenate([part["label"] for part in parts]); scene_id=np.concatenate([np.full(len(part["label"]),i) for i,part in enumerate(parts)]); count=max(1,int(.1*len(label))); selected=np.argsort(-probability)[:count]
    return {"name":name,"global":{**probability_metrics(probability,label),"top10_precision":float(label[selected].mean()),"top10_scene_fraction":{parts[i]["scene"]:float(np.mean(scene_id[selected]==i)) for i in range(len(parts))}},"by_scene":{part["scene"]:{**probability_metrics(part["probability"],part["label"]),"top10_precision":float(part["label"][np.argsort(-part["probability"])[:max(1,int(.1*len(part["label"])))]].mean()),"positive_score_mean":float(part["probability"][part["label"]>0].mean()),"negative_score_mean":float(part["probability"][part["label"]==0].mean())} for part in parts}}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cache",type=Path,required=True); parser.add_argument("--pairs",type=Path,required=True); parser.add_argument("--geometry",type=Path,required=True); parser.add_argument("--tum-confidence",type=Path,required=True); parser.add_argument("--joint-confidence",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--reuse-error-threshold",type=float,default=.2); args=parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("expose exactly one checked idle GPU")
    if args.output.exists(): raise FileExistsError(args.output)
    device=torch.device("cuda:0"); cache=torch.load(args.cache,map_location="cpu",weights_only=False); rows=[r for r in json.loads(args.pairs.read_text())["pairs"] if r.get("ok") and r["scene"].endswith("_test")]; geometry=LatentGeometryHeadV2().to(device).eval(); geometry.load_state_dict(torch.load(args.geometry,map_location="cpu",weights_only=False)["model"])
    cases=[("tum_weights_tum_norm",args.tum_confidence,args.tum_confidence),("joint_weights_joint_norm",args.joint_confidence,args.joint_confidence),("joint_weights_tum_norm",args.joint_confidence,args.tum_confidence),("tum_weights_joint_norm",args.tum_confidence,args.joint_confidence)]
    results=[evaluate(name,w,n,geometry,cache,rows,device,args.reuse_error_threshold) for name,w,n in cases]
    report={"schema_version":1,"stage":"TUM top-k confidence causal normalization audit","repository_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),"results":results,"interpretation_rule":"If swapping only normalization restores top10 precision, motion-stat drift is causal; otherwise joint weight optimization causes the trade-off."}
    args.output.mkdir(parents=True); (args.output/"metrics.json").write_text(json.dumps(report,indent=2)); print(json.dumps([{"name":r["name"],"top10_precision":r["global"]["top10_precision"],"scene_fraction":r["global"]["top10_scene_fraction"],"auc":r["global"]["auc"]} for r in results],indent=2)); print("实验已完成")
if __name__=="__main__": main()
