#!/usr/bin/env python3
"""固定 TUM 训练的 confidence，在新域评估 reuse 概率与风险覆盖。"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from latent_geometry_head_v2 import LatentGeometryHeadV2
from latent_motion_confidence import LatentMotionConfidence
from tools.train_reuse_probability_confidence import Pairs, collect, probability_metrics, risk_coverage

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cache",type=Path,required=True); parser.add_argument("--pairs",type=Path,required=True); parser.add_argument("--geometry",type=Path,required=True); parser.add_argument("--confidence",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--scenes",nargs="+",required=True); parser.add_argument("--batch-size",type=int,default=8); parser.add_argument("--reuse-error-threshold",type=float,default=.2); args=parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("expose exactly one checked idle GPU")
    if args.output.exists(): raise FileExistsError(args.output)
    device=torch.device("cuda:0"); started=time.perf_counter(); cache=torch.load(args.cache,map_location="cpu",weights_only=False); rows=[r for r in json.loads(args.pairs.read_text())["pairs"] if r.get("ok") and r["scene"] in set(args.scenes)]
    checkpoint=torch.load(args.confidence,map_location="cpu",weights_only=False); mean=checkpoint["motion_mean"]; std=checkpoint["motion_std"]; temperature=float(checkpoint["temperature"])
    loader=DataLoader(Pairs(cache,rows,mean,std),batch_size=args.batch_size,shuffle=False)
    geometry=LatentGeometryHeadV2().to(device).eval(); geometry.load_state_dict(torch.load(args.geometry,map_location="cpu",weights_only=False)["model"])
    confidence=LatentMotionConfidence().to(device).eval(); confidence.load_state_dict(checkpoint["model"])
    logits,labels,errors,total=collect(geometry,confidence,loader,device,args.reuse_error_threshold); z=logits.numpy(); y=labels.numpy(); error=errors.numpy(); raw=1/(1+np.exp(-z)); calibrated=1/(1+np.exp(-z/temperature))
    report={"schema_version":1,"stage":"fixed TUM reuse-probability zero-shot transfer","repository_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),"config":{**{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},"fixed_temperature":temperature,"motion_normalization":"fixed TUM train statistics"},"pairs":len(rows),"raw":probability_metrics(raw,y),"fixed_temperature":probability_metrics(calibrated,y),"risk_coverage_raw":risk_coverage(raw,y,error,total),"runtime":{"seconds":time.perf_counter()-started,"gpu":torch.cuda.get_device_name(0),"peak_allocated_bytes":torch.cuda.max_memory_allocated()},"evidence_boundary":"No target-domain fitting, calibration, threshold selection, or weight update."}
    args.output.mkdir(parents=True); (args.output/"metrics.json").write_text(json.dumps(report,indent=2)); print(json.dumps({"raw":report["raw"],"fixed_temperature":report["fixed_temperature"],"risk_coverage_raw":report["risk_coverage_raw"]},indent=2)); print("实验已完成")
if __name__=="__main__": main()
