#!/usr/bin/env python3
"""评估联合 confidence 在 multi-source priority-fill 输出上的概率质量。"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from latent_geometry_head_v2 import LatentGeometryHeadV2
from latent_motion_confidence import LatentMotionConfidence
from latent_reprojection_loss import forward_splat_latent, merge_latent_warps_priority
from tools.evaluate_multisource_coverage import compose_paths
from tools.train_reuse_probability_confidence import probability_metrics, risk_coverage

def frame(value): return int(value.rsplit("_",1)[1])
def motion_from_matrix(matrix):
    r,_=cv2.Rodrigues(matrix[:3,:3].cpu().numpy()); return torch.from_numpy(np.concatenate((r.reshape(-1),matrix[:3,3].cpu().numpy())).astype(np.float32))

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cache",type=Path,required=True); parser.add_argument("--pairs",type=Path,required=True); parser.add_argument("--geometry",type=Path,required=True); parser.add_argument("--confidence",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--scenes",nargs="+",required=True); parser.add_argument("--compose-max-hops",type=int,default=3); parser.add_argument("--reuse-error-threshold",type=float,default=.2); args=parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("expose exactly one checked idle GPU")
    if args.output.exists(): raise FileExistsError(args.output)
    device=torch.device("cuda:0"); started=time.perf_counter(); cache=torch.load(args.cache,map_location="cpu",weights_only=False); index={v:i for i,v in enumerate(cache["sample_ids"])}
    direct=[r for r in json.loads(args.pairs.read_text())["pairs"] if r.get("ok") and r["scene"] in set(args.scenes)]; rows=compose_paths(direct,args.compose_max_hops); groups=defaultdict(list)
    for row in rows: groups[(row["scene"],row["target"])].append(row)
    groups={key:value for key,value in groups.items() if len(value)>=2}
    geometry=LatentGeometryHeadV2().to(device).eval(); geometry.load_state_dict(torch.load(args.geometry,map_location="cpu",weights_only=False)["model"])
    checkpoint=torch.load(args.confidence,map_location="cpu",weights_only=False); confidence=LatentMotionConfidence().to(device).eval(); confidence.load_state_dict(checkpoint["model"]); mean=checkpoint["motion_mean"].to(device); std=checkpoint["motion_std"].to(device)
    logits=[]; labels=[]; errors=[]; total=0; group_rows=[]
    for (scene,target_id),source_rows in groups.items():
        target=cache["latent"][index[target_id]:index[target_id]+1].to(device); feature_warps=[]; confidence_warps=[]
        order=sorted(range(len(source_rows)),key=lambda i:abs(frame(target_id)-frame(source_rows[i]["source"])))
        for row in source_rows:
            si=index[row["source"]]; source=cache["latent"][si:si+1].to(device); K=cache["intrinsics"][si:si+1].to(device); T=row["_matrix"][None].to(device); output=geometry(source,K); valid=(output.latent_valid>=.5).float(); motion=((motion_from_matrix(row["_matrix"]).to(device)-mean)/std)[None]
            feature_warps.append(forward_splat_latent(source,output.latent_points,valid,K,T)); source_logit=confidence(source,output.latent_depth,output.latent_valid_logits,motion); confidence_warps.append(forward_splat_latent(source_logit,output.latent_points,valid,K,T))
        merged=merge_latent_warps_priority(feature_warps,order); merged_confidence=merge_latent_warps_priority(confidence_warps,order); error=(merged.latent-target).abs().mean(1,keepdim=True); mask=merged.projected_valid; label=(error<=args.reuse_error_threshold)
        logits.append(merged_confidence.latent[mask].cpu()); labels.append(label[mask].float().cpu()); errors.append(error[mask].cpu()); total+=mask.numel(); group_rows.append({"scene":scene,"target":target_id,"sources":len(source_rows),"coverage":float(mask.float().mean()),"safe_coverage":float((mask&label).float().mean()),"l1":float(error[mask].mean())})
    z=torch.cat(logits).numpy(); y=torch.cat(labels).numpy(); e=torch.cat(errors).numpy(); probability=1/(1+np.exp(-z))
    report={"schema_version":1,"stage":"multi-source priority-fill reuse confidence","repository_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),"config":{**{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},"probability":"raw BCE probability; no target-domain calibration"},"target_groups":len(group_rows),"metrics":probability_metrics(probability,y),"risk_coverage":risk_coverage(probability,y,e,total),"aggregate":{"coverage":float(np.mean([r["coverage"] for r in group_rows])),"safe_coverage":float(np.mean([r["safe_coverage"] for r in group_rows])),"l1":float(np.mean([r["l1"] for r in group_rows]))},"rows":group_rows,"runtime":{"seconds":time.perf_counter()-started,"gpu":torch.cuda.get_device_name(0),"peak_allocated_bytes":torch.cuda.max_memory_allocated()},"new_trainable_parameters":0,"public_shape_changed":False}
    args.output.mkdir(parents=True); (args.output/"metrics.json").write_text(json.dumps(report,indent=2)); print(json.dumps({"target_groups":report["target_groups"],"metrics":report["metrics"],"risk_coverage":report["risk_coverage"],"aggregate":report["aggregate"]},indent=2)); print("实验已完成")
if __name__=="__main__": main()
