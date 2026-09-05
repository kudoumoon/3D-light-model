#!/usr/bin/env python3
"""评估 chunk 内多 source latent warp 的零参数 coverage 增益。"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_geometry_head import points_from_depth
from latent_geometry_head_v2 import LatentGeometryHeadV2
from latent_reprojection_loss import forward_splat_latent, merge_latent_warps_priority
from tools.audit_tum_coverage_sources import masked_l1, pose_matrix

def merge_nearest(warps):
    latent = torch.cat([warp.latent for warp in warps], dim=0)
    valid = torch.cat([warp.projected_valid for warp in warps], dim=0)
    depth = torch.cat([warp.target_depth for warp in warps], dim=0)
    depth = torch.where(valid, depth, torch.full_like(depth, torch.inf))
    index = depth.argmin(dim=0, keepdim=True)
    selected = torch.gather(latent, 0, index.expand(1, latent.shape[1], *latent.shape[-2:]))
    union = valid.any(dim=0, keepdim=True)
    return selected, union

def mean(rows, key): return float(np.nanmean([row[key] for row in rows]))

def safe_coverage(latent, target, valid, threshold):
    error=(latent-target).abs().mean(dim=1,keepdim=True)
    return valid & (error<=threshold)

def compose_paths(pairs, max_hops):
    by_scene=defaultdict(list)
    for row in pairs:
        item=dict(row); item["_matrix"]=pose_matrix(row)[0]; by_scene[row["scene"]].append(item)
    expanded=[]
    for scene,items in by_scene.items():
        incoming=defaultdict(list)
        for item in items: incoming[item["target"]].append(item)
        for target in incoming:
            queue=[(item["source"],item["_matrix"],1) for item in incoming[target]]; found={}
            while queue:
                source,transform,hops=queue.pop(0)
                if source not in found or hops<found[source][1]: found[source]=(transform,hops)
                if hops>=max_hops: continue
                for previous in incoming.get(source,[]): queue.append((previous["source"],transform@previous["_matrix"],hops+1))
            for source,(transform,hops) in found.items(): expanded.append({"scene":scene,"source":source,"target":target,"_matrix":transform,"hops":hops})
    return expanded

def merge_max_support(warps):
    latent=torch.cat([warp.latent for warp in warps],dim=0); valid=torch.cat([warp.projected_valid for warp in warps],dim=0)
    support=torch.cat([warp.support_mass for warp in warps],dim=0); score=torch.where(valid,support,torch.full_like(support,-1))
    index=score.argmax(dim=0,keepdim=True); selected=torch.gather(latent,0,index.expand(1,latent.shape[1],*latent.shape[-2:])); return selected,valid.any(dim=0,keepdim=True)

def merge_priority_fill(warps, source_rows):
    def frame_number(value): return int(value.rsplit("_",1)[1])
    target=frame_number(source_rows[0]["target"]); order=sorted(range(len(warps)),key=lambda i:abs(target-frame_number(source_rows[i]["source"])))
    merged=merge_latent_warps_priority(warps,order)
    return merged.latent,merged.projected_valid

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True); parser.add_argument("--pairs",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--test-scenes",nargs="+",default=("tum_xyz_test","tum_rpy_test")); parser.add_argument("--valid-threshold",type=float,default=.5)
    parser.add_argument("--compose-max-hops",type=int,default=1)
    args=parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("expose exactly one checked idle GPU")
    if args.output.exists(): raise FileExistsError(args.output)
    cache=torch.load(args.cache,map_location="cpu",weights_only=False); index={v:i for i,v in enumerate(cache["sample_ids"])}
    pairs=[r for r in json.loads(args.pairs.read_text())["pairs"] if r.get("ok") and r["scene"] in set(args.test_scenes)]
    pairs=compose_paths(pairs,args.compose_max_hops)
    groups=defaultdict(list)
    for row in pairs: groups[(row["scene"],row["target"])].append(row)
    groups={k:v for k,v in groups.items() if len(v)>=2}
    if not groups: raise RuntimeError("no target has multiple source frames")
    device=torch.device("cuda:0"); model=LatentGeometryHeadV2().to(device).eval(); model.load_state_dict(torch.load(args.checkpoint,map_location="cpu",weights_only=False)["model"])
    rows=[]; started=time.perf_counter()
    for (scene,target_id),source_rows in groups.items():
        ti=index[target_id]; target=cache["latent"][ti:ti+1].to(device); student=[]; teacher=[]; singles=[]
        for pair in source_rows:
            si=index[pair["source"]]; source=cache["latent"][si:si+1].to(device); depth=cache["depth"][si:si+1].to(device); gt_valid=cache["valid"][si:si+1].to(device); K=cache["intrinsics"][si:si+1].to(device); T=pair["_matrix"][None].to(device)
            output=model(source,K); pred_valid=(output.latent_valid>=args.valid_threshold).float()
            sw=forward_splat_latent(source,output.latent_points,pred_valid,K,T); tw=forward_splat_latent(source,points_from_depth(depth,K),gt_valid,K,T)
            student.append(sw); teacher.append(tw); singles.append({"coverage":float(sw.coverage.mean()),"l1":masked_l1(sw.latent,target,sw.projected_valid.float())})
        merge_start=torch.cuda.Event(enable_timing=True); merge_end=torch.cuda.Event(enable_timing=True)
        merge_start.record(); merged,union=merge_nearest(student); merge_end.record(); torch.cuda.synchronize()
        teacher_merged,teacher_union=merge_nearest(teacher); support_merged,support_union=merge_max_support(student); priority_merged,priority_union=merge_priority_fill(student,source_rows)
        best_single=max(singles,key=lambda row:row["coverage"])
        row={"scene":scene,"target":target_id,"sources":len(source_rows),"single_mean_coverage":mean(singles,"coverage"),"single_best_coverage":best_single["coverage"],"single_mean_l1":mean(singles,"l1"),"union_coverage":float(union.float().mean()),"union_l1":masked_l1(merged,target,union.float()),"coverage_gain_over_best":float(union.float().mean())-best_single["coverage"],"teacher_union_coverage":float(teacher_union.float().mean()),"teacher_union_l1":masked_l1(teacher_merged,target,teacher_union.float()),"merge_runtime_ms":float(merge_start.elapsed_time(merge_end))}
        for threshold in (.1,.2,.3,.4):
            single_safe=[safe_coverage(w.latent,target,w.projected_valid,threshold) for w in student]
            union_safe=safe_coverage(merged,target,union,threshold)
            oracle_safe=torch.stack(single_safe).any(dim=0)
            best_safe=max(float(mask.float().mean()) for mask in single_safe)
            row[f"safe_coverage_{threshold:.1f}_best_single"]=best_safe
            row[f"safe_coverage_{threshold:.1f}_union"]=float(union_safe.float().mean())
            row[f"safe_coverage_{threshold:.1f}_oracle_selector"]=float(oracle_safe.float().mean())
            row[f"safe_gain_{threshold:.1f}"]=float(union_safe.float().mean())-best_safe
            for name,candidate,candidate_valid in (("support",support_merged,support_union),("priority",priority_merged,priority_union)):
                candidate_safe=safe_coverage(candidate,target,candidate_valid,threshold)
                row[f"{name}_safe_coverage_{threshold:.1f}"]=float(candidate_safe.float().mean())
                row[f"{name}_safe_gain_{threshold:.1f}"]=float(candidate_safe.float().mean())-best_safe
        row["support_union_l1"]=masked_l1(support_merged,target,support_union.float()); row["priority_union_l1"]=masked_l1(priority_merged,target,priority_union.float())
        rows.append(row)
    numeric=[key for key,value in rows[0].items() if isinstance(value,(int,float)) and key not in {"sources"}]
    aggregate={"sources":mean(rows,"sources"),**{key:mean(rows,key) for key in numeric}}
    for _ in range(30): merge_nearest(student)
    bench_start=torch.cuda.Event(enable_timing=True); bench_end=torch.cuda.Event(enable_timing=True); bench_start.record()
    for _ in range(200): merge_nearest(student)
    bench_end.record(); torch.cuda.synchronize(); warm_merge_ms=bench_start.elapsed_time(bench_end)/200
    for _ in range(30): merge_priority_fill(student,source_rows)
    priority_start=torch.cuda.Event(enable_timing=True); priority_end=torch.cuda.Event(enable_timing=True); priority_start.record()
    for _ in range(200): merge_priority_fill(student,source_rows)
    priority_end.record(); torch.cuda.synchronize(); warm_priority_fill_ms=priority_start.elapsed_time(priority_end)/200
    args.output.mkdir(parents=True); report={"schema_version":1,"stage":"zero-parameter multi-source chunk coverage","repository_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),"config":{"test_scenes":args.test_scenes,"valid_threshold":args.valid_threshold,"compose_max_hops":args.compose_max_hops,"merge":"nearest/support/temporal-priority-fill ablation"},"target_groups":len(rows),"aggregate":aggregate,"rows":rows,"runtime":{"seconds":time.perf_counter()-started,"gpu":torch.cuda.get_device_name(0),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"warm_nearest_merge_ms":warm_merge_ms,"warm_priority_fill_ms":warm_priority_fill_ms},"new_trainable_parameters":0,"public_shape_changed":False}
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)); print(json.dumps(aggregate,indent=2)); print("实验已完成")
if __name__=="__main__": main()
