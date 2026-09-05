#!/usr/bin/env python3
"""冻结 Geometry Head，训练现有轻量 confidence 预测真实 latent reuse 成功概率。"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from latent_geometry_head_v2 import LatentGeometryHeadV2
from latent_motion_confidence import LatentMotionConfidence
from latent_reprojection_loss import forward_splat_latent

def pose_and_motion(row):
    r=np.asarray(row["rvec"],np.float32); t=np.asarray(row["tvec"],np.float32); R,_=cv2.Rodrigues(r)
    T=torch.eye(4); T[:3,:3]=torch.from_numpy(R); T[:3,3]=torch.from_numpy(t)
    return T,torch.from_numpy(np.concatenate((r,t)))

class Pairs(Dataset):
    def __init__(self,cache,rows,mean,std): self.c=cache; self.r=rows; self.i={v:k for k,v in enumerate(cache["sample_ids"])}; self.mean=mean; self.std=std
    def __len__(self): return len(self.r)
    def __getitem__(self,j):
        row=self.r[j]; si=self.i[row["source"]]; ti=self.i[row["target"]]; T,m=pose_and_motion(row)
        return self.c["latent"][si],self.c["latent"][ti],self.c["intrinsics"][si],T,(m-self.mean)/self.std

def auc(score,label):
    order=np.argsort(score,kind="mergesort"); rank=np.empty_like(order,dtype=np.float64); rank[order]=np.arange(1,len(score)+1); positive=label.astype(bool); p=positive.sum(); n=len(label)-p
    return float((rank[positive].sum()-p*(p+1)/2)/(p*n)) if p and n else float("nan")

def pr_auc(score,label):
    order=np.argsort(-score); y=label[order]; tp=np.cumsum(y); precision=tp/np.arange(1,len(y)+1); recall=tp/max(1,y.sum())
    return float(np.sum(precision*(recall-np.concatenate(([0.],recall[:-1])))))

def probability_metrics(probability,label):
    p=np.clip(probability,1e-6,1-1e-6); y=label.astype(np.float64); ece=0.; bins=[]
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        mask=(p>=lo)&((p<=hi) if hi==1 else (p<hi))
        if mask.any():
            confidence=float(p[mask].mean()); accuracy=float(y[mask].mean()); ece+=float(mask.mean())*abs(confidence-accuracy); bins.append({"lo":float(lo),"hi":float(hi),"count":int(mask.sum()),"confidence":confidence,"accuracy":accuracy})
    return {"auc":auc(p,y),"pr_auc":pr_auc(p,y),"ece":ece,"brier":float(np.mean((p-y)**2)),"nll":float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))),"positive_rate":float(y.mean()),"mean_probability":float(p.mean()),"bins":bins}

@torch.inference_mode()
def collect(geometry,confidence,loader,device,error_threshold):
    confidence.eval(); logits=[]; labels=[]; errors=[]; total=0
    for source,target,K,T,motion in loader:
        source,target,K,T,motion=[x.to(device) for x in (source,target,K,T,motion)]; output=geometry(source,K); valid=(output.latent_valid>=.5).float()
        warp=forward_splat_latent(source,output.latent_points,valid,K,T); source_logits=confidence(source,output.latent_depth,output.latent_valid_logits,motion)
        target_logits=forward_splat_latent(source_logits,output.latent_points,valid,K,T).latent
        error=(warp.latent-target).abs().mean(1,keepdim=True); mask=warp.projected_valid
        logits.append(target_logits[mask].cpu()); labels.append((error[mask]<=error_threshold).float().cpu()); errors.append(error[mask].cpu()); total+=mask.numel()
    return torch.cat(logits),torch.cat(labels),torch.cat(errors),total

def fit_temperature(logits,labels):
    temperature=torch.tensor(1.,dtype=torch.float64,requires_grad=True); x=logits.double(); y=labels.double(); optimizer=torch.optim.LBFGS([temperature],lr=.1,max_iter=80,line_search_fn="strong_wolfe")
    def closure(): optimizer.zero_grad(); loss=F.binary_cross_entropy_with_logits(x/temperature.clamp_min(.05),y); loss.backward(); return loss
    optimizer.step(closure); return float(temperature.detach().clamp_min(.05))

def risk_coverage(probability,label,error,total_cells):
    order=np.argsort(-probability); result=[]
    for fraction in (.1,.25,.5,.75,1.):
        count=max(1,int(len(order)*fraction)); selected=order[:count]
        result.append({"projected_kept_fraction":fraction,"kept_cells":count,"precision":float(label[selected].mean()),"selected_l1":float(error[selected].mean()),"safe_coverage_full_grid":float(label[selected].sum()/total_cells),"selected_coverage_full_grid":float(count/total_cells)})
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cache",type=Path,required=True); parser.add_argument("--pairs",type=Path,required=True); parser.add_argument("--geometry",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--epochs",type=int,default=20); parser.add_argument("--batch-size",type=int,default=8); parser.add_argument("--learning-rate",type=float,default=3e-4); parser.add_argument("--reuse-error-threshold",type=float,default=.2); parser.add_argument("--seed",type=int,default=23); args=parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("expose exactly one checked idle GPU")
    if args.output.exists(): raise FileExistsError(args.output)
    torch.manual_seed(args.seed); np.random.seed(args.seed); device=torch.device("cuda:0"); started=time.perf_counter()
    cache=torch.load(args.cache,map_location="cpu",weights_only=False); rows=[r for r in json.loads(args.pairs.read_text())["pairs"] if r.get("ok")]
    train=[r for r in rows if r["scene"].endswith("_train")]; val=[r for r in rows if r["scene"].endswith("_val")]; test=[r for r in rows if r["scene"].endswith("_test")]
    if not train or not val or not test: raise RuntimeError("scene-disjoint split is empty")
    motions=torch.stack([pose_and_motion(row)[1] for row in train]); mean=motions.mean(0); std=motions.std(0).clamp_min(1e-4)
    loaders={name:DataLoader(Pairs(cache,data,mean,std),batch_size=args.batch_size,shuffle=name=="train") for name,data in (("train",train),("val",val),("test",test))}
    geometry=LatentGeometryHeadV2().to(device).eval(); geometry.load_state_dict(torch.load(args.geometry,map_location="cpu",weights_only=False)["model"])
    for parameter in geometry.parameters(): parameter.requires_grad_(False)
    confidence=LatentMotionConfidence().to(device); optimizer=torch.optim.AdamW(confidence.parameters(),lr=args.learning_rate,weight_decay=1e-4); history=[]; best=None; best_nll=float("inf")
    for epoch in range(1,args.epochs+1):
        confidence.train(); losses=[]
        for source,target,K,T,motion in loaders["train"]:
            source,target,K,T,motion=[x.to(device) for x in (source,target,K,T,motion)]
            with torch.no_grad(): output=geometry(source,K); valid=(output.latent_valid>=.5).float(); warp=forward_splat_latent(source,output.latent_points,valid,K,T); error=(warp.latent-target).abs().mean(1,keepdim=True); label=(error<=args.reuse_error_threshold).float(); mask=warp.projected_valid
            source_logits=confidence(source,output.latent_depth,output.latent_valid_logits,motion); target_logits=forward_splat_latent(source_logits,output.latent_points,valid,K,T).latent
            loss=F.binary_cross_entropy_with_logits(target_logits[mask],label[mask]); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        val_logits,val_labels,_,_=collect(geometry,confidence,loaders["val"],device,args.reuse_error_threshold); val_nll=float(F.binary_cross_entropy_with_logits(val_logits,val_labels)); history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val_nll":val_nll})
        if val_nll<best_nll: best_nll=val_nll; best={k:v.detach().cpu().clone() for k,v in confidence.state_dict().items()}
        print(json.dumps(history[-1]),flush=True)
    confidence.load_state_dict(best); val_logits,val_labels,_,_=collect(geometry,confidence,loaders["val"],device,args.reuse_error_threshold); temperature=fit_temperature(val_logits,val_labels)
    test_logits,test_labels,test_errors,total=collect(geometry,confidence,loaders["test"],device,args.reuse_error_threshold); z=test_logits.numpy(); y=test_labels.numpy(); error=test_errors.numpy(); raw=1/(1+np.exp(-z)); calibrated=1/(1+np.exp(-z/temperature))
    report={"schema_version":1,"stage":"frozen-geometry latent reuse probability","repository_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),"config":{**{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},"label":"projected cell mean-channel latent L1 <= threshold","motion_normalization_mean":mean.tolist(),"motion_normalization_std":std.tolist()},"splits":{"train_pairs":len(train),"val_pairs":len(val),"test_pairs":len(test)},"parameters":{"geometry_frozen":sum(p.numel() for p in geometry.parameters()),"confidence_trainable":sum(p.numel() for p in confidence.parameters())},"history":history,"selection":{"metric":"validation NLL","best_nll":best_nll},"temperature":temperature,"test":{"raw":probability_metrics(raw,y),"calibrated":probability_metrics(calibrated,y),"risk_coverage":risk_coverage(calibrated,y,error,total)},"runtime":{"seconds":time.perf_counter()-started,"gpu":torch.cuda.get_device_name(0),"peak_allocated_bytes":torch.cuda.max_memory_allocated()},"output_contract":{"latent_confidence":"[B,1,44,80] source-grid logits; splatted to target grid with geometry","shape_changed":False}}
    args.output.mkdir(parents=True); torch.save({"model":confidence.state_dict(),"temperature":temperature,"motion_mean":mean,"motion_std":std,"config":report["config"]},args.output/"checkpoint.pt"); (args.output/"metrics.json").write_text(json.dumps(report,indent=2)); print(json.dumps({"temperature":temperature,"test":report["test"]},indent=2)); print("实验已完成")
if __name__=="__main__": main()
