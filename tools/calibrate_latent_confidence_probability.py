#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch, cv2
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from latent_geometry_head import LatentGeometryHead
from latent_motion_confidence import LatentMotionConfidence
from latent_reprojection_loss import forward_splat_latent
from tools.train_latent_reprojection_head import PairDataset, pose_matrix

def auc(s,y):
 o=np.argsort(s,kind="mergesort"); r=np.empty_like(o,dtype=float); r[o]=np.arange(1,len(s)+1); yb=y.astype(bool); p=yb.sum(); n=len(yb)-p
 return float((r[yb].sum()-p*(p+1)/2)/(p*n)) if p and n else float("nan")
def metrics(p,y):
 p=np.clip(p,1e-6,1-1e-6); y=y.astype(float); bins=[]; e=0.
 for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
  m=(p>=lo)&((p<=hi) if hi==1 else (p<hi));
  if m.any():
   c=float(p[m].mean()); a=float(y[m].mean()); e+=m.mean()*abs(c-a); bins.append({"lo":float(lo),"hi":float(hi),"count":int(m.sum()),"confidence":c,"accuracy":a})
 return {"auc":auc(p,y),"ece":float(e),"brier":float(np.mean((p-y)**2)),"positive_rate":float(y.mean()),"mean_probability":float(p.mean()),"bins":bins}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--cache",type=Path,required=True); ap.add_argument("--pairs",type=Path,required=True); ap.add_argument("--geometry",type=Path,required=True); ap.add_argument("--confidence",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("expose exactly one approved GPU")
 dev=torch.device("cuda:0"); cache=torch.load(a.cache,map_location="cpu",weights_only=False); rows=[r for r in json.loads(a.pairs.read_text())["pairs"] if r.get("ok")]
 ds=PairDataset(cache,rows); g=LatentGeometryHead().to(dev,dtype=torch.bfloat16).eval(); g.load_state_dict(torch.load(a.geometry,map_location="cpu",weights_only=False)["model"])
 q=LatentMotionConfidence().to(dev,dtype=torch.bfloat16).eval(); q.load_state_dict(torch.load(a.confidence,map_location="cpu",weights_only=False)["model"])
 logits=[]; labels=[]; split=[]
 with torch.inference_mode():
  for i in range(len(ds)):
   source,target,depth,valid,K,T,motion=ds[i]; source,target,valid,K,T=[x.unsqueeze(0).to(dev) for x in (source,target,valid,K,T)]
   out=g(source.to(torch.bfloat16),K); rvec=np.asarray(rows[i]["rvec"],dtype=np.float32); tvec=np.asarray(rows[i]["tvec"],dtype=np.float32); mv=torch.from_numpy(np.concatenate([rvec.reshape(-1),tvec]).astype(np.float32)).view(1,6).to(dev,dtype=torch.bfloat16); raw=q(source.to(torch.bfloat16),out.latent_depth.to(torch.bfloat16),out.latent_valid_logits.to(torch.bfloat16),mv).float()
   warp=forward_splat_latent(source,out.latent_points.float(),valid,K,T); y=warp.projected_valid[:,0].reshape(-1).float(); logits.append(raw[:,0].reshape(-1).cpu()); labels.append(y.cpu()); split.append(i)
 z=torch.cat(logits).numpy(); y=torch.cat(labels).numpy().astype(float); n=len(z); cal=np.arange(n)[np.arange(n)%len(ds)<max(1,len(ds)//3)]; test=np.setdiff1d(np.arange(n),cal)
 lt=torch.tensor(z[cal],dtype=torch.float64); yt=torch.tensor(y[cal],dtype=torch.float64); temp=torch.tensor(1.,dtype=torch.float64,requires_grad=True); opt=torch.optim.LBFGS([temp],lr=.1,max_iter=80,line_search_fn="strong_wolfe")
 def closure(): opt.zero_grad(); loss=F.binary_cross_entropy_with_logits(lt/temp.clamp_min(.05),yt); loss.backward(); return loss
 opt.step(closure); temperature=float(temp.detach().clamp_min(.05)); rawp=1/(1+np.exp(-z[test])); calp=1/(1+np.exp(-z[test]/temperature)); report={"schema_version":1,"stage":"latent confidence probability calibration","label":"projected-valid under sensor GT pose; calibration-only label, not semantic correctness","calibration_pairs":int(len(ds)//3),"test_pairs":int(len(ds)-len(ds)//3),"temperature":temperature,"raw_test":metrics(rawp,y[test]),"calibrated_test":metrics(calp,y[test]),"scope":"post-hoc temperature scaling on held-out TUM pairs; geometry and confidence weights frozen"}
 a.output.mkdir(parents=True); (a.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); print("实验已完成")
if __name__=="__main__": main()
