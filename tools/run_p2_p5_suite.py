#!/usr/bin/env python3
"""P2-P5 compact audit: staged transport, target occupancy, chunk consistency."""
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from latent_geometry_head_v2 import LatentGeometryHeadV2
from latent_reprojection_loss import forward_splat_latent, compare_warp_to_copy
from tools.train_latent_reprojection_head import PairDataset, filter_pairs

class OccupancyHead(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(c+1+4,48,3,padding=1),nn.GroupNorm(8,48),nn.SiLU(),
                               nn.Conv2d(48,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,1,1))
    def forward(self,z,depth,motion):
        m=motion.view(-1,1,1,1).expand(-1,4,z.shape[-2],z.shape[-1])
        d=depth.log().clamp(-8,8)
        return self.net(torch.cat((z.float(),d,m),1))

def load_model(path, device):
    m=LatentGeometryHeadV2().to(device); ck=torch.load(path,map_location='cpu',weights_only=False)
    m.load_state_dict(ck['model'],strict=True); return m

def pair_pose(row):
    R,_=cv2.Rodrigues(np.asarray(row['rvec'],np.float32)); T=torch.eye(4); T[:3,:3]=torch.from_numpy(R); T[:3,3]=torch.tensor(row['tvec']); return T

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--init',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--epochs1',type=int,default=2); ap.add_argument('--epochs2',type=int,default=3); ap.add_argument('--batch',type=int,default=8); args=ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    device=torch.device('cuda:0'); torch.manual_seed(19); np.random.seed(19)
    cache=torch.load(args.cache,map_location='cpu',weights_only=False); rows=filter_pairs(argparse.Namespace(pairs=args.pairs,min_inliers=200,min_inlier_ratio=.6,max_median_reprojection_px=1.5))
    val_s={'tum_rpy_val','tum_xyz_val'}; test_s={'tum_rpy_test','tum_xyz_test'}
    tr=[r for r in rows if r['scene'] not in val_s|test_s]; va=[r for r in rows if r['scene'] in val_s]; te=[r for r in rows if r['scene'] in test_s]
    model=load_model(args.init,device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    def loop(data, transport):
        model.train(); vals=[]
        for s,t,d,v,K,T,motion in DataLoader(PairDataset(cache,data),batch_size=args.batch,shuffle=transport):
            s,t,d,v,K,T,motion=[x.to(device) for x in (s,t,d,v,K,T,motion)]
            out=model(s,K); geom=((out.latent_depth.log()-d.clamp_min(1e-5).log()).abs()*v).sum()/v.sum().clamp_min(1)
            valid=F.binary_cross_entropy_with_logits(out.latent_valid_logits,v)
            loss=geom+.1*valid
            if transport:
                w=forward_splat_latent(s,out.latent_points,v,K,T); mask=w.projected_valid.float(); trn=((w.latent-t).abs()*mask).sum()/(mask.sum()*s.shape[1]).clamp_min(1); loss=loss+.35*trn
            opt.zero_grad(); loss.backward(); opt.step(); vals.append(float(loss.detach()))
        return float(np.mean(vals))
    @torch.inference_mode()
    def evaluate(data):
        model.eval(); a=[]
        for s,t,d,v,K,T,motion in DataLoader(PairDataset(cache,data),batch_size=1):
            s,t,d,v,K,T=[x.to(device) for x in (s,t,d,v,K,T)]; o=model(s,K); w=forward_splat_latent(s,o.latent_points,v,K,T); c=compare_warp_to_copy(w,s,t); a.append({'warp_l1':float(c['warp_valid_l1']),'copy_l1':float(c['copy_valid_l1']),'coverage':float(c['coverage']),'abs_rel':float((((o.latent_depth-d).abs()/d.clamp_min(1e-5))*v).sum()/v.sum().clamp_min(1))})
        return {'count':len(a),'warp_l1':float(np.mean([x['warp_l1'] for x in a])),'copy_l1':float(np.mean([x['copy_l1'] for x in a])),'win_rate':float(np.mean([x['warp_l1']<x['copy_l1'] for x in a])),'coverage':float(np.mean([x['coverage'] for x in a])),'abs_rel':float(np.mean([x['abs_rel'] for x in a]))}
    history=[]
    for phase,n in [('geometry',args.epochs1),('transport',args.epochs2)]:
        for e in range(n): history.append({'phase':phase,'epoch':e+1,'loss':loop(tr,phase=='transport'),'val':evaluate(va)})
    test=evaluate(te)
    occ=OccupancyHead().to(device); oo=torch.optim.AdamW(occ.parameters(),lr=5e-4)
    occ_hist=[]
    for e in range(3):
        occ.train(); ls=[]
        for s,t,d,v,K,T,motion in DataLoader(PairDataset(cache,tr),batch_size=args.batch,shuffle=True):
            s,t,d,v,K,T,motion=[x.to(device) for x in (s,t,d,v,K,T,motion)]; o=model(s,K).latent_points; w=forward_splat_latent(s,o,v,K,T); y=w.projected_valid.float(); logits=occ(s,d,motion/10); loss=F.binary_cross_entropy_with_logits(logits,y); oo.zero_grad(); loss.backward(); oo.step(); ls.append(float(loss.detach()))
        occ_hist.append(float(np.mean(ls)))
    @torch.inference_mode()
    def occ_eval(data):
        occ.eval(); ys=[]; ps=[]
        for s,t,d,v,K,T,motion in DataLoader(PairDataset(cache,data),batch_size=1):
            s,d,K,T,motion=[x.to(device) for x in (s,d,K,T,motion)]; o=model(s,K); y=forward_splat_latent(s,o.latent_points,v.to(device),K,T).projected_valid.float(); p=torch.sigmoid(occ(s,d,motion.to(device)/10)); ys.extend(y.flatten().cpu().tolist()); ps.extend(p.flatten().cpu().tolist())
        y=np.asarray(ys); p=np.asarray(ps); order=np.argsort(p); ranks=np.empty_like(order); ranks[order]=np.arange(len(order)); pos=y.sum(); neg=len(y)-pos; auc=float(((ranks[y>0]).sum()-pos*(pos-1)/2)/(pos*neg)) if pos>0 and neg>0 else float('nan'); return {'auc':auc,'brier':float(np.mean((p-y)**2)),'positive_rate':float(y.mean())}
    occupancy=occ_eval(te)
    # P5: framewise chunk consistency; groups are consecutive cache entries within each scene.
    model.eval(); chunk={}
    for L in (1,3,5):
        vals=[]
        for scene in sorted(set(cache['scenes'])):
            ix=[i for i,s in enumerate(cache['scenes']) if s==scene]
            if len(ix)<L: continue
            z=cache['latent'][ix[:L]].to(device); K=cache['intrinsics'][ix[:L]].to(device); o=model(z,K); ld=o.latent_depth.log(); vals.append(float((ld[1:]-ld[:-1]).abs().mean().cpu()) if L>1 else 0.0)
        chunk[str(L)]={'mean_adjacent_log_depth_change':float(np.mean(vals)),'groups':len(vals)}
    args.out.mkdir(parents=True); torch.save({'model':model.state_dict(),'occupancy':occ.state_dict()},args.out/'checkpoint.pt')
    report={'schema_version':1,'stage':'P2-P5 latent 3D completion suite','repository':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip(),'splits':{'train':len(tr),'val':len(va),'test':len(te)},'p3_staged_transport':{'history':history,'test':test},'p4_target_view_occupancy':{'train_loss':occ_hist,'test':occupancy,'label':'projected_valid under registered estimated pose; not full reuse correctness'},'p5_chunk_consistency':chunk,'output_contract':{'latent_depth':'[B,1,44,80]','latent_points':'[B,3,44,80]','latent_valid':'[B,1,44,80]','latent_confidence':'[B,1,44,80]','shape_changed':False}}
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
