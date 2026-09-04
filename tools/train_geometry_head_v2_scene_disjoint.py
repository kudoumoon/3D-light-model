#!/usr/bin/env python3
"""Train/evaluate V2 on real TUM latent/depth cache; no M2 changes."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_geometry_head_v2 import LatentGeometryHeadV2

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--lr',type=float,default=1e-4); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('expose exactly one approved GPU')
    d=torch.load(a.cache,map_location='cpu',weights_only=False); train=[i for i,s in enumerate(d['scenes']) if s in {'tum_xyz_train','tum_rpy_train'}]; val=[i for i,s in enumerate(d['scenes']) if s in {'tum_xyz_val','tum_rpy_val'}]; test=[i for i,s in enumerate(d['scenes']) if s in {'tum_xyz_test','tum_rpy_test'}]
    def loader(ix,shuffle): return DataLoader(TensorDataset(d['latent'][ix],d['depth'][ix],d['valid'][ix],d['intrinsics'][ix]),batch_size=a.batch_size,shuffle=shuffle)
    dev=torch.device('cuda:0'); m=LatentGeometryHeadV2().to(dev,dtype=torch.bfloat16); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4)
    def evaluate(ix):
        m.eval(); rows=[]
        with torch.inference_mode():
            for z,depth,valid,K in loader(ix,False):
                z,depth,valid,K=[x.to(dev) for x in (z,depth,valid,K)]; o,aux=m.forward_with_auxiliary(z.to(torch.bfloat16),K.to(torch.bfloat16)); pred=o.latent_depth.float(); mask=valid.float(); rows.append({'abs_rel':float((((pred-depth.float()).abs()/depth.float().clamp_min(1e-5))*mask).sum()/mask.sum().clamp_min(1)), 'log_rmse':float((((pred.clamp_min(1e-5).log()-depth.float().clamp_min(1e-5).log()).square()*mask).sum()/mask.sum().clamp_min(1)).sqrt()), 'valid_iou':float((((o.latent_valid>0.5)&(valid>0.5)).float().sum()/((o.latent_valid>0.5)|(valid>0.5)).float().sum().clamp_min(1)))} )
        return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    history=[]; started=time.perf_counter()
    for epoch in range(a.epochs):
        m.train(); losses=[]
        for z,depth,valid,K in loader(train,True):
            z,depth,valid,K=[x.to(dev) for x in (z,depth,valid,K)]; o,aux=m.forward_with_auxiliary(z.to(torch.bfloat16),K.to(torch.bfloat16)); mask=valid.float(); logerr=F.smooth_l1_loss(o.latent_depth.float().clamp_min(1e-5).log(),depth.float().clamp_min(1e-5).log(),reduction='none'); loss=(logerr*mask).sum()/mask.sum().clamp_min(1)+0.1*F.binary_cross_entropy_with_logits(o.latent_valid_logits.float(),valid.float())+0.05*F.smooth_l1_loss(aux['depth_log_variance'].float(),torch.zeros_like(aux['depth_log_variance']).float())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        row={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val':evaluate(val)}; history.append(row); print(json.dumps(row),flush=True)
    report={'schema_version':1,'stage':'Geometry Head V2 scene-disjoint TUM training','split':{'train':len(train),'val':len(val),'test':len(test)},'epochs':a.epochs,'history':history,'test':evaluate(test),'parameters':sum(x.numel() for x in m.parameters()),'runtime_seconds':time.perf_counter()-started,'output_contract':{'latent_depth':'[B,1,44,80]','latent_points':'[B,3,44,80]','latent_valid':'[B,1,44,80]','latent_confidence':'[B,1,44,80]','shape_changed':False}}
    a.output.mkdir(parents=True); torch.save({'model':m.state_dict(),'report':report},a.output/'checkpoint.pt'); (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+"\n"); print('实验已完成',flush=True)
if __name__=='__main__': main()
