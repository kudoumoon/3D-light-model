#!/usr/bin/env python3
"""合并使用同一 frozen VAE 的 latent3D cache 与 pair manifest。"""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cache",type=Path,action="append",required=True); parser.add_argument("--pairs",type=Path,action="append",required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    if len(args.cache)!=len(args.pairs): raise ValueError("cache and pairs counts must match")
    if args.output.exists(): raise FileExistsError(args.output)
    caches=[torch.load(path,map_location="cpu",weights_only=False) for path in args.cache]
    shape={tuple(cache["latent"].shape[1:]) for cache in caches}; channels={tuple(cache["depth"].shape[1:]) for cache in caches}
    if len(shape)!=1 or len(channels)!=1: raise ValueError("cache tensor shapes do not match")
    ids=[sample for cache in caches for sample in cache["sample_ids"]]
    if len(ids)!=len(set(ids)): raise ValueError("sample ids overlap across caches")
    merged={key:torch.cat([cache[key] for cache in caches]) for key in ("latent","depth","valid","intrinsics")}; merged["sample_ids"]=ids; merged["scenes"]=[scene for cache in caches for scene in cache["scenes"]]; merged["sources"]=[str(path) for path in args.cache]
    pair_rows=[row for path in args.pairs for row in json.loads(path.read_text())["pairs"]]
    args.output.mkdir(parents=True); torch.save(merged,args.output/"cache.pt"); (args.output/"pairs.json").write_text(json.dumps({"schema_version":1,"stage":"merged real latent3d training pairs","pairs":pair_rows},indent=2))
    report={"schema_version":1,"stage":"merge compatible real latent3d training sets","repository_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),"inputs":[{"cache":str(c),"cache_sha256":sha256(c),"pairs":str(p),"pairs_sha256":sha256(p)} for c,p in zip(args.cache,args.pairs)],"frames":len(ids),"pairs":len(pair_rows),"latent_shape":list(merged["latent"].shape),"scenes":sorted(set(merged["scenes"]))}
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); print("实验已完成")
if __name__=="__main__": main()
