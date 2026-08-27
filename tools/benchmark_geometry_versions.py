"""Single-image geometry A/B harness. Full inference NOT run in this assessment.

Uses unmodified upstream infer(). No substitute kernels, model stubs or fake weights.
Run each checkpoint in a fresh process. Repeated-image p95 is a microbenchmark,
not cross-scene p95, action latency or game end-to-end latency.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--version", choices=["v2", "v3"], required=True)
    p.add_argument("--checkpoint", required=True, help="Local model.pt or Hugging Face repo ID")
    p.add_argument("--revision", help="Required for remote checkpoint: immutable HF commit")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", default="0,1,3", help="v3 SSR steps; v2 always uses 0")
    p.add_argument("--num-tokens", type=int, default=1200)
    p.add_argument("--max-size", type=int, default=640)
    p.add_argument("--fov-x", type=float)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = p.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.num_tokens < 1 or args.max_size < 1:
        p.error("repeat/num-tokens/max-size must be positive; warmup must be nonnegative")
    if (args.output / "metadata.json").exists():
        p.error("Output already contains metadata.json; choose a fresh output directory")
    steps = sorted(set(map(int, args.steps.split(",")))) if args.version == "v3" else [0]
    if not steps or min(steps) < 0:
        p.error("steps must be nonnegative")
    local_checkpoint = Path(args.checkpoint)
    if not local_checkpoint.is_file() and not args.revision:
        p.error("Remote checkpoints require --revision for reproducibility")
    import numpy as np
    from PIL import Image
    import torch
    from huggingface_hub import hf_hub_download
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    sys.path.insert(0, str(args.repo.resolve()))
    model_class = importlib.import_module(f"moge.model.{args.version}").MoGeModel
    if not local_checkpoint.is_file():
        local_checkpoint = Path(hf_hub_download(args.checkpoint, "model.pt", revision=args.revision))
    checkpoint_hash = sha256(local_checkpoint)
    load_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model = model_class.from_pretrained(local_checkpoint).cuda().eval()
    warning_text = [str(w.message) for w in recorded]
    bad_keys = [x for x in warning_text if "parameter(s)" in x]
    if bad_keys:
        raise RuntimeError("Checkpoint/model mismatch: " + " | ".join(bad_keys))
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - load_start) * 1000
    pil = Image.open(args.image).convert("RGB")
    if max(pil.size) > args.max_size:
        scale = args.max_size / max(pil.size)
        pil = pil.resize(tuple(round(v * scale) for v in pil.size), Image.Resampling.BOX)
    rgb = np.array(pil)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).cuda().float().div_(255)
    common = dict(num_tokens=args.num_tokens, fov_x=args.fov_x,
                  use_fp16=args.precision == "fp16", force_projection=True, apply_mask=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for count in steps:
        kwargs = dict(common)
        if args.version == "v3":
            kwargs.update(refine_steps=count, return_per_step=False)
        def infer():
            with torch.inference_mode():
                return model.infer(image, **kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = infer()
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - start) * 1000
        del output
        for _ in range(args.warmup):
            output = infer()
            del output
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = infer()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
            del output
        allocated = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.max_memory_reserved() / 2**20
        output = infer()  # Untimed export; no per-step output during latency measurements.
        arrays = {k: v.detach().cpu().numpy() for k, v in output.items() if torch.is_tensor(v)}
        arrays.setdefault("normal", np.empty((0,), dtype=np.float32))
        arrays.update(rgb=rgb, coordinate_convention=np.array("opencv_x_right_y_down_z_forward"))
        depth, mask = arrays["depth"], arrays["mask"].astype(bool)
        valid = mask & np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise RuntimeError("No valid positive depth pixels")
        step_dir = args.output / f"step_{count}"
        step_dir.mkdir(exist_ok=True)
        np.savez_compressed(step_dir / "geometry.npz", **arrays)
        results.append(dict(refine_steps=count, first_call_ms=first_call_ms,
                            warmup_runs=args.warmup, repeats=args.repeat,
                            mean_ms=float(np.mean(times)), p50_ms=float(np.median(times)),
                            p95_ms=float(np.percentile(times, 95)), all_ms=times,
                            peak_allocated_mib=allocated, peak_reserved_mib=reserved,
                            valid_fraction=float(valid.mean()),
                            intrinsics_normalized=arrays["intrinsics"].tolist(),
                            depth_median=float(np.median(depth[valid]))))
        del output, arrays
    record = dict(scope="Geometry infer() only; resident RGB/model; CPU focal recovery included; decoding/H2D/export/download excluded",
                  model_version=args.version, checkpoint_sha256=checkpoint_hash,
                  checkpoint_id=args.checkpoint if args.revision else local_checkpoint.name,
                  checkpoint_revision=args.revision, model_load_ms=model_load_ms,
                  repo_commit=subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
                  image_sha256=sha256(args.image), image_hw=list(rgb.shape[:2]),
                  num_tokens=args.num_tokens, precision=args.precision, fov_x=args.fov_x,
                  refiner_depth_resolution=getattr(model, "refiner_depth_resolution", None),
                  parameters=sum(x.numel() for x in model.parameters()),
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                  python=platform.python_version(), os=platform.system(), load_warnings=warning_text,
                  results=results)
    (args.output / "metadata.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
