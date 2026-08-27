"""Fetch exact upstream revisions without overwriting an existing checkout."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = {
    "MoGe": ("https://github.com/microsoft/MoGe.git", "925b8ed835a7a9cdb7578ba15c658a0afc969030"),
    "Matrix-Game": ("https://github.com/SkyworkAI/Matrix-Game.git", "71c3cd7f741311f8100f6cf9cde942b6c1378d11"),
}


def main():
    root = ROOT / "third_party"
    root.mkdir(exist_ok=True)
    for name, (url, revision) in UPSTREAM.items():
        dest = root / name
        if dest.exists():
            actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(dest), "status", "--porcelain"], text=True)
            if actual != revision or dirty:
                raise SystemExit(f"{name}: existing checkout differs or has local changes; preserved. Use a fresh repository copy for reproducibility.")
            print(f"{name}: already at {revision}")
            continue
        subprocess.run(["git", "clone", "--no-checkout", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", revision], check=True)
    print("Dependencies ready. Install CUDA torch, requirements.txt, then: python -m pip install -e third_party/MoGe")


if __name__ == "__main__":
    main()
