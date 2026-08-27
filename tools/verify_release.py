"""Check the bounded release for broken local links, secrets and unwanted assets."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "__pycache__", "third_party", "runs", ".pytest_cache"}
FORBIDDEN = {".pt", ".pth", ".ckpt", ".safetensors", ".npz", ".npy", ".ply", ".mp4", ".pem", ".key"}


def main():
    failures, count, total = [], 0, 0
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(p in IGNORED for p in relative.parts):
            continue
        count += 1
        size = path.stat().st_size
        total += size
        if size > 10 * 1024 * 1024 or path.suffix in FORBIDDEN or path.name.startswith(".env"):
            failures.append(f"Unwanted/large file: {relative}")
        if path.suffix not in {".py", ".md", ".json", ".txt", ".ps1"}:
            continue
        content = path.read_text(encoding="utf-8")
        if re.search(r"[A-Za-z]:[\\/](?:Users|home)[\\/]", content):
            failures.append(f"Personal absolute path: {relative}")
        if re.search(r"(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{20,}", content):
            failures.append(f"Possible credential: {relative}")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", content):
                if target.startswith(("https://", "http://", "#", "mailto:")):
                    continue
                local = target.split("#")[0]
                if local and not (path.parent / local).exists():
                    failures.append(f"Broken local link in {relative}: {target}")
    print(f"Release files: {count}; total: {total / 1024**2:.2f} MiB")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print("PASS: no broken relative Markdown links, common credential patterns, personal absolute paths, large files or excluded binary model/data assets.")
    print("This bounded scan does not replace review; live website permissions and GPU timings are separate checks.")


if __name__ == "__main__":
    main()
