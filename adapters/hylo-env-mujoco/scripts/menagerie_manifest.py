"""Write an asset manifest (`menagerie_<name>.json` beside hylo.envs.mujoco.assets) for one MuJoCo Menagerie folder at
the pinned commit: every file the models need (XML, assets/, the LICENSE), each with its sha256, fetched once to
compute the digests. Top-level pictures and Markdown files other than the LICENSE are left out.

    uv run python bench/adapters/hylo-env-mujoco/scripts/menagerie_manifest.py unitree_go2 [--name go2] [--license BSD-3-Clause]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMMIT = "b0655057a9df2abe39ea7fc989914ba6f248301c"
TREE = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{commit}?recursive=1"
RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{commit}/{path}"
OUT = Path(__file__).resolve().parents[1] / "src" / "hylo" / "envs" / "mujoco"
SKIP_TOP = (".png", ".jpg", ".jpeg", ".gif", ".md", ".webp", ".mp4")
LICENSES = {"BSD-3-Clause": "BSD 3-Clause", "Apache-2.0": "Apache License", "MIT": "MIT License",
            "BSD-2-Clause": "BSD 2-Clause", "CC-BY-4.0": "Creative Commons Attribution 4.0"}

_tree: list[dict] | None = None


def tree(commit: str) -> list[dict]:
    global _tree
    if _tree is None:
        with urllib.request.urlopen(TREE.format(commit=commit), timeout=60) as r:
            _tree = json.load(r)["tree"]
    return _tree


def guess_license(text: str) -> str:
    for spdx, words in LICENSES.items():
        if words.lower() in text.lower():
            return spdx
    if "Apache" in text:
        return "Apache-2.0"
    if "Redistribution and use in source and binary forms" in text:
        return "BSD-3-Clause" if "Neither the name" in text else "BSD-2-Clause"
    return "see LICENSE"


def manifest(folder: str, name: str, commit: str = COMMIT) -> dict:
    files = [t["path"][len(folder) + 1:] for t in tree(commit) if t["type"] == "blob" and t["path"].startswith(folder + "/")]
    keep = [f for f in files if "/" in f or f == "LICENSE" or not f.lower().endswith(SKIP_TOP)]
    if "LICENSE" not in keep:
        raise SystemExit(f"{folder} has no LICENSE at {commit[:12]}")

    def digest(rel: str) -> tuple[str, str, bytes]:
        with urllib.request.urlopen(RAW.format(commit=commit, path=f"{folder}/{rel}"), timeout=120) as r:
            data = r.read()
        return rel, hashlib.sha256(data).hexdigest(), data if rel == "LICENSE" else b""

    with ThreadPoolExecutor(max_workers=12) as pool:
        out = list(pool.map(digest, sorted(keep)))
    lic = next(d for rel, _, d in out if rel == "LICENSE").decode(errors="replace")
    return {"source": "https://github.com/google-deepmind/mujoco_menagerie", "commit": commit, "folder": folder,
            "license": guess_license(lic), "sha256": {rel: h for rel, h, _ in out}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="menagerie folders, or folder=name to choose the manifest name")
    ap.add_argument("--commit", default=COMMIT)
    a = ap.parse_args()
    for spec in a.folders:
        folder, _, name = spec.partition("=")
        name = name or folder
        path = OUT / f"menagerie_{name}.json"
        if path.exists():
            print(f"exists: {path.name}")
            continue
        m = manifest(folder, name, a.commit)
        path.write_text(json.dumps(m, indent=1) + "\n")
        print(f"wrote {path.name}: {len(m['sha256'])} files, {m['license']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
