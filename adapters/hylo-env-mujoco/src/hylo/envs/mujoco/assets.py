"""Third-party simulation assets, fetched on first use into the asset cache and never vendored.

The cache is `$HYLO_ASSETS` (default `~/.cache/hylo/assets`). Each asset is one upstream folder pinned to one commit
and verified file by file against a digest list kept beside this module (a `<name>.json` manifest: source, commit,
folder, licence, sha256 per file, the upstream LICENSE among them), so a cached copy that was changed or truncated is
refused rather than loaded.

    assets("menagerie_panda")     -> Path   MuJoCo Menagerie `franka_emika_panda` (Apache-2.0)
    assets("menagerie_aloha")     -> Path   MuJoCo Menagerie `aloha`: two ViperX 300 6-DoF arms (BSD-3-Clause)
    assets("menagerie_g1_hands")  -> Path   MuJoCo Menagerie `unitree_g1` with the three-finger hands (BSD-3-Clause)
    assets("menagerie_allegro")   -> Path   MuJoCo Menagerie `wonik_allegro`: the Allegro hand, left and right
                                            (BSD-2-Clause)
    panda_assets() / panda_assets_dir()     the Panda shortcuts used since the first environment

Manifests that pin folders of the same repository at the same commit share one cache folder per upstream folder;
a manifest may list only some files of it. Files a manifest needs that are missing from an existing cached folder
are fetched into it (same commit, verified digests), so one manifest never disturbs another's files.

Offline machines: put the upstream folder at `assets_dir(name)` by hand; it is verified the same way.

Other GitHub repositories (e.g. a robot model and pretrained policies published with a training framework): a
manifest with `"repo": "<owner>/<name>"` is fetched from that repository's raw files at its commit into
`<cache>/github/<owner>_<name>/<commit12>/<folder>`; `"paths": {rel: path from the repository root}` names files
that live outside the folder (its LICENSE at the root).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).parent
_RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{commit}/{folder}/{path}"
PANDA = "menagerie_panda"
ALOHA = "menagerie_aloha"
G1_HANDS = "menagerie_g1_hands"
ALLEGRO = "menagerie_allegro"


def cache_root() -> Path:
    return Path(os.environ.get("HYLO_ASSETS") or Path.home() / ".cache" / "hylo" / "assets").expanduser()


def manifests() -> list[str]:
    """Names of the asset manifests kept beside this module."""
    return sorted(p.stem for p in _HERE.glob("menagerie_*.json")) + sorted(p.stem for p in _HERE.glob("github_*.json"))


@lru_cache(maxsize=None)
def manifest(name: str = PANDA) -> dict:
    path = _HERE / f"{name}.json"
    if not path.is_file():
        raise KeyError(f"no asset manifest {name!r}; known: {manifests()}")
    m = json.loads(path.read_text())
    if "LICENSE" not in m["sha256"]:
        raise ValueError(f"asset manifest {name} must keep the upstream LICENSE")
    return m


def assets_dir(name: str = PANDA) -> Path:
    """Where the asset's folder lives in the cache (it may not be fetched yet)."""
    m = manifest(name)
    if m.get("repo"):
        return cache_root() / "github" / m["repo"].replace("/", "_") / m["commit"][:12] / m["folder"]
    return cache_root() / "mujoco_menagerie" / m["commit"][:12] / m["folder"]


def _marker(name: str) -> str:
    return f".hylo-verified-{name}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(folder: Path, name: str = PANDA) -> list[str]:
    """Files of manifest `name` missing from `folder` or whose digest differs (empty: all good)."""
    bad = []
    for rel, digest in manifest(name)["sha256"].items():
        p = folder / rel
        if not p.is_file() or _sha256(p) != digest:
            bad.append(rel)
    return bad


def _download(url: str, dest: Path, digest: str, attempts: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f"{dest.name}.partial-{os.getpid()}")
    err: Exception | None = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f)
            if _sha256(part) == digest:
                os.replace(part, dest)
                return
            err = ValueError(f"digest mismatch for {url}")
        except OSError as exc:
            err = exc
    part.unlink(missing_ok=True)
    raise RuntimeError(f"could not fetch {url}: {err}")


def _url(m: dict, rel: str) -> str:
    if m.get("repo"):
        path = (m.get("paths") or {}).get(rel) or f"{m['folder']}/{rel}"
        return f"https://raw.githubusercontent.com/{m['repo']}/{m['commit']}/{path}"
    return _RAW.format(commit=m["commit"], folder=m["folder"], path=rel)


def _fetch(m: dict, folder: Path, rels: list[str]) -> None:
    jobs = [(_url(m, rel), folder / rel, m["sha256"][rel]) for rel in rels]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda j: _download(*j), jobs))


def assets(name: str = PANDA, *, recheck: bool = False) -> Path:
    """The asset's upstream folder at the pinned commit, fetched and verified on first use. `recheck` re-hashes
    every file of a cached copy (normally a marker written after verification is trusted)."""
    import fcntl

    m = manifest(name)
    final = assets_dir(name)
    if (final / _marker(name)).is_file() and not recheck:
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    with open(final.parent / ".lock", "w") as lock:   # one fetch at a time across worker processes
        fcntl.flock(lock, fcntl.LOCK_EX)
        if final.is_dir():
            bad = verify(final, name)
            changed = [rel for rel in bad if (final / rel).is_file()]
            if changed:
                raise RuntimeError(f"{final} does not match the digests of {name} ({len(changed)} files differ, e.g. "
                                   f"{changed[:3]}); delete it to fetch it again")
            if bad:                                   # files this manifest needs that another one did not fetch
                try:
                    _fetch(m, final, bad)
                except Exception as exc:  # noqa: BLE001 - re-raised with what to do about it
                    raise RuntimeError(_offline_hint(name, final, exc)) from exc
            (final / _marker(name)).write_text(m["commit"] + "\n")
            return final
        tmp = final.with_name(f"{final.name}.partial-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            _fetch(m, tmp, list(m["sha256"]))
        except Exception as exc:  # noqa: BLE001 - re-raised with what to do about it
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError(_offline_hint(name, final, exc)) from exc
        (tmp / _marker(name)).write_text(m["commit"] + "\n")
        try:
            tmp.rename(final)
        except OSError:                       # another process finished first
            shutil.rmtree(tmp, ignore_errors=True)
    return final


def _offline_hint(name: str, final: Path, exc: Exception) -> str:
    m = manifest(name)
    repo = m.get("repo", "google-deepmind/mujoco_menagerie")
    return (f"fetching {repo}/{m['folder']} failed ({exc}). Offline: copy the {m['folder']} folder of {repo} at "
            f"commit {m['commit']} to {final}")


def panda_assets_dir() -> Path:
    """Where the Panda folder lives in the cache (it may not be fetched yet)."""
    return assets_dir(PANDA)


def panda_assets(*, recheck: bool = False) -> Path:
    """The Menagerie `franka_emika_panda` folder at the pinned commit, fetched and verified on first use."""
    return assets(PANDA, recheck=recheck)


__all__ = ["ALLEGRO", "ALOHA", "G1_HANDS", "PANDA", "assets", "assets_dir", "cache_root", "manifest", "manifests", "panda_assets",
           "panda_assets_dir", "verify"]
