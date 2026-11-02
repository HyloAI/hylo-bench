"""Pretrained locomotion policies in the stage world (RFC-0008 part 8): inspected without running them, matched to a
robot before use, and executed on the CPU with numpy.

Weights files are never deserialised by a general unpickler and never executed:

```text
  sniff(path)           the format from its bytes: onnx (protobuf), safetensors (length-prefixed JSON header),
                        torchscript (a zip with code/), torch-checkpoint (a zip with data.pkl and no code/), unknown
  read_torchscript      the tensors of a TorchScript archive as numpy arrays: its tensor index (data.pkl) is read by
                        an unpickler that knows only tensor rebuilding and stand-ins for module objects (anything
                        else is refused), and the raw storages are read as little-endian arrays
  NumpyPolicy           the network rebuilt from those tensors: an optional LSTM memory (one layer) followed by an
                        MLP (Linear / ELU), the common shape of exported locomotion policies
```

A `LocoPolicy` says what the network expects: the robot pack, the controlled joints in order, their default angles,
PD gains, the observation layout (base angular velocity, projected gravity, command, joint offsets, joint
velocities, last action, gait phase) with its scales, the action scale and the control period. `match(policy, robot)`
lists every difference between that and the robot in the world (joints, drive, timestep, free base) as
{path, expected, actual}; the world refuses a policy action while the list is not empty.
"""

from __future__ import annotations

import io
import json
import math
import pickle
import struct
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from hylo.core.contract import Contract

from .assets import assets


# ------------------------------------------------------------------------------------------------ formats


def sniff(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    head = p.read_bytes()[:16] if p.is_file() else b""
    out: dict[str, Any] = {"path": str(p), "bytes": p.stat().st_size if p.is_file() else 0}
    if head[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
        code = any("/code/" in n for n in names)
        pkl = any(n.endswith("data.pkl") for n in names)
        out["format"] = "torchscript" if code else "torch-checkpoint" if pkl else "zip"
        out["entries"] = len(names)
    elif len(head) >= 8 and 0 < struct.unpack("<Q", head[:8])[0] < 100_000_000 and head[8:9] == b"{":
        out["format"] = "safetensors"
    elif head[:1] == b"\x08":
        out["format"] = "onnx"
    else:
        out["format"] = "unknown"
    return out


class _Stub:
    """Stand-in for a TorchScript module object: keeps its state, does nothing else."""

    def __init__(self, *a: Any, **k: Any) -> None:
        self.__dict__["_args"] = a

    def __setstate__(self, state: Any) -> None:
        self.__dict__["_state"] = state


_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
           "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_, "ByteStorage": np.uint8}


class _TensorIndex(pickle.Unpickler):
    def __init__(self, data: bytes, storages: dict[str, bytes]) -> None:
        super().__init__(io.BytesIO(data))
        self.storages = storages

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild
        if module == "torch" and name in _DTYPES:
            return name
        if module == "collections" and name == "OrderedDict":
            return dict
        if module.startswith("__torch__.") or module == "__torch__":
            return type(name, (_Stub,), {})
        if module == "torch.jit._pickle" and name == "restore_type_tag":
            return lambda value, tag: value
        if module == "torch.jit._pickle" and name in ("build_tensorlist", "build_intlist", "build_doublelist",
                                                       "build_boollist", "build_tensor_from_id"):
            return lambda value: value                    # plain lists of already-read values
        raise pickle.UnpicklingError(f"refused {module}.{name}: only tensors and module stand-ins are read")

    def persistent_load(self, pid: Any) -> Any:
        _, kind, key, _location, numel = pid
        dt = _DTYPES.get(kind if isinstance(kind, str) else getattr(kind, "__name__", ""), np.float32)
        return np.frombuffer(self.storages[str(key)], dtype=np.dtype(dt).newbyteorder("<"))[: int(numel)]


def _rebuild(storage: np.ndarray, offset: int, size: tuple, stride: tuple, *rest: Any) -> np.ndarray:
    """A tensor from its storage, offset, shape and strides (in elements), copied into its own array."""
    if not size:
        return np.array(storage[offset], dtype=storage.dtype)
    view = np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                           strides=tuple(int(s) * storage.itemsize for s in stride))
    return np.array(view)


def read_torchscript(path: str | Path) -> dict[str, np.ndarray]:
    """Module-path -> array for every tensor attribute of a TorchScript archive (no code is run)."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pkl = next(n for n in names if n.endswith("/data.pkl") or n == "data.pkl")
        base = pkl[: -len("data.pkl")]
        storages = {n[len(base) + 5:]: z.read(n) for n in names if n.startswith(base + "data/")}
        root = _TensorIndex(z.read(pkl), storages).load()
    out: dict[str, np.ndarray] = {}

    def walk(obj: Any, prefix: str) -> None:
        state = obj.__dict__.get("_state") if isinstance(obj, _Stub) else obj
        if isinstance(state, dict):
            for k, v in state.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, np.ndarray):
                    out[key] = v
                elif isinstance(v, (_Stub, dict)):
                    walk(v, key)
    walk(root, "")
    return out


# ------------------------------------------------------------------------------------------------ the network


class NumpyPolicy:
    """LSTM memory (optional) + MLP with ELU between layers, from read_torchscript's tensors."""

    def __init__(self, t: dict[str, np.ndarray]) -> None:
        self.lstm = None
        if "memory.weight_ih_l0" in t:
            self.lstm = {k: t[f"memory.{k}"].astype(np.float64) for k in
                         ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")}
            self.hidden = self.lstm["weight_hh_l0"].shape[1]
        idx = sorted({int(k.split(".")[1]) for k in t if k.startswith("actor.") and k.endswith(".weight")})
        self.layers = [(t[f"actor.{i}.weight"].astype(np.float64), t[f"actor.{i}.bias"].astype(np.float64)) for i in idx]
        if not self.layers:
            raise ValueError("no actor.<i>.weight tensors: not an exported actor network")
        self.inputs = (self.lstm["weight_ih_l0"].shape[1] if self.lstm else self.layers[0][0].shape[1])
        self.outputs = self.layers[-1][0].shape[0]
        self.reset()

    def reset(self) -> None:
        if self.lstm is not None:
            self.h = np.zeros(self.hidden)
            self.c = np.zeros(self.hidden)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, np.float64)
        if self.lstm is not None:
            L = self.lstm
            g = L["weight_ih_l0"] @ x + L["bias_ih_l0"] + L["weight_hh_l0"] @ self.h + L["bias_hh_l0"]
            i, f, gg, o = np.split(g, 4)                              # PyTorch gate order: input, forget, cell, output
            sig = lambda v: 1.0 / (1.0 + np.exp(-v))                  # noqa: E731
            self.c = sig(f) * self.c + sig(i) * np.tanh(gg)
            self.h = sig(o) * np.tanh(self.c)
            x = self.h
        for k, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if k < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))   # ELU(alpha=1)
        return x


# ------------------------------------------------------------------------------------------------ policy specs


class LocoPolicy(Contract):
    name: str
    title: str
    pack: str                                     # the robot pack it was trained for
    asset: str                                    # asset manifest holding the weights
    weights: str                                  # file inside it
    joints: tuple[str, ...]                       # controlled joints, in the network's order
    default: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    action_scale: float = 0.25
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    cmd_scale: tuple[float, float, float] = (2.0, 2.0, 0.25)
    phase_period: float = 0.8
    sim_dt: float = 0.002
    decimation: int = 10
    license: str = ""
    source: str = ""
    notes: str = ""
    max_command: tuple[float, float, float] = Field(default=(0.8, 0.4, 1.0), description="|vx|, |vy|, |wz| limits")

    def weights_path(self) -> Path:
        return assets(self.asset) / self.weights


G1_LEGS = ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
           "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
           "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint")
H1_LEGS = ("left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint",
           "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
           "right_ankle_joint")
RLGYM = "https://github.com/unitreerobotics/unitree_rl_gym"

POLICIES: dict[str, LocoPolicy] = {p.name: p for p in [
    LocoPolicy(name="g1_walk", title="G1 velocity-tracking walk (12 leg joints, LSTM)", pack="g1_legs",
               asset="github_rlgym_policies", weights="g1/motion.pt", joints=G1_LEGS,
               default=(-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2, kp=(100, 100, 100, 150, 40, 40) * 2,
               kd=(2, 2, 2, 4, 2, 2) * 2, license="BSD-3-Clause", source=RLGYM),
    LocoPolicy(name="h1_walk", title="H1 velocity-tracking walk (10 leg joints, LSTM)", pack="h1_legs",
               asset="github_rlgym_policies", weights="h1/motion.pt", joints=H1_LEGS,
               default=(0.0, 0.0, -0.1, 0.3, -0.2) * 2, kp=(150, 150, 150, 200, 40) * 2, kd=(2, 2, 2, 4, 2) * 2,
               license="BSD-3-Clause", source=RLGYM),
]}


def policy(name: str) -> LocoPolicy:
    try:
        return POLICIES[name]
    except KeyError:
        raise KeyError(f"no policy {name!r}; policies: {sorted(POLICIES)}") from None


@lru_cache(maxsize=None)
def network(name: str) -> tuple[NumpyPolicy, dict[str, Any]]:
    p = policy(name)
    path = p.weights_path()
    info = sniff(path)
    if info["format"] != "torchscript":
        raise ValueError(f"{name}: expected a TorchScript archive, found {info['format']}")
    net = NumpyPolicy(read_torchscript(path))
    info.update(inputs=net.inputs, outputs=net.outputs, recurrent=net.lstm is not None)
    return net, info


def observation_size(p: LocoPolicy) -> int:
    return 3 + 3 + 3 + 3 * len(p.joints) + 2


def match(p: LocoPolicy, world: Any, robot: Any) -> list[dict[str, Any]]:
    """Every difference between what the policy expects and the robot in the world."""
    diffs = []
    if robot.pack.name != p.pack:
        diffs.append({"path": "pack", "expected": p.pack, "actual": robot.pack.name})
    missing = [j for j in p.joints if j not in robot.joints]
    if missing:
        diffs.append({"path": "joints", "expected": list(p.joints), "actual": f"missing {missing}"})
    if robot.free_dof is None:
        diffs.append({"path": "base", "expected": "free", "actual": "fixed"})
    if abs(world.model.opt.timestep - p.sim_dt) > 1e-9:
        diffs.append({"path": "timestep", "expected": p.sim_dt, "actual": world.model.opt.timestep})
    if robot.pack.drive != "torque":
        diffs.append({"path": "drive", "expected": "torque", "actual": robot.pack.drive})
    try:
        net, _ = network(p.name)
        if net.inputs != observation_size(p) or net.outputs != len(p.joints):
            diffs.append({"path": "network", "expected": [observation_size(p), len(p.joints)],
                          "actual": [net.inputs, net.outputs]})
    except Exception as exc:  # noqa: BLE001
        diffs.append({"path": "weights", "expected": "a readable TorchScript archive", "actual": str(exc)[:200]})
    return diffs


@dataclass
class PolicyRun:
    """World controller: PD torques every step toward the network's targets, the network every `decimation` steps."""

    robot: Any
    world: Any
    spec: LocoPolicy
    command: np.ndarray
    n: int
    done: bool = False
    steps: int = 0

    def __post_init__(self) -> None:
        m = self.world.model
        r = self.robot
        self.net, _ = network(self.spec.name)
        self.net.reset()
        self.jid = [r.joint_ids[r.joints.index(j)] for j in self.spec.joints]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.jid])
        self.act = []
        for j in self.jid:
            k = next(k for k, jj in r.act_joint.items() if jj == j)
            self.act.append(r.actuator_ids[k])
        self.kp, self.kd = np.asarray(self.spec.kp, float), np.asarray(self.spec.kd, float)
        self.default = np.asarray(self.spec.default, float)
        self.action = np.zeros(len(self.jid))
        self.target = self.default.copy()
        lim = np.asarray(self.spec.max_command)
        self.command = np.clip(np.asarray(self.command, float), -lim, lim)
        self.start = r.base_pose(self.world.data)
        self.t0 = float(self.world.data.time)
        limited = m.actuator_ctrllimited[self.act].astype(bool)
        lo, hi = m.actuator_ctrlrange[self.act].T
        self.lo, self.hi = np.where(limited, lo, -np.inf), np.where(limited, hi, np.inf)
        self.robot.policy_active = True

    def _obs(self) -> np.ndarray:
        d, s = self.world.data, self.spec
        f = self.robot.free_qpos
        qw, qx, qy, qz = d.qpos[f + 3: f + 7]
        grav = np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])
        omega = d.qvel[self.robot.free_dof + 3: self.robot.free_dof + 6]
        t = (self.steps + 1) * s.sim_dt
        ph = (t % s.phase_period) / s.phase_period
        return np.concatenate([omega * s.ang_vel_scale, grav, self.command * np.asarray(s.cmd_scale),
                               (d.qpos[self.qadr] - self.default) * s.dof_pos_scale, d.qvel[self.dadr] * s.dof_vel_scale,
                               self.action, [math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)]])

    def update(self, world: Any) -> None:
        d = world.data
        self.steps += 1
        if self.steps % self.spec.decimation == 0:
            self.action = self.net(self._obs())
            self.target = self.action * self.spec.action_scale + self.default
        tau = self.kp * (self.target - d.qpos[self.qadr]) - self.kd * d.qvel[self.dadr]
        d.ctrl[self.act] = np.clip(tau, self.lo, self.hi)
        self.done = self.steps >= self.n
        if self.done:
            self.robot.policy_active = False
            for j, k in zip(self.jid, self.act):       # keep standing in the last posture under the world's PD
                kk = self.robot.actuator_ids.index(k)
                self.robot.target[kk] = float(d.qpos[world.model.jnt_qposadr[j]])

    def measured(self, world: Any) -> dict[str, Any]:
        p0, y0 = self.start
        p1, y1 = self.robot.base_pose(world.data)
        dt = max(float(world.data.time) - self.t0, 1e-9)
        dxy = p1[:2] - p0[:2]
        fwd = np.array([math.cos(y0), math.sin(y0)])
        return {"forward_m": round(float(dxy @ fwd), 3), "moved_m": round(float(np.linalg.norm(dxy)), 3),
                "speed_mps": round(float(np.linalg.norm(dxy) / dt), 3),
                "turned_deg": round(math.degrees((y1 - y0 + math.pi) % (2 * math.pi) - math.pi), 1),
                "upright": bool(self.robot.base_up(world.data) > 0.8), "base_height": round(float(p1[2]), 3),
                "command": np.round(self.command, 3).tolist()}


def describe() -> list[dict[str, Any]]:
    return [{"name": p.name, "title": p.title, "pack": p.pack, "joints": len(p.joints), "license": p.license,
             "source": p.source, "max_command": list(p.max_command)} for p in POLICIES.values()]


__all__ = ["LocoPolicy", "NumpyPolicy", "POLICIES", "PolicyRun", "describe", "match", "network", "policy",
           "read_torchscript", "sniff"]


if __name__ == "__main__":  # pragma: no cover
    for name in POLICIES:
        net, info = network(name)
        print(name, json.dumps(info))
