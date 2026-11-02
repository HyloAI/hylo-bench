# hylo: worker-side (runs in a robosuite 1.5 venv: robosuite, mujoco)
"""robosuite 1.5 glue for the robot contract: environments built with absolute joint-position control of each arm,
cheap offscreen rendering, in-place reseeding, and the Panda's exported model with its measured hand."""

from __future__ import annotations

import importlib.metadata
from typing import Any

import numpy as np

from .export import cached, export_model, face_hand_geometry, parallel_hand_geometry, point_hand_geometry
from .panda import FINGER_TRAVEL, PandaNames

EXPORT_VERSION = "1"


def joint_position_config(kp: float, damping_ratio: float, robot: str | None = None) -> dict:
    """A composite controller config with the arm ("right" part) on JOINT_POSITION taking absolute targets (the action
    is the goal itself, unscaled; gravity compensated) and the GRIP gripper part. `robot` (e.g. "PandaOmron") starts
    from that robot's default config, so its other parts (a mobile base, a torso) keep their controllers."""
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(controller=None if robot else "BASIC", robot=robot)
    part = {"type": "JOINT_POSITION", "input_type": "absolute", "kp": kp, "damping_ratio": damping_ratio,
            "impedance_mode": "fixed", "interpolation": None, "input_max": 10.0, "input_min": -10.0,
            "output_max": 10.0, "output_min": -10.0, "qpos_limits": None, "gripper": {"type": "GRIP"}}
    parts = dict(cfg.get("body_parts", {}))
    parts["right"] = part
    if not robot:
        parts = {"right": part}
    cfg["body_parts"] = parts
    return cfg


def make(env_name: str, robots: str | list[str], kp: float = 300.0, damping_ratio: float = 1.0,
         controller_robot: str | None = None, **kwargs: Any) -> Any:
    """`robosuite.make` for the contract: offscreen rendering only (the adapter renders its own cameras), no camera
    observations, robosuite's done flag ignored (the adapter ends episodes), a fresh model on every reset, single-
    sample offscreen buffers. Extra kwargs go to the environment."""
    import robosuite

    kwargs.setdefault("has_renderer", False)
    env = robosuite.make(env_name, robots=robots, controller_configs=joint_position_config(kp, damping_ratio, controller_robot),
                         has_offscreen_renderer=True, use_camera_obs=False, ignore_done=True, **kwargs)
    single_sample_offscreen(env)
    return env


def single_sample_offscreen(env: Any) -> None:
    """Software rendering (OSMesa) pays for every multisample of the offscreen buffer on each render and read-out.
    robosuite builds a new model on each reset and creates the offscreen context right after: turn multisampling off
    on every model it builds, before that context exists (and rebuild the context made in __init__). Edges get
    aliased; nothing else changes."""
    original = env._initialize_sim

    def initialize(xml_string=None):
        original(xml_string)
        env.sim.model._model.vis.quality.offsamples = 0

    env._initialize_sim = initialize
    if getattr(env, "sim", None) is not None and env.sim._render_context_offscreen is not None:
        from robosuite.utils.binding_utils import MjRenderContextOffscreen

        env.sim.model._model.vis.quality.offsamples = 0
        env.sim._render_context_offscreen = None
        env.sim.add_render_context(MjRenderContextOffscreen(env.sim, device_id=env.render_gpu_device_id))


def reseed(env: Any, seed: int) -> None:
    """Reseed robosuite's generator in place (placement samplers and object builders keep a reference to it), so the
    next reset follows from `seed` alone."""
    env.rng.bit_generator.state = np.random.default_rng(int(seed)).bit_generator.state


def panda_xml(names: PandaNames) -> str:
    """robosuite 1.5's Panda (arm `names.index`) with its gripper, as one MJCF string (no mount: it never moves)."""
    from robosuite.models.grippers import gripper_factory
    from robosuite.models.robots import Panda

    robot = Panda(idn=names.index)
    robot.add_gripper(gripper_factory(names.gripper, idn=f"{names.index}_right"))
    return robot.get_xml()


def panda_model(names: PandaNames) -> dict[str, Any]:
    """{"model": exported MJCF path, "geometry": {"tcp_in_ee", "max_width", "tcp_offset"}} for an arm, from the asset
    cache's index when present (no robosuite import), else exported and measured now."""
    versions = "-".join(importlib.metadata.version(d) for d in ("robosuite", "mujoco"))

    def build() -> dict[str, Any]:
        path = export_model(panda_xml(names), "robosuite/panda", root_body=names.base,
                            flip_joints=names.finger_joints[1:], file_name="panda.xml", version=EXPORT_VERSION,
                            source_note=f"Exported by hylo.envs.robosuite_contract from robosuite {versions} (MIT): "
                                        f"robots/panda/robot.xml + the {names.gripper} gripper, as arm {names.index} "
                                        "of a task builds them.")
        if names.fingered:
            geo = parallel_hand_geometry(path, names.ee_body, names.finger_joints, names.pads, FINGER_TRAVEL)
        elif names.gripper == "WipingGripper":
            geo = face_hand_geometry(path, names.ee_body)
        else:
            geo = point_hand_geometry(path, names.ee_body, names.eef_body)
        return {"model": str(path), "geometry": geo}

    return cached(f"v{EXPORT_VERSION}-{versions}-{names.index}-{names.gripper}", "robosuite/panda", build)


__all__ = ["joint_position_config", "make", "panda_model", "panda_xml", "reseed", "single_sample_offscreen"]
