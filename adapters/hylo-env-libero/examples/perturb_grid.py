"""A labelled picture of every LIBERO perturbation on one task (runs in the LIBERO simulator venv).

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH=... PYTHONPATH=<hylo-core src>:<hylo-env-libero src> \\
        <libero venv>/bin/python bench/adapters/hylo-env-libero/examples/perturb_grid.py OUT.jpg [--suite libero_goal --task 8 --seed 3]
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hylo.core.task import TaskInstance
from hylo.envs.libero.env import LiberoEnv
from hylo.envs.libero.suites import libero_tasks

KINDS = [None, "position", "task", "language", "object", "camera", "noise", "robot_init"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    env = LiberoEnv(control="joint_position", tool_obs=True, image_size=256)
    tiles = []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for kind in KINDS:
        t = libero_tasks(a.suite, [a.task], perturb=kind)[0]
        obs = env.reset(TaskInstance(spec=t, seed=a.seed))
        img = Image.fromarray(obs.images["agentview"]).convert("RGB")
        canvas = Image.new("RGB", (256, 316), (24, 26, 30))
        canvas.paste(img, (0, 60))
        d = ImageDraw.Draw(canvas)
        d.text((8, 6), kind or "original", fill=(255, 220, 90), font=font)
        words = obs.instruction
        lines, cur = [], ""
        for w in words.split():
            if len(cur) + len(w) > 26:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        for i, ln in enumerate(lines[:2]):
            d.text((8, 24 + 16 * i), ln, fill=(230, 230, 230), font=font)
        tiles.append(np.asarray(canvas))
    env.close()
    row1, row2 = np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)
    Image.fromarray(np.concatenate([row1, row2], 0)).save(a.out, quality=88)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
