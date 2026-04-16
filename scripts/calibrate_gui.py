"""
calibrate_gui.py
================
Finds the exact coordinates on the screen to tell agent where each button is in the crafting inventory

Moves the cursor across the inventory GUI in small camera-degree steps and
saves an annotated screenshot at each position.  Filenames encode the exact
(pitch, yaw) offset from screen centre

These values are used in SLOTS

Three phases are needed to find all slot positions.

Phase 1 — recipe_book_btn, output_slot
---------------------------------------
  python scripts/calibrate_gui.py

  Sweeps ±14° pitch × ±20° yaw in 2° steps (~225 frames).
  Find the green recipe-book button and the output slot (right of 2×2 grid).

Phase 2 — recipe_planks, recipe_table (first pass), inv_slot
-------------------------------------------------------------
  python scripts/calibrate_gui.py --phase 2 --btn-pitch P --btn-yaw Y

  P Y are the recipe_book_btn values from phase 1.
  Clicks the recipe book to open it, then sweeps -24°→+14° yaw × ±12° pitch
  in 2° steps (~247 frames).  Find recipe_planks and inv_slot here.
  recipe_table may or may not appear until planks are in inventory.

Phase 3 — recipe_table (after planks are crafted)
--------------------------------------------------
  python scripts/calibrate_gui.py --phase 3 \\
      --btn-pitch P1  --btn-yaw Y1  \\
      --planks-pitch P2  --planks-yaw Y2  \\
      --output-pitch P3  --output-yaw Y3  \\
      --inv-pitch P4  --inv-yaw Y4

  Crafts planks (recipe book → planks recipe → output slot → inv slot), then
  sweeps the recipe book area so you can find the crafting table recipe entry.

Filename key
------------
  pm03_yp08.png  →  pitch = −3°,  yaw = +8°
  pp06_ym12.png  →  pitch = +6°,  yaw = −12°
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("Pillow required:  pip install Pillow")

import gym
import minerl
from shared_runtime import WORLD_SEED, patch_jvm_memory
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handlers.agent.start import InventoryAgentStart
from minerl.env._singleagent import _SingleAgentEnv

STEP_DELAY = 0.1   # seconds between every env step


# ── env setup ─────────────────────────────────────────────────────────────────
class CraftingEnvSpec(HumanSurvival):
    def __init__(self):
        super().__init__(name="CraftingTask-v0", max_episode_steps=2400)

    def create_agent_start(self):
        handlers = super().create_agent_start()
        handlers.append(InventoryAgentStart({0: {"type": "oak_log", "quantity": 1}}))
        return handlers


def _make_env():
    patch_jvm_memory("6G")
    print("Launching Minecraft…")
    spec = CraftingEnvSpec()
    env  = _SingleAgentEnv(env_spec=spec)
    env.seed(WORLD_SEED)
    obs  = env.reset()
    noop = env.action_space.no_op()

    print("Waiting for world to settle…")
    for _ in range(20):
        obs, _, done, _ = env.step(noop)
        time.sleep(STEP_DELAY)
        if done:
            raise RuntimeError("Env terminated during world settle")

    print("Opening inventory…")
    open_act = dict(noop)
    open_act["inventory"] = 1
    obs, _, _, _ = env.step(open_act)
    time.sleep(STEP_DELAY)
    for _ in range(5):
        obs, _, _, _ = env.step(noop)
        time.sleep(STEP_DELAY)

    return env, obs, noop


# ── low-level helpers ─────────────────────────────────────────────────────────
def _annotate(pov_hwc, label):
    img  = Image.fromarray(pov_hwc[:, :, :3])
    draw = ImageDraw.Draw(img)
    draw.text((11, 11), label, fill=(0, 0, 0))
    draw.text((10, 10), label, fill=(255, 255, 0))
    return img


def _step(env, noop, **overrides):
    """Send one action with the given key overrides, sleep, return (obs, done)."""
    act = dict(noop)
    for k, v in overrides.items():
        act[k] = v
    obs, _, done, _ = env.step(act)
    time.sleep(STEP_DELAY)
    return obs, done


def _wait(env, noop, n=8):
    """Send n no-ops and return (final_obs, done)."""
    obs, done = None, False
    for _ in range(n):
        obs, _, done, _ = env.step(noop)
        time.sleep(STEP_DELAY)
        if done:
            break
    return obs, done


def _move(env, noop, cur_pitch, cur_yaw, target_pitch, target_yaw):
    """Move cursor from current to target and return (obs, done, new_pitch, new_yaw)."""
    dp = target_pitch - cur_pitch
    dy = target_yaw   - cur_yaw
    obs, done = _step(env, noop,
                      camera=np.array([dp, dy], dtype=np.float32))
    return obs, done, target_pitch, target_yaw


def _click(env, noop, cur_pitch, cur_yaw, target_pitch, target_yaw):
    """Move to target then left-click. Returns (obs, done, new_pitch, new_yaw)."""
    obs, done, cp, cy = _move(env, noop, cur_pitch, cur_yaw, target_pitch, target_yaw)
    if done:
        return obs, done, cp, cy
    obs, done = _step(env, noop, attack=1)
    return obs, done, cp, cy


# ── sweep ─────────────────────────────────────────────────────────────────────
def _sweep(env, noop, out_dir, pitch_range, yaw_range, cur_pitch=0.0, cur_yaw=0.0):
    """
    Sweep over a grid, save one annotated frame per point.
    Returns (frames_saved, final_pitch, final_yaw).
    """
    os.makedirs(out_dir, exist_ok=True)
    total = len(pitch_range) * len(yaw_range)
    saved = 0

    for tp in pitch_range:
        for ty in yaw_range:
            obs, done, cur_pitch, cur_yaw = _move(
                env, noop, cur_pitch, cur_yaw, tp, ty)

            if done:
                print(f"\n  [WARNING] env terminated at p={tp} y={ty} "
                      f"after {saved}/{total} frames.")
                return saved, cur_pitch, cur_yaw

            label = f"p{tp:+.0f}  y{ty:+.0f}"
            fname = (f"p{tp:+03.0f}_y{ty:+03.0f}.png"
                     .replace("+", "p").replace("-", "m"))
            _annotate(obs["pov"], label).save(os.path.join(out_dir, fname))
            saved += 1

            if saved % 50 == 0 or saved == total:
                print(f"  {saved}/{total} frames…")

    return saved, cur_pitch, cur_yaw


# ── phase 1 ───────────────────────────────────────────────────────────────────
def phase1():
    env, obs, noop = _make_env()
    out_dir = os.path.join("calibration_frames", "phase1")
    os.makedirs(out_dir, exist_ok=True)

    _annotate(obs["pov"], "baseline  p0  y0").save(
        os.path.join(out_dir, "baseline.png"))
    print(f"Baseline saved → {out_dir}/baseline.png")

    pitch_range = list(range(-14, 15, 2))   # 15 values
    yaw_range   = list(range(-20, 21, 2))   # 21 values  →  315 frames
    total = len(pitch_range) * len(yaw_range)
    print(f"Sweeping {len(pitch_range)}×{len(yaw_range)} = {total} frames "
          f"(~{total * STEP_DELAY:.0f}s) into {out_dir}/…")

    saved, _, _ = _sweep(env, noop, out_dir, pitch_range, yaw_range)
    env.close()

    print(
        f"\nPhase 1 done — {saved} frames saved.\n\n"
    )


# ── phase 2 ───────────────────────────────────────────────────────────────────
def phase2(btn_pitch, btn_yaw):
    env, obs, noop = _make_env()
    out_dir = os.path.join("calibration_frames", "phase2")
    os.makedirs(out_dir, exist_ok=True)

    cur_p, cur_y = 0.0, 0.0

    # Click the recipe book button
    print(f"Clicking recipe book button at (p={btn_pitch:+.1f}, y={btn_yaw:+.1f})…")
    obs, done, cur_p, cur_y = _click(
        env, noop, cur_p, cur_y, btn_pitch, btn_yaw)
    if done:
        raise RuntimeError("Env terminated before clicking recipe book")

    obs, done = _wait(env, noop, n=8)
    if done:
        raise RuntimeError("Env terminated waiting for recipe book to open")

    _annotate(obs["pov"],
              f"recipe book open  p{btn_pitch:+.1f}  y{btn_yaw:+.1f}").save(
        os.path.join(out_dir, "recipe_book_open.png"))
    print(f"Recipe book baseline saved → {out_dir}/recipe_book_open.png")

    # Return cursor to centre before sweeping
    obs, done, cur_p, cur_y = _move(env, noop, cur_p, cur_y, 0.0, 0.0)
    if done:
        raise RuntimeError("Env terminated returning cursor to centre")

    # Wide sweep: left to catch recipe list, right to catch shifted output slot
    pitch_range = list(range(-12, 13, 2))   # 13 values
    yaw_range   = list(range(-24, 26, 2))   # 25 values  →  325 frames
    total = len(pitch_range) * len(yaw_range)
    print(f"Sweeping {len(pitch_range)}×{len(yaw_range)} = {total} frames "
          f"(~{total * STEP_DELAY:.0f}s) into {out_dir}/…")

    saved, _, _ = _sweep(env, noop, out_dir, pitch_range, yaw_range,
                         cur_pitch=cur_p, cur_yaw=cur_y)
    env.close()

    print(
        f"\nPhase 2 done — {saved} frames saved.\n\n"
    )


# ── phase 3 ───────────────────────────────────────────────────────────────────
def phase3(btn_pitch, btn_yaw,
           planks_pitch, planks_yaw,
           output_pitch, output_yaw,
           inv_pitch, inv_yaw):
    env, obs, noop = _make_env()
    out_dir = os.path.join("calibration_frames", "phase3")
    os.makedirs(out_dir, exist_ok=True)

    cur_p, cur_y = 0.0, 0.0

    # 1. Open recipe book
    print(f"Opening recipe book at (p={btn_pitch:+.1f}, y={btn_yaw:+.1f})…")
    obs, done, cur_p, cur_y = _click(
        env, noop, cur_p, cur_y, btn_pitch, btn_yaw)
    if done:
        raise RuntimeError("Env terminated clicking recipe book")
    obs, done = _wait(env, noop, n=8)
    if done:
        raise RuntimeError("Env terminated waiting for recipe book")

    _annotate(obs["pov"], "recipe book open").save(
        os.path.join(out_dir, "01_recipe_book_open.png"))

    # 2. Click planks recipe (auto-places log in grid)
    print(f"Clicking planks recipe at (p={planks_pitch:+.1f}, y={planks_yaw:+.1f})…")
    obs, done, cur_p, cur_y = _click(
        env, noop, cur_p, cur_y, planks_pitch, planks_yaw)
    if done:
        raise RuntimeError("Env terminated clicking planks recipe")
    obs, done = _wait(env, noop, n=5)
    if done:
        raise RuntimeError("Env terminated waiting for grid to populate")

    _annotate(obs["pov"], "planks in grid").save(
        os.path.join(out_dir, "02_planks_in_grid.png"))

    # 3. Click output slot (picks up 4 planks into cursor)
    print(f"Clicking output slot at (p={output_pitch:+.1f}, y={output_yaw:+.1f})…")
    obs, done, cur_p, cur_y = _click(
        env, noop, cur_p, cur_y, output_pitch, output_yaw)
    if done:
        raise RuntimeError("Env terminated clicking output slot")
    obs, done = _wait(env, noop, n=3)

    _annotate(obs["pov"], "planks picked up").save(
        os.path.join(out_dir, "03_planks_picked_up.png"))

    # 4. Deposit planks into first inventory slot
    print(f"Depositing planks into inv slot at (p={inv_pitch:+.1f}, y={inv_yaw:+.1f})…")
    obs, done, cur_p, cur_y = _click(
        env, noop, cur_p, cur_y, inv_pitch, inv_yaw)
    if done:
        raise RuntimeError("Env terminated depositing planks")
    obs, done = _wait(env, noop, n=5)

    _annotate(obs["pov"], "planks in inventory").save(
        os.path.join(out_dir, "04_planks_in_inventory.png"))
    print("Planks deposited. Recipe book should now show crafting table recipe.")

    # 5. Return cursor to centre and sweep to find crafting table recipe
    obs, done, cur_p, cur_y = _move(env, noop, cur_p, cur_y, 0.0, 0.0)
    if done:
        raise RuntimeError("Env terminated returning cursor to centre")

    pitch_range = list(range(-12, 13, 2))   # 13 values
    yaw_range   = list(range(-24, 26, 2))   # 25 values  →  325 frames
    total = len(pitch_range) * len(yaw_range)
    print(f"Sweeping {len(pitch_range)}×{len(yaw_range)} = {total} frames "
          f"(~{total * STEP_DELAY:.0f}s) into {out_dir}/…")

    saved, _, _ = _sweep(env, noop, out_dir, pitch_range, yaw_range,
                         cur_pitch=cur_p, cur_yaw=cur_y)
    env.close()

    print(
        f"\nPhase 3 done — {saved} frames saved.\n\n"
    )


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate GUI slot positions")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])

    # phase 2
    parser.add_argument("--btn-pitch",    type=float, default=None)
    parser.add_argument("--btn-yaw",      type=float, default=None)

    # phase 3 (also needs btn-pitch/yaw from above)
    parser.add_argument("--planks-pitch", type=float, default=None)
    parser.add_argument("--planks-yaw",   type=float, default=None)
    parser.add_argument("--output-pitch", type=float, default=None)
    parser.add_argument("--output-yaw",   type=float, default=None)
    parser.add_argument("--inv-pitch",    type=float, default=None)
    parser.add_argument("--inv-yaw",      type=float, default=None)

    args = parser.parse_args()

    if args.phase == 1:
        phase1()

    elif args.phase == 2:
        if args.btn_pitch is None or args.btn_yaw is None:
            parser.error("--phase 2 requires --btn-pitch and --btn-yaw")
        phase2(args.btn_pitch, args.btn_yaw)

    elif args.phase == 3:
        missing = [
            name for name, val in [
                ("--btn-pitch",    args.btn_pitch),
                ("--btn-yaw",      args.btn_yaw),
                ("--planks-pitch", args.planks_pitch),
                ("--planks-yaw",   args.planks_yaw),
                ("--output-pitch", args.output_pitch),
                ("--output-yaw",   args.output_yaw),
                ("--inv-pitch",    args.inv_pitch),
                ("--inv-yaw",      args.inv_yaw),
            ] if val is None
        ]
        if missing:
            parser.error(f"--phase 3 requires: {', '.join(missing)}")
        phase3(
            args.btn_pitch,    args.btn_yaw,
            args.planks_pitch, args.planks_yaw,
            args.output_pitch, args.output_yaw,
            args.inv_pitch,    args.inv_yaw,
        )
