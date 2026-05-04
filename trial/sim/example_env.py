"""
trial/sim/example_env.py  –  ProbeEnv basic usage examples.

Run from the RecON repository root:
    python trial/sim/example_env.py
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))          # RecON/trial/sim
_root = os.path.dirname(os.path.dirname(_here))             # RecON
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import shared objects from playground.py.
# Heavy setup (data loading, volume reconstruction, surf_mesh) runs once here.
# The plotter.show() block is guarded by __main__ so it does NOT block on import.
import rl_env
from rl_env import (
    ProbeEnv,
    probe_surface, volume_up,
    scale_h, scale_w, H_img, W_img,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Create a headless environment  (plotter=None → no PyVista window)
# ─────────────────────────────────────────────────────────────────────────────
env = ProbeEnv(
    probe_surface, volume_up, scale_h, scale_w, H_img, W_img,
    plotter=None,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. reset()
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("2. reset()")
print("=" * 60)

obs = env.reset()
print(f"  obs  : shape={obs.shape}  dtype={obs.dtype}"
      f"  range=[{obs.min():.3f}, {obs.max():.3f}]")
print(f"  state: {env.state}")

obs = env.reset(s=0.2, t=0.7, angle=np.pi / 4)
print(f"\n  reset(s=0.2, t=0.7, angle=π/4)")
print(f"  state: {env.state}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. step()  –  translate / rotate in the local probe frame
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. step(dx, dz, dtheta)")
print("=" * 60)

_ = env.reset()

obs, state = env.step(dx=5.0, dz=0.0, dtheta=0.0)
print(f"  step(dx=+5 mm)       → s={state['s']:.4f}  t={state['t']:.4f}"
      f"  θ={np.degrees(state['angle']):.2f}°")

obs, state = env.step(dx=0.0, dz=3.0, dtheta=0.0)
print(f"  step(dz=+3 mm)       → s={state['s']:.4f}  t={state['t']:.4f}"
      f"  θ={np.degrees(state['angle']):.2f}°")

obs, state = env.step(dx=0.0, dz=0.0, dtheta=0.1)
print(f"  step(dtheta=+0.1 rad)→ s={state['s']:.4f}  t={state['t']:.4f}"
      f"  θ={np.degrees(state['angle']):.2f}°")

obs, state = env.step(dx=2.0, dz=-1.5, dtheta=-0.05)
print(f"  step(dx=2, dz=-1.5, dtheta=-0.05)"
      f" → s={state['s']:.4f}  t={state['t']:.4f}"
      f"  θ={np.degrees(state['angle']):.2f}°")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Random-policy rollout
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Random rollout  (20 steps)")
print("=" * 60)

obs = env.reset()
rng = np.random.default_rng(42)

for i in range(20):
    dx     = float(rng.uniform(-3.0,  3.0))
    dz     = float(rng.uniform(-3.0,  3.0))
    dtheta = float(rng.uniform(-0.1,  0.1))
    obs, state = env.step(dx, dz, dtheta)
    print(f"  {i:2d} | dx={dx:+5.2f}  dz={dz:+5.2f}  dθ={dtheta:+.3f}"
          f" → s={state['s']:.3f}  t={state['t']:.3f}"
          f"  θ={np.degrees(state['angle']):+6.1f}°"
          f"  obs_μ={obs.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Read probe frame vectors
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Probe frame vectors at current pose")
print("=" * 60)

origin_np, normal_np, ax_x_np, ax_z_np = env._frame()
print(f"  origin  (top-edge centre) : {origin_np}")
print(f"  normal  (outward, = ax_y) : {normal_np}")
print(f"  ax_x    (lateral)         : {ax_x_np}")
print(f"  ax_z    (elevational)     : {ax_z_np}  [ = normal × ax_x ]")
print(f"  orthogonality check:")
print(f"    ax_x · normal = {float(np.dot(ax_x_np, normal_np)):.2e}  (≈ 0)")
print(f"    ax_x · ax_z   = {float(np.dot(ax_x_np, ax_z_np)):.2e}  (≈ 0)")
print(f"    normal · ax_z = {float(np.dot(normal_np, ax_z_np)):.2e}  (≈ 0)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualise three slices side-by-side
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Matplotlib visualisation")
print("=" * 60)

poses = [
    dict(s=0.5, t=0.3, angle=0.0),
    dict(s=0.5, t=0.5, angle=0.0),
    dict(s=0.5, t=0.7, angle=0.0),
]

fig, axes = plt.subplots(1, len(poses), figsize=(4 * len(poses), 4))
fig.suptitle("ProbeEnv  –  slices at different t positions", fontsize=11)

for ax, pose in zip(axes, poses):
    obs = env.reset(**pose)
    ax.imshow(obs.numpy(), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"s={pose['s']}  t={pose['t']}  θ={np.degrees(pose['angle']):.0f}°",
                 fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()
print("Done.")