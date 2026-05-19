"""
trial/env/keyboard_control.py  –  Interactive keyboard control of US probe.

Run from the RecON repository root:
    python trial/env/keyboard_control.py

Keyboard controls (3D window must be focused):
  →  / ←   lateral  ±2 mm  (±x)
  ↑  / ↓   elevational ±2 mm  (±z)
  z  / x   rotate CCW / CW  ±2°
  r        reset to (s=0.5, t=0.5, θ=0°)
  q        quit (close 3D window)

Displays:
  3D window  –  reconstructed volume · fitted contact surface · probe pose
  2D window  –  current observation (left)  |  target at t=1 (right)
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib.pyplot as plt

from rl_env import (
    ProbeEnv,
    probe_surface, volume_up, plotter,
    scale_h, scale_w, H_img, W_img,
    example_scan,
)

# ── Step sizes ─────────────────────────────────────────────────────────────────
_DX = 2.0                        # mm per lateral keypress
_DZ = 2.0                        # mm per elevational keypress
_DA = float(np.radians(2.0))     # rad per rotation keypress

# ── Low-level environment (attaches probe rendering to the pre-built plotter) ──
env = ProbeEnv(
    probe_surface, volume_up, scale_h, scale_w, H_img, W_img,
    plotter=plotter,
    axis_len=50.0,
)

# ── Target observation: last acquired raw frame example_scan[566] ─────────────
_target_obs = example_scan[566].astype(np.float32) / 255.0

# ── Replace the auto-created single-panel figure with a 2-panel comparison ─────
plt.close(env._fig2d)
plt.ion()

_fig, (_ax_cur, _ax_tgt) = plt.subplots(1, 2, figsize=(10, 4))
_fig.suptitle("Observation", fontsize=10)

_hdl_cur = _ax_cur.imshow(
    np.zeros((H_img, W_img), dtype=np.float32), cmap="gray", vmin=0, vmax=1,
)
_ax_cur.axis("off")

_ax_tgt.imshow(_target_obs, cmap="gray", vmin=0, vmax=1)
_ax_tgt.set_title("Target  (s=0.5  t=1.0  θ=0°)", fontsize=8)
_ax_tgt.axis("off")
_fig.tight_layout(rect=[0, 0, 1, 0.92])
_fig.canvas.draw()

# Redirect ProbeEnv's 2-D rendering to the new figure
env._fig2d   = _fig
env._ax2d    = _ax_cur
env._img_hdl = _hdl_cur

# ── NCC with target ────────────────────────────────────────────────────────────
def _ncc(obs_np):
    a = obs_np.ravel().astype(np.float64);  a -= a.mean()
    b = _target_obs.ravel().astype(np.float64); b -= b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float((np.dot(a, b) / (na * nb) + 1.0) * 0.5)

# ── HUD text overlay in the 3D window ─────────────────────────────────────────
_hud = [None]

def _update_hud(obs_np=None):
    if _hud[0] is not None:
        try:
            plotter.remove_actor(_hud[0])
        except Exception:
            pass
        _hud[0] = None
    ncc_str = f"  NCC={_ncc(obs_np):.3f}" if obs_np is not None else ""
    _hud[0] = plotter.add_text(
        f"s={env.s:.3f}  t={env.t:.3f}  θ={np.degrees(env.angle):.1f}°{ncc_str}\n"
        "→/← ±x   ↑/↓ ±z   z/x ±θ   r reset   q quit",
        position="lower_left", font_size=9, color="white",
    )

# ── Action helpers ─────────────────────────────────────────────────────────────
def _step(dx=0.0, dz=0.0, dtheta=0.0):
    obs_t, _ = env.step(dx, dz, dtheta)
    _update_hud(obs_t.numpy())

def _reset():
    obs_t = env.reset(s=0.5, t=0.5, angle=0.0)
    _update_hud(obs_t.numpy())

# ── Key bindings ───────────────────────────────────────────────────────────────
plotter.add_key_event("Right", lambda: _step(dx=+_DX))
plotter.add_key_event("Left",  lambda: _step(dx=-_DX))
plotter.add_key_event("Up",    lambda: _step(dz=+_DZ))
plotter.add_key_event("Down",  lambda: _step(dz=-_DZ))
plotter.add_key_event("z",     lambda: _step(dtheta=+_DA))
plotter.add_key_event("x",     lambda: _step(dtheta=-_DA))
plotter.add_key_event("r",     _reset)

# ── Initial render ─────────────────────────────────────────────────────────────
_reset()

# ── Launch ─────────────────────────────────────────────────────────────────────
plotter.show_axes()
plotter.set_background("black")
print(
    "\nKeyboard controls (3D window must be focused):\n"
    "  →/←   lateral ±2 mm        ↑/↓  elevational ±2 mm\n"
    "  z/x   rotate ±2°           r    reset        q quit\n"
)
plotter.show()