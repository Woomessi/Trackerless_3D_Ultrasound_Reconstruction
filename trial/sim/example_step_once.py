"""
trial/sim/example_step_once.py  –  minimal one-step example.

Input : action = (dx, dz, dtheta)
Output: updated state + visualised slice

Run from the RecON root:
    python trial/sim/example_step_once.py
"""

import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib.pyplot as plt
import rl_env
# from rl_env import ProbeEnv, probe_surface, volume_up, scale_h, scale_w, H_img, W_img
from rl_env import ProbeEnv, probe_surface, volume_up, scale_h, scale_w, H_img, W_img, plotter
# ── action ────────────────────────────────────────────────────────────────────
dx     =  5.0   # mm, lateral
dz     =  10.0   # mm, elevational
dtheta =  1   # rad, around normal

# ── environment ───────────────────────────────────────────────────────────────
# env = ProbeEnv(probe_surface, volume_up, scale_h, scale_w, H_img, W_img, plotter=None)
env = ProbeEnv(probe_surface, volume_up, scale_h, scale_w,
               H_img, W_img, plotter=plotter)
obs_before = env.reset()
state_before = dict(env.state)

obs_after, state_after = env.step(dx, dz, dtheta)

# ── print ─────────────────────────────────────────────────────────────────────
print(f"action : dx={dx}  dz={dz}  dtheta={dtheta}")
print(f"before : {state_before}")
print(f"after  : {state_after}")

# ── visualise ─────────────────────────────────────────────────────────────────
plt.tight_layout()

plotter.show()   # 阻塞，PyVista 事件循环同时保持 plt 窗口存活