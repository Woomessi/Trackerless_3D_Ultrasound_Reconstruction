"""
trial/rl/animate_agent.py  –  Animate a saved Q-agent navigating the US probe.

For every step of a greedy episode the script:
  1. captures an off-screen PyVista screenshot (3-D volume + contact surface
     + moving probe plane, matching the keyboard_control.py scene)
  2. composites it with matplotlib side panels
     (current observation | target | NCC-over-steps curve)
  3. appends the result to an animated GIF.

Run from the RecON repository root:
    python trial/rl/animate_agent.py

Saves GIF via Pillow (PIL), which is already part of the environment.
"""

import os
import sys
import pickle
from collections import defaultdict

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
_here    = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(os.path.dirname(_here))
_env_dir = os.path.join(os.path.dirname(_here), "env")
for _p in (_root, _env_dir, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gym_env import ProbeNavEnv          # trial/rl/gym_env.py
from rl_env import (                     # trial/env/rl_env.py
    ProbeEnv,
    probe_surface, volume, volume_up,
    scale_h, scale_w, H_img, W_img,
    example_scan,
    # geometry needed to rebuild the static 3-D scene
    bias_np, contact_vox,
    arc_vox_left_top, arc_vox_right_top,
    corners_vox, N,
)
from q_learning import ProbeNavAgent, obs_key, _ACTION_LABELS


# ── off-screen 3-D scene ──────────────────────────────────────────────────────

def _make_plotter(window_size=(900, 600)):
    """Return an off-screen PyVista plotter with the static scene elements
    identical to what keyboard_control.py displays interactively."""
    plt3d = pv.Plotter(off_screen=True, window_size=window_size)
    plt3d.set_background("black")

    # Reconstructed 3-D volume
    vol_np = volume.numpy()
    grid   = pv.ImageData()
    grid.dimensions            = np.array(vol_np.shape)
    grid.spacing               = (1, 1, 1)
    grid.point_data["value"]   = vol_np.flatten(order="F")
    plt3d.add_volume(grid, scalars="value", cmap="bone", opacity="sigmoid")

    # Fitted contact surface
    _ns, _nt = 40, 120
    _SS, _TT = np.meshgrid(np.linspace(0, 1, _ns), np.linspace(0, 1, _nt))
    _pts     = probe_surface.evaluate(_SS, _TT)
    surf     = pv.StructuredGrid(_pts[..., 0], _pts[..., 1], _pts[..., 2])
    plt3d.add_mesh(surf, color="cyan", opacity=0.25, show_edges=False)

    # Contact-point trajectory (magenta polyline)
    traj        = pv.PolyData()
    traj.points = contact_vox
    traj.lines  = np.hstack([[N], np.arange(N)])
    plt3d.add_mesh(traj, color="magenta", line_width=2)

    # Arc top-boundary points (deepskyblue = left, gold = right)
    valid = ~np.isnan(arc_vox_left_top[:, 0])
    plt3d.add_points(arc_vox_left_top[valid],  color="deepskyblue",
                     point_size=6, render_points_as_spheres=True)
    plt3d.add_points(arc_vox_right_top[valid], color="gold",
                     point_size=6, render_points_as_spheres=True)

    # First and last frame: yellow outline + textured US image
    _uv = np.array([[0., 1.], [1., 1.], [1., 0.], [0., 0.]], dtype=np.float32)
    for i in [0, N - 1]:
        pts     = corners_vox[i]
        outline = pv.PolyData()
        outline.points = pts
        outline.lines  = np.array([2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0])
        plt3d.add_mesh(outline, color="yellow", line_width=2, opacity=0.9)

        img_u8 = example_scan[i]
        tex    = pv.Texture(np.stack([img_u8, img_u8, img_u8], axis=-1))
        quad   = pv.PolyData()
        quad.points = pts
        quad.faces  = np.array([4, 0, 1, 2, 3])
        quad.active_texture_coordinates = _uv
        plt3d.add_mesh(quad, texture=tex, opacity=0.85)

    plt3d.show_axes()
    plt3d.reset_camera()
    return plt3d


# ── single-frame composition ──────────────────────────────────────────────────

def _compose_frame(pv_img, obs, target_img, info_hist, step, action=None):
    """Composite one animation frame from a PyVista screenshot + side panels.

    Returns a (H, W, 3) uint8 numpy array.
    """
    nccs     = [d["ncc"] for d in info_hist]
    cur_info = info_hist[-1]

    fig = plt.figure(figsize=(13, 5), dpi=100)
    fig.patch.set_facecolor("black")

    # ── left 57%: 3-D view ────────────────────────────────────────────────────
    ax3d = fig.add_axes([0.0, 0.0, 0.57, 1.0])
    ax3d.imshow(pv_img)
    ax3d.axis("off")
    ax3d.set_facecolor("black")
    act_str = _ACTION_LABELS[action] if action is not None else "reset"
    ax3d.set_title(
        f"step {step:3d}  action: {act_str:>2s}   "
        f"s={cur_info['s']:.3f}  t={cur_info['t']:.3f}  "
        f"θ={np.degrees(cur_info['angle']):.1f}°   "
        f"NCC={cur_info['ncc']:.3f}",
        color="white", fontsize=8, pad=3,
    )

    # ── top-right: current obs | target ───────────────────────────────────────
    ax_obs = fig.add_axes([0.59, 0.52, 0.19, 0.43])
    ax_obs.imshow(obs, cmap="gray", vmin=0, vmax=1)
    ax_obs.set_title("Current obs", fontsize=8, color="white")
    ax_obs.axis("off")

    ax_tgt = fig.add_axes([0.80, 0.52, 0.19, 0.43])
    ax_tgt.imshow(target_img, cmap="gray", vmin=0, vmax=1)
    ax_tgt.set_title("Target", fontsize=8, color="white")
    ax_tgt.axis("off")

    # ── bottom-right: NCC curve ────────────────────────────────────────────────
    ax_ncc = fig.add_axes([0.59, 0.06, 0.39, 0.38])
    ax_ncc.set_facecolor("#111111")
    ax_ncc.plot(nccs, color="steelblue", lw=1.5)
    ax_ncc.axhline(0.95, color="red", lw=1, ls="--", label="0.95 threshold")
    ax_ncc.set_xlim(0, max(len(nccs) - 1, 10))
    ax_ncc.set_ylim(0, 1.05)
    ax_ncc.set_xlabel("Step",  fontsize=7, color="white")
    ax_ncc.set_ylabel("NCC",   fontsize=7, color="white")
    ax_ncc.tick_params(colors="white", labelsize=6)
    for sp in ax_ncc.spines.values():
        sp.set_color("gray")
    ax_ncc.set_title("NCC over steps", fontsize=8, color="white")
    ax_ncc.legend(fontsize=6, facecolor="#222222", labelcolor="white",
                  edgecolor="gray", loc="upper left")

    fig.canvas.draw()
    w, h  = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return frame


# ── main animation function ───────────────────────────────────────────────────

def animate_navigation(
    qt_path: str  = None,
    save_path: str = None,
    fps: int       = 10,
    stride: int    = 1,
    camera_position = None,
):
    """Load q_table.pkl, run one greedy episode, save 3-D navigation as GIF.

    Parameters
    ----------
    qt_path         : path to saved Q-table pickle (default: q_table.pkl here)
    save_path       : output file path; .gif or .mp4 (default: navigation_animation.gif)
    fps             : frames per second in the output
    stride          : record every `stride`-th step (1 = every step)
    camera_position : PyVista camera_position override (None = auto)
    """
    from PIL import Image

    if qt_path is None:
        qt_path = os.path.join(_here, "q_table.pkl")
    if save_path is None:
        save_path = os.path.join(_here, "navigation_animation.gif")

    with open(qt_path, "rb") as f:
        q_dict = pickle.load(f)
    print(f"Loaded Q-table: {len(q_dict)} states  ←  {qt_path}")

    # ── build scene & environments ────────────────────────────────────────────
    print("Building off-screen 3-D scene…")
    plt3d = _make_plotter()
    if camera_position is not None:
        plt3d.camera_position = camera_position

    base_env = ProbeEnv(
        probe_surface, volume_up, scale_h, scale_w, H_img, W_img,
        plotter=plt3d, axis_len=50.0,
    )
    target_img = example_scan[566].astype(np.float32) / 255.0
    raw_env    = ProbeNavEnv(
        base_env, target_img=target_img, max_steps=200, success_threshold=0.95,
    )

    agent = ProbeNavAgent(
        env=raw_env, learning_rate=0.0, initial_epsilon=0.0,
        epsilon_decay=0.0, final_epsilon=0.0,
    )
    agent.q_values = defaultdict(lambda: np.zeros(raw_env.action_space.n), q_dict)

    # ── episode loop (capture after each render) ──────────────────────────────
    obs, info = raw_env.reset()
    state     = obs_key(info)
    info_hist = [info]
    frames    = []

    # initial frame (after reset, probe is already rendered in plt3d)
    frames.append(_compose_frame(plt3d.screenshot(), obs, target_img,
                                 info_hist, step=0, action=None))

    done   = False
    step   = 0
    while not done:
        action = agent.get_action(state)
        obs, _r, terminated, truncated, info = raw_env.step(action)
        state = obs_key(info)
        info_hist.append(info)
        step += 1
        done = terminated or truncated

        if step % stride == 0 or done:
            frames.append(_compose_frame(plt3d.screenshot(), obs, target_img,
                                         info_hist, step=step, action=action))

    final_ncc = info_hist[-1]["ncc"]
    status    = "Success" if final_ncc >= 0.95 else f"NCC={final_ncc:.3f}"
    print(f"Episode: {step} steps, {status}")
    print(f"Saving {len(frames)} frames to {save_path} …")

    pil_frames = [Image.fromarray(f) for f in frames]
    duration_ms = 1000 // fps
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
    )

    print(f"Saved  →  {save_path}")


if __name__ == "__main__":
    animate_navigation()
