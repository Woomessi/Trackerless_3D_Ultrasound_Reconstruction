"""Visualise the coordinate frames of T_1 and T_combined for every STEP-th frame.

T_combined[i]  :  pixel [u, v, 0, 1] → world mm
    origin  = T_combined[:, :3, 3]  (where pixel origin maps to)
    col 0   = u-axis (world mm per pixel in column direction)
    col 1   = v-axis (world mm per pixel in row direction)
    col 2   = normal (out-of-plane direction)

T_1[i]  =  image local frame (ax_x, ax_y, ax_z) + image-centre as origin
    (T_1 = inv(inv(axis)) = axis  — the raw local-frame matrix from get_axis)
    col 0   = ax_x  (lower-right − lower-left, normalised → rightward)
    col 1   = ax_y  (2·center − lower-right − lower-left, normalised → upward)
    col 2   = ax_z  = ax_x × ax_y  (outward normal)
    col 3   = image-centre world mm
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import h5py
import numpy as np
import pyvista as pv

from utils.plot_functions import (
    data_pairs_adjacent, transform_t2t, read_calib_matrices,
)
from utils.reconstruction import get_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = '/home/wu/Documents/projects/cloned_repositories/RecON/datasets'
FILENAME_CALIB = '/home/wu/Documents/projects/cloned_repositories/RecON/datasets/calib_matrix.csv'

# ── Load data ─────────────────────────────────────────────────────────────────
with h5py.File(os.path.join(DATA_DIR, "LH_Per_L_DtP.h5"), 'r') as f:
    example_scan           = f["frames"][()]   # (N, H, W) uint8
    example_transformation = f["tforms"][()]   # (N, 4, 4) float32

H_img, W_img = example_scan.shape[1], example_scan.shape[2]

# ── Build T_combined: pixel [u,v,0,1] → world mm ─────────────────────────────
data_pairs = data_pairs_adjacent(example_scan.shape[0])
tforms_each_frame2frame0 = transform_t2t(
    torch.from_numpy(example_transformation),
    torch.linalg.inv(torch.from_numpy(example_transformation)),
    data_pairs,
)
tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(FILENAME_CALIB)
tform_calib_inv_R_T = torch.linalg.inv(tform_calib_R_T)

T_combined = torch.matmul(
    tform_calib_inv_R_T.unsqueeze(0),
    torch.matmul(tforms_each_frame2frame0, tform_calib.unsqueeze(0)),
)  # (N, 4, 4)

# ── Build series & T_1 ────────────────────────────────────────────────────────
def T_combined_to_series(T_combined, H, W):
    dtype, device = T_combined.dtype, T_combined.device
    pixel_pts = torch.tensor([
        [W / 2.0,  H / 2.0,  0.0, 1.0],
        [1.0,      float(H), 0.0, 1.0],
        [float(W), float(H), 0.0, 1.0],
    ], dtype=dtype, device=device).T  # (4, 3)
    N = T_combined.shape[0]
    world_pts = torch.bmm(T_combined, pixel_pts.unsqueeze(0).expand(N, 4, 3))  # (N, 4, 3)
    return world_pts[:, :3, :].permute(0, 2, 1)  # (N, 3, 3)

series = T_combined_to_series(T_combined, H_img, W_img)

# T_1 = inv(inv(axis)) = axis  (get_matrix does axis → inv(axis); then we inv again)
T_1 = torch.inverse(get_matrix(series))  # (N, 4, 4)

# ── Subsample for a readable plot ─────────────────────────────────────────────
STEP     = 10      # show every N-th frame
AXIS_LEN = 5.0     # arrow length in world mm

idx      = np.arange(0, len(series), STEP)
Tc_np    = T_combined[idx].float().numpy()   # (M, 4, 4)
T1_np    = T_1[idx].float().numpy()          # (M, 4, 4)

# ── Helper: extract normalised axes and origins from a batch of 4x4 mats ─────
def extract_frame(mats, axis_len):
    """Return origins (M,3), ax0 (M,3), ax1 (M,3), ax2 (M,3) normalised to axis_len."""
    origins = mats[:, :3, 3]
    axes = []
    for col in range(3):
        v = mats[:, :3, col]
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12) * axis_len
        axes.append(v)
    return origins, axes[0], axes[1], axes[2]

Tc_orig, Tc_u, Tc_v, Tc_n = extract_frame(Tc_np, AXIS_LEN)
T1_orig, T1_x, T1_y, T1_z = extract_frame(T1_np, AXIS_LEN)

# ── Frame outlines from series points ─────────────────────────────────────────
series_np   = series[idx].float().numpy()        # (M, 3, 3)
centres     = series_np[:, 0, :]
lower_left  = series_np[:, 1, :]
lower_right = series_np[:, 2, :]

col_u   = (lower_right - lower_left) / (W_img - 1)
H_col_v = lower_left + lower_right - 2 * centres - col_u
upper_left  = lower_left  - (H_img - 1) / H_img * H_col_v
upper_right = lower_right - (H_img - 1) / H_img * H_col_v

rect_pts = np.stack(
    [upper_left, upper_right,
     upper_right, lower_right,
     lower_right, lower_left,
     lower_left,  upper_left],
    axis=1,
).reshape(-1, 3).astype(np.float32)

# ── Force equal axis scales ───────────────────────────────────────────────────
all_pts   = np.concatenate([centres, lower_left, lower_right, upper_left, upper_right])
mins, maxs = all_pts.min(0), all_pts.max(0)
ranges    = maxs - mins
max_range = ranges.max()
pad_lo    = mins - (max_range - ranges) / 2
pad_hi    = maxs + (max_range - ranges) / 2
cube_corners = np.array(np.meshgrid(
    [pad_lo[0], pad_hi[0]],
    [pad_lo[1], pad_hi[1]],
    [pad_lo[2], pad_hi[2]],
)).T.reshape(-1, 3).astype(np.float32)

# ── Plot ──────────────────────────────────────────────────────────────────────
pl = pv.Plotter(title='T_1 vs T_combined coordinate frames')

pl.add_points(cube_corners, opacity=0.0, point_size=1)   # invisible bounding box

# Frame outlines
pl.add_lines(rect_pts, color='white', width=1, label='frame boundary')

# --- T_combined axes (pixel-space directions in world) ---
# col 0 = u-axis (pixel column direction) — red
# col 1 = v-axis (pixel row direction)    — green
# col 2 = normal                          — blue
pl.add_arrows(Tc_orig, Tc_u, mag=1.0, color='red',   label='T_combined col-0 (u-axis)')
pl.add_arrows(Tc_orig, Tc_v, mag=1.0, color='green', label='T_combined col-1 (v-axis)')
pl.add_arrows(Tc_orig, Tc_n, mag=1.0, color='blue',  label='T_combined col-2 (normal)')

# --- T_1 axes (local image-plane frame) ---
# col 0 = ax_x (rightward, lower-right − lower-left) — magenta (dashed style via different color)
# col 1 = ax_y (upward,  2·center − lower-right − lower-left) — cyan
# col 2 = ax_z = ax_x × ax_y — yellow
pl.add_arrows(T1_orig, T1_x, mag=1.0, color='magenta', label='T_1 col-0 (ax_x, rightward)')
pl.add_arrows(T1_orig, T1_y, mag=1.0, color='cyan',    label='T_1 col-1 (ax_y, upward)')
pl.add_arrows(T1_orig, T1_z, mag=1.0, color='yellow',  label='T_1 col-2 (ax_z, normal)')

# Origins
pl.add_points(Tc_orig, color='white',  point_size=5, render_points_as_spheres=True,
              label='T_combined origin (pixel [0,0,0,1])')
pl.add_points(T1_orig, color='orange', point_size=5, render_points_as_spheres=True,
              label='T_1 origin (image centre)')

pl.add_legend(size=(0.35, 0.35))
pl.show_axes()
pl.set_background('black')
pl.show()