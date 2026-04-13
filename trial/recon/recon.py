import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import h5py
import numpy as np
import torch.nn.functional as F
import pyvista as pv
import trial.my_utils.functions as my_utils
from utils.plot_functions import (
    data_pairs_adjacent, transform_t2t, read_calib_matrices,
)
from utils.reconstruction import reco

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR       = '/home/wu/Documents/projects/cloned_repositories/RecON/datasets'
FILENAME_CALIB = '/home/wu/Documents/projects/cloned_repositories/RecON/datasets/calib_matrix.csv'

# ── Load data ──────────────────────────────────────────────────────────────────
with h5py.File(os.path.join(DATA_DIR, "LH_Per_L_DtP.h5"), 'r') as f:
    example_scan           = f["frames"][()]   # (N, H, W) uint8
    example_transformation = f["tforms"][()]   # (N, 4, 4) float32

H_img, W_img = example_scan.shape[1], example_scan.shape[2]

# ── Compute T_combined: pixel [u,v,0,1] → world mm ────────────────────────────
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

# ── Build series (N, 3, 3): [centre, lower-left, lower-right] in world mm ─────
# T_combined[i]: pixel [u, v, 0, 1] → world mm
#   centre      pixel: (W/2, H/2, 0, 1)
#   lower-left  pixel: (1,   H,   0, 1)
#   lower-right pixel: (W,   H,   0, 1)
# Note: series is in world mm.  source is downsampled so that 1 pixel ≈ 1 mm,
# making reco() consistent with the mm-unit series without any mat_scale.
#
pixel_pts = torch.tensor(
    [
        [W_img / 2.0,  H_img / 2.0, 0.0, 1.0],   # centre
        [1.0,          float(H_img), 0.0, 1.0],   # lower-left
        [float(W_img), float(H_img), 0.0, 1.0],   # lower-right
    ],
    dtype=T_combined.dtype,
).T  # (4, 3)

N = T_combined.shape[0]
world_pts = torch.bmm(T_combined, pixel_pts.unsqueeze(0).expand(N, 4, 3))  # (N, 4, 3)
series = world_pts[:, :3, :].permute(0, 2, 1)  # (N, 3, 3)

# ── Prepare source frames: (N, H, W) float32 in [0, 1] ────────────────────────
source = torch.from_numpy(example_scan.astype(np.float32) / 255.0)  # (N, H, W)

# ── Downsample ──────────────────────────────────

down_ratio = 1
mat_scale = torch.eye(4, dtype=torch.float32, device=DEVICE)
mat_scale[0, 0] = down_ratio
mat_scale[1, 1] = down_ratio
mat_scale[2, 2] = down_ratio

source_down = F.interpolate(
    source.unsqueeze(1), scale_factor=down_ratio, mode='bilinear', align_corners=False
).squeeze(1)  # (N, H*dr, W*dr)

scale_w = float(tform_calib_scale[0, 0])  # u → W dimension
scale_h = float(tform_calib_scale[1, 1])  # v → H dimension

# ── Reconstruct 3-D volume ─────────────────────────────────────────────────────

volume, bias = my_utils.reco(source_down.to(DEVICE), series.to(DEVICE), scale_w, scale_h, mat_scale)
# volume, bias = my_utils.reco(source.to(DEVICE), series.to(DEVICE), scale_w, scale_h)

volume = volume.cpu()

# ── 3D volume rendering with PyVista ──────────────────────────────────────────
vol_np = volume.numpy()
grid = pv.ImageData()
grid.dimensions = np.array(vol_np.shape)
grid.spacing = (1, 1, 1)
grid.point_data["Intensity"] = vol_np.flatten(order="F")

plotter = pv.Plotter(title="3D US reconstruction")
plotter.add_volume(grid, scalars="Intensity", cmap="bone", opacity="sigmoid")
plotter.show_axes()
plotter.set_background("black")
plotter.show()