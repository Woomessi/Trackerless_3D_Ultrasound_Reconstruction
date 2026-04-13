###################
### preparation ###
###################

import sys
import os
import torch
import h5py
import numpy as np
import pyvista as pv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import trial.my_utils.functions as my_utils

from utils.plot_functions import (
    data_pairs_adjacent, transform_t2t, read_calib_matrices, add_series_rects,
)
from utils.simulation import dof_to_series
from utils.reconstruction import reco, get_slice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR       = '/home/wu/Documents/projects/cloned_repositories/RecON/datasets'
FILENAME_CALIB = '/home/wu/Documents/projects/cloned_repositories/RecON/datasets/calib_matrix.csv'

#################
### Load data ###
#################

with h5py.File(os.path.join(DATA_DIR, "LH_Per_L_DtP.h5"), 'r') as f:
            source = f["frames"][()]   # (N, H, W) uint8
            T = f["tforms"][()]   # (N, 4, 4) float32

H, W = source.shape[1], source.shape[2]

######################################################
### Compute T_combined: pixel [u,v,0,1] → world mm ###
######################################################

data_pairs = data_pairs_adjacent(source.shape[0])
T_each_frame2frame0 = transform_t2t(
    torch.from_numpy(T),
    torch.linalg.inv(torch.from_numpy(T)),
    data_pairs,
)

T_calib_scale, T_calib_R_T, T_calib = read_calib_matrices(FILENAME_CALIB)
T_calib_inv_R_T = torch.linalg.inv(T_calib_R_T)

scale_w = float(T_calib_scale[0, 0])  # u → W dimension
scale_h = float(T_calib_scale[1, 1])  # v → H dimension

T_combined = torch.matmul(
    T_calib_inv_R_T.unsqueeze(0),
    torch.matmul(T_each_frame2frame0, T_calib.unsqueeze(0)),
)  # (N, 4, 4)

#############################################################################
### Build series (N, 3, 3): [centre, lower-left, lower-right] in world mm ###
#############################################################################

pixel_pts = torch.tensor(
    [
        [W / 2.0,  H / 2.0, 0.0, 1.0],   # centre
        [1.0,          float(H), 0.0, 1.0],   # lower-left
        [float(W), float(H), 0.0, 1.0],   # lower-right
    ],
    dtype=T_combined.dtype,
).T  # (4, 3)

N = T_combined.shape[0]
world_pts = torch.bmm(T_combined, pixel_pts.unsqueeze(0).expand(N, 4, 3))  # (N, 4, 3)
series = world_pts[:, :3, :].permute(0, 2, 1)  # (N, 3, 3)

##################
### Downsample ###
##################

source = torch.from_numpy(source.astype(np.float32) / 255.0)  # (N, H, W)

down_ratio = 0.5

mat_scale = torch.eye(4, dtype=torch.float32, device=DEVICE)
mat_scale[0, 0] = down_ratio
mat_scale[1, 1] = down_ratio
mat_scale[2, 2] = down_ratio

source_down = F.interpolate(
    source.unsqueeze(1), scale_factor=down_ratio, mode='bilinear', align_corners=False
).squeeze(1)  # (N, H*dr, W*dr)

##############################
### Reconstruct 3-D volume ###
##############################

volume, bias = my_utils.reco(source_down.to(DEVICE), series.to(DEVICE), scale_w, scale_h, mat_scale)

volume = volume.cpu()
bias = bias.cpu()

series_biased = series - bias
volume_up = F.interpolate(volume.unsqueeze(0).unsqueeze(0), scale_factor=1 / down_ratio).squeeze(0).squeeze(0)

########################
### Slice 3-D volume ###
########################
idx_start_point = 300
start_point = series[idx_start_point].unsqueeze(0)
dof = torch.tensor([
    [
        [0., 0., 0., 1., 0., 0.]
    ]
])

slicer = dof_to_series(start_point, dof)
slicer = slicer.squeeze(0)

slicer_biased = slicer[1].unsqueeze(0) - bias.unsqueeze(0).unsqueeze(0)

slice = my_utils.get_slice(volume_up, slicer_biased, source.shape[-2:], scale_h = scale_h, scale_w = scale_w)  # (1, N, 1, H_down, W_down)
slice = slice.squeeze(0,1,2)

########################
### Visualise slice ###
########################

fig, axes = plt.subplots(2, 1, figsize=(3, 6), squeeze=False)
fig.suptitle("Top: original frame   |   Bottom: slice from volume")

orig = source[idx_start_point]
axes[0, 0].imshow(orig, cmap="gray", vmin=0, vmax=1)
# axes[0, 0].set_title(f"frame {1}")
axes[0, 0].axis("off")

rec_slice = slice
axes[1, 0].imshow(rec_slice, cmap="gray", vmin=0, vmax=1)
# axes[1, 0].set_title(f"slice {1}")
axes[1, 0].axis("off")

plt.tight_layout()
plt.show()

########################################
### 3D volume rendering with PyVista ###
########################################

plotter = pv.Plotter(title='Series rectangles')

vol_np = volume_up.numpy()
grid = pv.ImageData()
grid.dimensions = np.array(vol_np.shape)
grid.spacing = (1, 1, 1)
grid.point_data["Intensity"] = vol_np.flatten(order="F")
plotter.add_volume(grid, scalars="Intensity", cmap="bone", opacity="sigmoid")

add_series_rects(plotter, series_biased, indices=[0,300,N-1], colors='red', opacity=1, frames=source)
add_series_rects(plotter, slicer_biased, indices=[0], colors='blue', opacity=1)

plotter.show_axes()
plotter.set_background('black')
plotter.show()