import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import h5py
import numpy as np
import torch.nn.functional as F
import pyvista as pv

from utils.plot_functions import (
    data_pairs_adjacent, transform_t2t, read_calib_matrices,
)
from utils.reconstruction import (_get_weight, get_slice)

DATA_DIR       = '/home/wu/Documents/projects/cloned_repositories/RecON/datasets'
FILENAME_CALIB = '/home/wu/Documents/projects/cloned_repositories/RecON/datasets/calib_matrix.csv'

# ── Load data ──────────────────────────────────────────────────────────────────
with h5py.File(os.path.join(DATA_DIR, "LH_Per_L_DtP.h5"), 'r') as f:
    example_scan           = f["frames"][()]   # (N, H, W) uint8
    example_transformation = f["tforms"][()]   # (N, 4, 4) float32

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

H_img, W_img = example_scan.shape[1], example_scan.shape[2]
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

########################################
### DRA in RecON Paper (Section 3.3) ###
########################################

def _dra_block(slices, inv_T, voxels, H, W, temperature=0.001, eps=1e-10):
    """Core DRA computation for a block of voxels (Eq. 6–9).

    For each voxel v_i in the volume and each frame I_j:
      - Project v_i into frame j's pixel space via inv(T_combined[j]).
      - d_{ij} = |z_perp|  (out-of-plane distance, Eq. 6).
      - Compute DRA weights W(d_{ij}) via differentiable maxk (Eq. 7–9).
      - Sample G_{ij} = bilinear pixel value at the projected (u, v).
      - G_{v_i} = Σ_j W(d_{ij}) · G_{ij}                         (Eq. 6).

    Parameters
    ----------
    slices  : (N, H, W)  – normalised US frames
    inv_T   : (N, 4, 4)  – inv(T_combined); maps world mm → pixel [u, v, z, 1]
    voxels  : (P, 4)     – homogeneous world mm voxel centres
    H, W    : frame height / width (pixels are 1-indexed: 1 … H, 1 … W)

    Returns
    -------
    gray : (P,) – reconstructed gray values for this block
    """
    N = slices.shape[0]
    P = voxels.shape[0]

    # Project voxels into each frame's local pixel coordinate system
    # loca[n, p] = inv_T[n] @ voxels[p] → [u, v, z_perp, 1]
    loca = torch.einsum('Nij,Pj->NPi', inv_T, voxels)  # (N, P, 4)

    # Out-of-plane distance d_{ij} = |z_perp|  (Eq. 6)
    dist = torch.abs(loca[..., 2])  # (N, P)

    # Exclude voxels that lie entirely beyond all frames in the z direction:
    # flag = True when every frame has z < -0.5 OR every frame has z > +0.5
    flag = (
        (torch.sum(~(loca[..., 2] < -0.5), dim=0) == 0) |
        (torch.sum(~(loca[..., 2] >  0.5), dim=0) == 0)
    )  # (P,) boolean

    # DRA weights: 1/(d+ε) with differentiable k-nearest softmax mask (Eq. 7-9)
    # _get_weight(dist, iter=2) implements the iterative maxk approximation of Eq. 9
    weight = _get_weight(dist, iter=2, temperature=temperature, eps=eps)  # (N, P)

    # Normalise 1-indexed pixel coordinates to grid_sample range [-1, 1].
    # align_corners=False convention: pixel centre u (1-indexed) → (2u − 1)/W − 1
    u_norm = (2.0 * loca[..., 0] - 1.0) / W - 1.0  # (N, P)  x / column direction
    v_norm = (2.0 * loca[..., 1] - 1.0) / H - 1.0  # (N, P)  y / row direction

    # Zero weights for out-of-image projections and z-excluded voxels
    out_of_bounds = (torch.abs(u_norm) > 1) | (torch.abs(v_norm) > 1)
    weight = weight.clone()
    weight[out_of_bounds] = 0.0
    weight[:, flag] = 0.0

    # Sample pixel intensities G_{ij} at (u, v) via bilinear interpolation.
    # grid_sample input: (N, 1, H, W); grid (N, 1, P, 2) with (x, y) = (u, v) order
    grid   = torch.stack([u_norm, v_norm], dim=-1).view(N, 1, P, 2)
    values = F.grid_sample(
        slices.unsqueeze(1),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False,
    ).view(N, P)  # (N, P)

    # Weighted sum: G_{v_i} = Σ_j W(d_{ij}) · G_{ij}  (Eq. 6)
    gray = (weight * values).sum(dim=0)  # (P,)
    return gray


def reconstruct_dra(images, T_combined, chunk_size=(30, 30, 30),
                    temperature=0.001, eps=1e-10):
    """Reconstruct a 3D volume from 2D US frames using DRA (Section 3.3).

    Eq. (5): The spatial position of each frame P_j is given directly by
    T_combined[j], which encodes P_j = P_1 · Π_{k=2}^{j} M(θ̂_{k-1}) as a
    single accumulated 4×4 homogeneous matrix.

    Parameters
    ----------
    images     : (N, H, W) ndarray uint8 – US frames
    T_combined : (N, 4, 4) tensor – pixel [u,v,0,1] → world mm (Eq. 5)
    chunk_size : (cx, cy, cz) – voxel chunk size for memory efficiency
    temperature: DRA softmax temperature (0.001 is optimal per Table 4)
    eps        : numerical stability constant

    Returns
    -------
    volume : (rX, rY, rZ) tensor – reconstructed 3D volume
    bias   : (3,) tensor         – world mm origin; voxel (0,0,0) = bias mm
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N, H, W = images.shape

    ##################################
    ### Normalise frames to [0, 1] ###
    ##################################
    slices = torch.from_numpy(images.astype(np.float32)).to(device)

    # f_min, f_max = slices.min(), slices.max()
    # if f_max > f_min:
    #     slices = (slices - f_min) / (f_max - f_min)

    source = slices / 255.0  # (N, H, W)

    T = T_combined.to(device)

    # ── Eq. (5): derive volume bounding box from frame corner positions ────────
    # Frame corners in 1-indexed homogeneous pixel coordinates
    corners_px = torch.tensor([
        [1.0,      1.0,      0.0, 1.0],  # top-left
        [float(W), 1.0,      0.0, 1.0],  # top-right
        [1.0,      float(H), 0.0, 1.0],  # bottom-left
        [float(W), float(H), 0.0, 1.0],  # bottom-right
    ], dtype=T.dtype, device=device).T  # (4, 4) – each column is one corner

    # Map all frame corners to world mm: T_combined @ corner → world mm
    world_corners = torch.bmm(T, corners_px.unsqueeze(0).expand(N, 4, 4))  # (N, 4, 4)
    world_xyz     = world_corners[:, :3, :].reshape(-1, 3)                  # (4N, 3)
    # world_xyz = world_corners[:, :3,
    # :].permute(0, 2, 1).reshape(-1, 3)

    min_mm    = world_xyz.min(dim=0)[0]               # (3,) world mm
    max_mm    = world_xyz.max(dim=0)[0]               # (3,)
    bias      = min_mm - 0.5                           # world mm at voxel index 0
    reco_size = torch.ceil(max_mm - min_mm + 1).long()
    rX, rY, rZ = reco_size.tolist()

    # ── Precompute inv(T_combined): world mm → pixel [u, v, z_perp, 1] ────────
    inv_T = torch.linalg.inv(T)  # (N, 4, 4)

    # ── Reconstruct volume in voxel chunks (memory-efficient) ─────────────────
    volume = torch.zeros(rX, rY, rZ, device=device)
    cx, cy, cz = chunk_size

    for iz in range(0, rZ, cz):
        ez = min(iz + cz, rZ)
        for iy in range(0, rY, cy):
            ey = min(iy + cy, rY)
            for ix in range(0, rX, cx):
                ex = min(ix + cx, rX)

                # Voxel centre world mm coordinates for this chunk
                xs = torch.arange(ix, ex, dtype=T.dtype, device=device) + 0.5 + bias[0]
                ys = torch.arange(iy, ey, dtype=T.dtype, device=device) + 0.5 + bias[1]
                zs = torch.arange(iz, ez, dtype=T.dtype, device=device) + 0.5 + bias[2]
                gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')

                voxels = torch.stack([
                    gx.reshape(-1),
                    gy.reshape(-1),
                    gz.reshape(-1),
                    torch.ones(gx.numel(), dtype=T.dtype, device=device),
                ], dim=-1)  # (P, 4) homogeneous world mm

                gray = _dra_block(slices, inv_T, voxels, H, W, temperature, eps)
                volume[ix:ex, iy:ey, iz:ez] = gray.view(ex - ix, ey - iy, ez - iz)

    return volume.cpu(), bias.cpu()


# ── Run DRA reconstruction ─────────────────────────────────────────────────────
volume, bias = reconstruct_dra(example_scan, T_combined)

slices = get_slice(volume, series, example_scan.shape[-2:])

# ── 3D volume rendering ────────────────────────────────────────────────────────
def visualize_3d_render(volume):
    vol_np = volume.numpy()
    grid = pv.ImageData()
    grid.dimensions = np.array(vol_np.shape)
    grid.spacing = (1, 1, 1)
    grid.point_data["Intensity"] = vol_np.flatten(order="F")

    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="Intensity", cmap="bone", opacity="sigmoid")
    plotter.show_axes()
    plotter.set_background("black")
    plotter.show()

visualize_3d_render(volume)