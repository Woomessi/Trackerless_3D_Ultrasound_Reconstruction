"""
Inference + 3D reconstruction script.

Based on trial/inference/infer_baseline_reconstruction.py.
Reconstruction method taken from trial/recon/recon.py
(underlying implementation: trial/my_utils/functions.py).

Pipeline
--------
1. Load h5 scan and calibration.
2. Run Online_Baseline_Backbone to predict 6-DoF gaps.
3. Integrate gaps → predicted series (N, 3, 3) world-mm frame positions.
4. Reconstruct 3D US volume for predicted series.
5. Visualise: orthogonal-slice PNG + interactive PyVista.

Outputs (infer_baseline_output/)
---------------------------------
  volume_pred.npy          (X, Y, Z) float32 predicted volume
  slices_pred.png          axial / sagittal / coronal middle slices
"""

import sys
import os

# Project root → needed for "trial.my_utils.functions"
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

from trial.my_utils.functions import reco as my_reco, get_slice as my_get_slice
from utils.plot_functions import add_series_rects

# ── Paths ─────────────────────────────────────────────────────────────────────
H5_PATH    = "../../data/frames_transfs/001/LH_Per_L_DtP.h5"
CALIB_PATH = "../../data/calib_matrix.csv"
CKPT_PATH  = "../../save/online_baseline_bk-hp_bk-TUS_complete/online_baseline_bk_backbone_230.pth"
OUT_DIR    = "../../infer_baseline_output"

# ── Model config ──────────────────────────────────────────────────────────────
IN_PLANES   = 2
NUM_CLASSES = 6
IMG_H, IMG_W = 480, 640
BATCH_PAIRS  = 1

# ── Cross-section config ───────────────────────────────────────────────────────
FRAME_IDX = 0   # which frame to use as the reference plane

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────────────────────────────────────────
import timm

class Backbone(nn.Module):
    def __init__(self, in_planes, num_classes):
        super().__init__()
        self.efficientnet_b1 = timm.create_model(
            'efficientnet_b1', pretrained=False,
            in_chans=in_planes, num_classes=num_classes,
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = (x - torch.mean(x, dim=[3, 4], keepdim=True)) / \
            (torch.std(x, dim=[3, 4], keepdim=True) + 1e-6)
        x = x.view(b * t, c, h, w)
        x = self.efficientnet_b1(x)
        x = x.view(b, t, *x.shape[1:])
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_calib_matrices(filename_calib):
    tform_calib = np.empty((8, 4), np.float32)
    with open(filename_calib, 'r') as csv_file:
        txt = [i.strip('\n').split(',') for i in csv_file.readlines()]
        tform_calib[0:4, :] = np.array(txt[1:5]).astype(np.float32)
        tform_calib[4:8, :] = np.array(txt[6:10]).astype(np.float32)
    calib_scale = torch.tensor(tform_calib[0:4, :])
    calib_R_T   = torch.tensor(tform_calib[4:8, :])
    calib       = torch.tensor(tform_calib[4:8, :] @ tform_calib[0:4, :])
    return calib_scale, calib_R_T, calib


def build_series(tforms_np, calib_path, H_out, W_out):
    tforms = torch.from_numpy(tforms_np)
    N = tforms.shape[0]
    pairs = torch.tensor([[0, n] for n in range(N)])
    tforms_inv = torch.linalg.inv(tforms)
    tforms_world_to_f0 = tforms_inv[pairs[:, 0]]
    tforms_fn_to_world  = tforms[pairs[:, 1]]
    tforms_f2f0 = torch.matmul(tforms_world_to_f0, tforms_fn_to_world)

    _, calib_R_T, calib = read_calib_matrices(calib_path)
    T_combined = torch.matmul(
        torch.linalg.inv(calib_R_T).unsqueeze(0),
        torch.matmul(tforms_f2f0, calib.unsqueeze(0)),
    )

    pixel_pts = torch.tensor(
        [[W_out / 2.0, H_out / 2.0, 0.0, 1.0],
         [1.0,          float(H_out), 0.0, 1.0],
         [float(W_out), float(H_out), 0.0, 1.0]],
        dtype=T_combined.dtype,
    ).T  # (4, 3)

    world_pts = torch.bmm(T_combined, pixel_pts.unsqueeze(0).expand(N, 4, 3))
    return world_pts[:, :3, :].permute(0, 2, 1)   # (N, 3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# dof_to_series
# ─────────────────────────────────────────────────────────────────────────────
def get_axis(series):
    v1 = series[:, 2, :] - series[:, 1, :]
    v2 = series[:, 0, :] - series[:, 1, :]
    v3 = torch.cross(v1, v2, dim=1)
    axis = torch.stack([v1, v2, v3], dim=1)
    norms = torch.norm(axis, dim=2, keepdim=True).clamp(min=1e-8)
    return axis / norms


def euler_matrix_batch(angles):
    ai, aj, ak = angles[:, 0], angles[:, 1], angles[:, 2]
    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    B = angles.shape[0]
    M = torch.zeros(B, 4, 4, dtype=angles.dtype, device=angles.device)
    M[:, 0, 0] = cj * ck;  M[:, 0, 1] = sj * sc - cs;  M[:, 0, 2] = sj * cc + ss
    M[:, 1, 0] = cj * sk;  M[:, 1, 1] = sj * ss + cc;  M[:, 1, 2] = sj * cs - sc
    M[:, 2, 0] = -sj;      M[:, 2, 1] = cj * si;        M[:, 2, 2] = cj * ci
    M[:, 3, 3] = 1.0
    return M


def dof_to_series(start_point, dof):
    old_type = start_point.dtype
    start_point = start_point.double()
    dof = dof.double()
    b, t, _ = dof.shape
    dof_flat = dof.view(b * t, -1)
    matrix = euler_matrix_batch(dof_flat[:, 3:])
    matrix[:, :3, 3] = dof_flat[:, :3]
    matrix = matrix.view(b, t, 4, 4)

    start_axis = get_axis(start_point).permute(0, 2, 1)
    start_matrix = torch.cat(
        [start_axis, start_point[:, 0, :].unsqueeze(-1)], dim=-1
    )
    start_matrix = F.pad(start_matrix, (0, 0, 0, 1))
    start_matrix[:, 3, 3] = 1.0
    start_matrix_inv = torch.linalg.inv(start_matrix)

    matrix_chain = [start_matrix]
    for idx in range(matrix.shape[1]):
        matrix_chain.append(torch.bmm(matrix_chain[-1], matrix[:, idx]))
    matrix_chain = torch.stack(matrix_chain, dim=1)

    start_point_4d = F.pad(start_point, (0, 1))
    start_point_4d[:, :, 3] = 1.0
    series = torch.einsum(
        'btij,bjk,bkl->btil',
        matrix_chain,
        start_matrix_inv,
        start_point_4d.permute(0, 2, 1),
    ).permute(0, 1, 3, 2)[..., :3]

    return series.squeeze(0).to(old_type)


# ─────────────────────────────────────────────────────────────────────────────
# 3D reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_volume(source, series, scale_w, scale_h, label=""):
    """Reconstruct a 3D volume from US frames and their world-mm positions.

    Parameters
    ----------
    source   : (N, H, W) float32 tensor, pixel intensities in [0, 1]
    series   : (N, 3, 3) float32 tensor, world-mm frame positions
    scale_w  : float, mm per pixel in u direction (calib_scale[0,0])
    scale_h  : float, mm per pixel in v direction (calib_scale[1,1])

    Returns
    -------
    volume : (X, Y, Z) float32 tensor on CPU
    bias   : (3,) float32 tensor — world-mm offset of volume origin
    """
    print(f"  Reconstructing {label} volume ...")
    mat_scale = torch.eye(4, dtype=torch.float32, device=device)
    vol, bias = my_reco(
        source.to(device),
        series.to(device).float(),
        scale_w, scale_h,
        mat_scale,
    )
    vol  = vol.cpu()
    bias = bias.cpu()
    print(f"  {label} volume shape: {tuple(vol.shape)}  "
          f"(bias={bias.tolist()})")
    return vol, bias


# ─────────────────────────────────────────────────────────────────────────────
# Slice-view visualisation (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def save_slice_views(volume_np, title, out_path):
    """Save axial / sagittal / coronal middle slices to a PNG."""
    x, y, z = volume_np.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [
        (volume_np[x // 2, :, :], f"Axial  (x={x//2})"),
        (volume_np[:, y // 2, :], f"Sagittal (y={y//2})"),
        (volume_np[:, :, z // 2], f"Coronal  (z={z//2})"),
    ]
    for ax, (sl, label) in zip(axes, slices):
        ax.imshow(sl.T, origin='lower', cmap='bone',
                  vmin=0, vmax=volume_np.max() or 1)
        ax.set_title(label, fontsize=10)
        ax.axis('off')
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PyVista interactive visualisation
# ─────────────────────────────────────────────────────────────────────────────
def pyvista_visualize(volume_pred_np, out_dir):
    """Volume rendering of predicted reconstruction; saves screenshot."""
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not available — skipping interactive visualisation.")
        return

    def _make_grid(vol_np):
        grid = pv.ImageData()
        grid.dimensions = np.array(vol_np.shape) + 1   # cell-centred
        grid.spacing = (1, 1, 1)
        grid.cell_data["Intensity"] = vol_np.flatten(order="F")
        return grid

    grid_pred = _make_grid(volume_pred_np)

    pl = pv.Plotter(title="3D US Reconstruction — Predicted")
    pl.add_volume(grid_pred, scalars="Intensity", cmap="bone", opacity="sigmoid")
    pl.add_text("Predicted reconstruction", position="upper_edge", font_size=12)
    pl.show_axes()
    pl.set_background("black")

    screenshot_path = os.path.join(out_dir, "volume_pred.png")
    pl.show(screenshot=screenshot_path)
    print(f"Saved: {screenshot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-section extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_cross_slices(volume, bias, series_world, frame_idx,
                         H, W, scale_h, scale_w):
    """Interactively visualise two perpendicular planes cut from a 3D volume.

    Opens a PyVista window showing:
      - The reconstructed 3D volume (bone colormap)
      - Original scan plane of frame `frame_idx` (red outline, US image texture)
      - Perpendicular plane through same centre / horizontal axis (blue outline,
        extracted slice texture); obtained by rotating 90° around ax_x so that
        the original plane's normal becomes the new vertical axis.

    Parameters
    ----------
    volume       : (X, Y, Z) float32 tensor (CPU)
    bias         : (3,) float32 tensor — world-mm offset of volume origin
    series_world : (N, 3, 3) float32 tensor — true world-mm positions
    frame_idx    : int — which frame to use as the reference plane
    H, W         : int — slice resolution in pixels
    scale_h      : float — mm / pixel in row direction
    scale_w      : float — mm / pixel in column direction
    """
    import pyvista as pv

    # ── Retrieve frame position in world-mm ───────────────────────────────────
    frame_w = series_world[frame_idx:frame_idx + 1].float()   # (1, 3, 3)
    center = frame_w[0, 0]
    ll     = frame_w[0, 1]
    lr     = frame_w[0, 2]
    ul     = 2 * center - lr

    # ── Orthonormal frame axes (same convention as get_axis in functions.py) ──
    ax_x = F.normalize((lr - ll).unsqueeze(0), dim=-1).squeeze(0)
    ax_y = F.normalize((ul - ll).unsqueeze(0), dim=-1).squeeze(0)
    ax_z = torch.cross(ax_x, ax_y, dim=0)

    # ── Physical half-extents (mm) ────────────────────────────────────────────
    half_w = torch.norm(lr - ll) / 2.0
    half_h = torch.norm(ul - ll) / 2.0

    # ── Perpendicular plane: rotate 90° around ax_x (ax_y → ax_z) ───────────
    ll_perp = center - half_w * ax_x - half_h * ax_z
    lr_perp = center + half_w * ax_x - half_h * ax_z
    perp_w  = torch.stack([center, ll_perp, lr_perp], dim=0).unsqueeze(0)  # (1, 3, 3)

    # ── Shift to volume coordinates (world_mm − bias) ─────────────────────────
    bias_t  = bias.float()
    frame_v = frame_w - bias_t   # (1, 3, 3)
    perp_v  = perp_w  - bias_t   # (1, 3, 3)

    # ── Sample both planes from the volume ────────────────────────────────────
    vol_dev  = volume.to(device)
    img_orig = my_get_slice(vol_dev, frame_v.to(device), (H, W), scale_h, scale_w)[0, 0, 0].cpu().numpy()
    img_perp = my_get_slice(vol_dev, perp_v.to(device),  (H, W), scale_h, scale_w)[0, 0, 0].cpu().numpy()

    # ── Build PyVista scene ───────────────────────────────────────────────────
    plotter = pv.Plotter(title=f'Cross-sections  —  frame {frame_idx}')

    # 3-D volume (bone colormap)
    vol_np = volume.numpy()
    grid = pv.ImageData()
    grid.dimensions = np.array(vol_np.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.cell_data["Intensity"] = vol_np.flatten(order="F")
    plotter.add_volume(grid, scalars="Intensity", cmap="bone", opacity="sigmoid")

    # Original scan plane — red outline, US image textured on
    add_series_rects(plotter, frame_v, indices=[0], colors='red',  opacity=1.0,
                     frames=np.expand_dims(img_orig, 0))

    # Perpendicular plane — blue outline, extracted slice textured on
    add_series_rects(plotter, perp_v,  indices=[0], colors='blue', opacity=1.0,
                     frames=np.expand_dims(img_perp, 0))

    plotter.show_axes()
    plotter.set_background('black')
    plotter.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load data ─────────────────────────────────────────────────────────────
    print(f"Loading: {H5_PATH}")
    with h5py.File(H5_PATH, 'r') as f:
        frames_np = f['frames'][()]    # (N, H, W) uint8
        tforms_np = f['tforms'][()]    # (N, 4, 4) float32
    N, H_orig, W_orig = frames_np.shape
    print(f"  frames: {frames_np.shape},  tforms: {tforms_np.shape}")

    # 2. Initial series (used only to anchor gap integration at frame 0) ────────
    print("Building initial series ...")
    init_series = build_series(tforms_np, CALIB_PATH, IMG_H, IMG_W)  # (N, 3, 3)
    print(f"  initial series: {init_series.shape}")

    # 3. Source frames for backbone (may be resized) ───────────────────────────
    source_orig = torch.from_numpy(frames_np.astype(np.float32) / 255.0)  # (N, H, W)
    source_bb = source_orig
    if H_orig != IMG_H or W_orig != IMG_W:
        source_bb = F.interpolate(
            source_orig.unsqueeze(1), size=(IMG_H, IMG_W),
            mode='bilinear', align_corners=False,
        ).squeeze(1)
    source_bb = source_bb.unsqueeze(1)   # (N, 1, H, W)

    # 4. Pair inputs for backbone ──────────────────────────────────────────────
    pairs = torch.cat([source_bb[:-1], source_bb[1:]], dim=1)  # (N-1, 2, H, W)
    print(f"  pair input shape: {pairs.shape}")

    # 5. Load backbone ─────────────────────────────────────────────────────────
    print(f"Loading backbone from: {CKPT_PATH}")
    backbone = Backbone(in_planes=IN_PLANES, num_classes=NUM_CLASSES).to(device)
    state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    backbone.load_state_dict(state)
    backbone.eval()
    print("  Model loaded.")

    # 6. Inference ─────────────────────────────────────────────────────────────
    print(f"Running inference on {N-1} frame pairs ...")
    all_gaps = []
    with torch.no_grad():
        for start in range(0, N - 1, BATCH_PAIRS):
            end = min(start + BATCH_PAIRS, N - 1)
            batch = pairs[start:end].unsqueeze(0).to(device)
            out = backbone(batch)
            all_gaps.append(out.squeeze(0).cpu())
    fake_gaps = torch.cat(all_gaps, dim=0)   # (N-1, 6)
    fake_gaps[:, 3:] /= 100.0               # de-scale angles
    print(f"  predicted gaps: {fake_gaps.shape}")

    # 7. Integrate gaps → predicted series ─────────────────────────────────────
    print("Integrating gaps → predicted series ...")
    pred_series = dof_to_series(
        init_series[0:1],
        fake_gaps.unsqueeze(0),
    )                                        # (N, 3, 3)
    print(f"  predicted series: {pred_series.shape}")

    # Free backbone GPU memory before reconstruction
    del backbone
    torch.cuda.empty_cache()

    # 8. 3D reconstruction ─────────────────────────────────────────────────────
    calib_scale, _, _ = read_calib_matrices(CALIB_PATH)
    scale_w = float(calib_scale[0, 0])   # u pixel → mm
    scale_h = float(calib_scale[1, 1])   # v pixel → mm
    print(f"\nCalibration scales: scale_w={scale_w:.6f}, scale_h={scale_h:.6f} mm/pixel")

    source_recon = source_orig   # (N, H_orig, W_orig)

    print("\nReconstructing predicted volume ...")
    volume_pred, recon_bias = reconstruct_volume(source_recon, pred_series, scale_w, scale_h, "Pred")

    # 9. Save volume ───────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "volume_pred.npy"), volume_pred.numpy())
    print(f"\nVolume saved to '{OUT_DIR}/'")

    # 10. Slice-view figure ────────────────────────────────────────────────────
    print("\nSaving slice-view figure ...")
    save_slice_views(
        volume_pred.numpy(),
        "Predicted Reconstruction — Axial / Sagittal / Coronal",
        os.path.join(OUT_DIR, "slices_pred.png"),
    )

    # 11. PyVista interactive ──────────────────────────────────────────────────
    print("\nLaunching PyVista volume rendering ...")
    pyvista_visualize(volume_pred.numpy(), OUT_DIR)

    # 12. Cross-section extraction ─────────────────────────────────────────────
    print(f"\nExtracting cross-sections at frame {FRAME_IDX} ...")
    extract_cross_slices(
        volume_pred, recon_bias, init_series,
        FRAME_IDX, H_orig, W_orig, scale_h, scale_w,
    )


if __name__ == '__main__':
    main()
