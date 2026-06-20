"""
Extended inference + 3D reconstruction script.

Based on trial/inference/infer_baseline_reconstruction.py.
Reconstruction method taken from trial/recon/recon.py
(underlying implementation: trial/my_utils/functions.py).

Pipeline
--------
1. Load h5 scan and calibration.
2. Run Online_Baseline_Backbone to predict 6-DoF gaps.
3. Integrate gaps → predicted series (N, 3, 3) world-mm frame positions.
4. Reconstruct 3D US volumes for both GT and predicted series.
5. Visualise: trajectory PNGs + orthogonal-slice PNGs + interactive PyVista.

Outputs (infer_baseline_output/)
---------------------------------
  predicted_gaps.npy       (N-1, 6)
  predicted_series.npy     (N, 3, 3)
  gt_series.npy            (N, 3, 3)
  volume_gt.npy            (X, Y, Z) float32 ground-truth volume
  volume_pred.npy          (X, Y, Z) float32 predicted volume
  trajectory_3d.png
  trajectory_2d.png
  slices_gt.png            axial / sagittal / coronal middle slices
  slices_pred.png
"""

import sys
import os

# Project root → needed for "trial.my_utils.functions"
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

from trial.my_utils.functions import reco as my_reco

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
    vol = vol.cpu()
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
def pyvista_visualize(volume_gt_np, volume_pred_np, out_dir):
    """Side-by-side volume rendering; saves screenshots and shows window."""
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

    grid_gt   = _make_grid(volume_gt_np)
    grid_pred = _make_grid(volume_pred_np)

    pl = pv.Plotter(shape=(1, 2), title="3D US Reconstruction — GT (left) vs Predicted (right)")
    pl.subplot(0, 0)
    pl.add_volume(grid_gt,   scalars="Intensity", cmap="bone", opacity="sigmoid")
    pl.add_text("GT reconstruction",        position="upper_edge", font_size=12)
    pl.show_axes()
    pl.set_background("black")

    pl.subplot(0, 1)
    pl.add_volume(grid_pred, scalars="Intensity", cmap="bone", opacity="sigmoid")
    pl.add_text("Predicted reconstruction", position="upper_edge", font_size=12)
    pl.show_axes()
    pl.set_background("black")

    # Save screenshots before the window opens
    pl.show(auto_close=False)
    screenshot_path = os.path.join(out_dir, "volume_comparison.png")
    pl.screenshot(screenshot_path)
    pl.close()
    print(f"Saved: {screenshot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory visualisation  (from infer_baseline_reconstruction.py)
# ─────────────────────────────────────────────────────────────────────────────
def _series_to_corners(s):
    center, ll, lr = s[:, 0], s[:, 1], s[:, 2]
    ur = 2 * center - ll
    ul = 2 * center - lr
    return np.stack([ll, lr, ur, ul], axis=1)


def _add_quads(ax, series, color, alpha=0.25, step=50):
    corners = _series_to_corners(series)
    for i in range(0, len(corners), step):
        poly = Poly3DCollection([corners[i]], facecolors=color,
                                edgecolors=color, linewidths=0.5, alpha=alpha)
        ax.add_collection3d(poly)


def _set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    half_range = (limits[:, 1] - limits[:, 0]).max() / 2.0
    ax.set_xlim3d(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim3d(centers[1] - half_range, centers[1] + half_range)
    ax.set_zlim3d(centers[2] - half_range, centers[2] + half_range)


def visualize_trajectories(gt, pred, out_dir):
    n = gt.shape[0]
    gt_center   = gt[:, 0, :]
    pred_center = pred[:, 0, :]
    t = np.linspace(0, 1, n)

    fig = plt.figure(figsize=(14, 6))
    for col, (elev, azim, title) in enumerate([
            (20,  45,  "View 1  (elev=20°, azim=45°)"),
            (10, 135,  "View 2  (elev=10°, azim=135°)")]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        cmap_gt   = plt.get_cmap('Blues')
        cmap_pred = plt.get_cmap('Reds')
        for i in range(n - 1):
            ax.plot(gt_center[i:i+2, 0], gt_center[i:i+2, 1], gt_center[i:i+2, 2],
                    color=cmap_gt(0.3 + 0.7 * t[i]),   linewidth=1.2)
            ax.plot(pred_center[i:i+2, 0], pred_center[i:i+2, 1], pred_center[i:i+2, 2],
                    color=cmap_pred(0.3 + 0.7 * t[i]), linewidth=1.2)
        _add_quads(ax, gt,   color='steelblue', alpha=0.20, step=50)
        _add_quads(ax, pred, color='tomato',    alpha=0.20, step=50)
        ax.scatter(*gt_center[0],    color='blue', s=60, marker='o', zorder=5)
        ax.scatter(*gt_center[-1],   color='blue', s=60, marker='*', zorder=5)
        ax.scatter(*pred_center[0],  color='red',  s=60, marker='o', zorder=5)
        ax.scatter(*pred_center[-1], color='red',  s=60, marker='*', zorder=5)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=elev, azim=azim)
        _set_axes_equal_3d(ax)
        if col == 0:
            ax.legend(handles=[
                Patch(facecolor='steelblue', label='GT trajectory'),
                Patch(facecolor='tomato',    label='Predicted trajectory'),
            ], fontsize=8, loc='upper left')
    plt.suptitle("3D Frame Trajectories — GT vs Predicted", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "trajectory_3d.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    frame_idx = np.arange(n)
    dist_err  = np.linalg.norm(pred_center - gt_center, axis=-1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (xi, yi, xlabel, ylabel, title) in enumerate([
            (0, 1, 'X (mm)', 'Y (mm)', 'XY plane'),
            (0, 2, 'X (mm)', 'Z (mm)', 'XZ plane'),
            (1, 2, 'Y (mm)', 'Z (mm)', 'YZ plane')]):
        ax = axes[0, col]
        ax.plot(gt_center[:, xi],   gt_center[:, yi],
                color='steelblue', linewidth=1.2, label='GT')
        ax.plot(pred_center[:, xi], pred_center[:, yi],
                color='tomato',    linewidth=1.2, linestyle='--', label='Pred')
        for k in range(0, n, 100):
            ax.annotate(str(k), xy=(gt_center[k, xi],   gt_center[k, yi]),
                        fontsize=6, color='steelblue', ha='center')
            ax.annotate(str(k), xy=(pred_center[k, xi], pred_center[k, yi]),
                        fontsize=6, color='tomato',    ha='center')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8); ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
    for col, (ai, lbl, col_err) in enumerate(zip(
            [0, 1, 2], ['X', 'Y', 'Z'], ['#e74c3c', '#2ecc71', '#3498db'])):
        ax = axes[1, col]
        err = pred_center[:, ai] - gt_center[:, ai]
        ax.plot(frame_idx, err, color=col_err, linewidth=0.8, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
        ax.fill_between(frame_idx, err, 0, where=(err >= 0), alpha=0.15, color=col_err)
        ax.fill_between(frame_idx, err, 0, where=(err < 0),  alpha=0.15, color=col_err)
        ax.set_xlabel('Frame index'); ax.set_ylabel(f'{lbl} error (mm)')
        ax.set_title(f'{lbl}-axis error  (mean={np.mean(np.abs(err)):.2f} mm)')
        ax.grid(True, alpha=0.3)
        if col == 2:
            ax_in = ax.inset_axes([0.55, 0.55, 0.43, 0.40])
            ax_in.plot(frame_idx, dist_err, color='purple', linewidth=0.7)
            ax_in.set_title(f'3D dist  (mean={dist_err.mean():.1f} mm)', fontsize=7)
            ax_in.set_xlabel('frame', fontsize=6); ax_in.tick_params(labelsize=6)
    plt.suptitle("2D Projections & Per-Axis Error — GT vs Predicted", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "trajectory_2d.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


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

    # 2. GT series ──────────────────────────────────────────────────────────────
    print("Building GT series ...")
    gt_series = build_series(tforms_np, CALIB_PATH, IMG_H, IMG_W)  # (N, 3, 3)
    print(f"  gt_series: {gt_series.shape}")

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
        gt_series[0:1],
        fake_gaps.unsqueeze(0),
    )                                        # (N, 3, 3)
    print(f"  predicted series: {pred_series.shape}")

    # Free backbone GPU memory before reconstruction
    del backbone
    torch.cuda.empty_cache()

    # 8. Save arrays ───────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "predicted_gaps.npy"),   fake_gaps.numpy())
    np.save(os.path.join(OUT_DIR, "predicted_series.npy"), pred_series.numpy())
    np.save(os.path.join(OUT_DIR, "gt_series.npy"),        gt_series.numpy())

    dist_err = torch.norm(pred_series[:, 0, :] - gt_series[:, 0, :], dim=-1).mean().item()
    print(f"\nMean centre-point distance error: {dist_err:.4f} mm")

    # 9. Trajectory visualisation ──────────────────────────────────────────────
    print("\nGenerating trajectory figures ...")
    visualize_trajectories(gt_series.numpy(), pred_series.numpy(), OUT_DIR)

    # 10. 3D reconstruction ────────────────────────────────────────────────────
    calib_scale, _, _ = read_calib_matrices(CALIB_PATH)
    scale_w = float(calib_scale[0, 0])   # u pixel → mm
    scale_h = float(calib_scale[1, 1])   # v pixel → mm
    print(f"\nCalibration scales: scale_w={scale_w:.6f}, scale_h={scale_h:.6f} mm/pixel")

    # Use original (un-resized) frames for reconstruction
    source_recon = source_orig   # (N, H_orig, W_orig)

    print("\nReconstructing GT volume ...")
    volume_gt, bias_gt = reconstruct_volume(source_recon, gt_series, scale_w, scale_h, "GT")

    print("\nReconstructing predicted volume ...")
    volume_pred, bias_pred = reconstruct_volume(source_recon, pred_series, scale_w, scale_h, "Pred")

    # # 11. Save volumes ─────────────────────────────────────────────────────────
    # np.save(os.path.join(OUT_DIR, "volume_gt.npy"),   volume_gt.numpy())
    # np.save(os.path.join(OUT_DIR, "volume_pred.npy"), volume_pred.numpy())
    # print(f"\nVolumes saved to '{OUT_DIR}/'")
    #
    # # 12. Slice-view figures ───────────────────────────────────────────────────
    # print("\nSaving slice-view figures ...")
    # save_slice_views(
    #     volume_gt.numpy(),
    #     "GT Reconstruction — Axial / Sagittal / Coronal",
    #     os.path.join(OUT_DIR, "slices_gt.png"),
    # )
    # save_slice_views(
    #     volume_pred.numpy(),
    #     "Predicted Reconstruction — Axial / Sagittal / Coronal",
    #     os.path.join(OUT_DIR, "slices_pred.png"),
    # )

    # 13. PyVista interactive ──────────────────────────────────────────────────
    print("\nLaunching PyVista volume rendering ...")
    pyvista_visualize(volume_gt.numpy(), volume_pred.numpy(), OUT_DIR)


if __name__ == '__main__':
    main()
