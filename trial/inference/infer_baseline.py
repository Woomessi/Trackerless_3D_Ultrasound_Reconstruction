"""
Inference script for Online_Baseline_Backbone.

Loads the latest checkpoint (epoch 5) from
  save/online_baseline_bk-hp_bk-TUS_subject/
and runs it on one .h5 file from data/frames_transfs/.

Outputs are saved to infer_baseline_output/:
  - predicted_gaps.npy   : (N-1, 6)  [tx, ty, tz, rx, ry, rz]
  - predicted_series.npy : (N,   3, 3) world-mm frame positions
  - gt_series.npy        : (N,   3, 3) ground-truth series
  - trajectory_3d.png    : 3D centre-point trajectories + sampled frame quads
  - trajectory_2d.png    : 2D projections + per-axis error curves
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# ── Paths ─────────────────────────────────────────────────────────────────────
# H5_PATH    = "../../data/frames_transfs/001/LH_Per_L_DtP.h5"
H5_PATH    = "../../data/frames_transfs/002/LH_Par_L_PtD.h5"
CALIB_PATH = "../../data/calib_matrix.csv"
CKPT_PATH  = "../../save/online_jagged_bk-hp_bk-TUS_jagged_subject/online_jagged_bk_backbone_10.pth"
OUT_DIR    = "../../infer_baseline_output"

# ── Model config (must match training) ────────────────────────────────────────
IN_PLANES   = 2   # two consecutive frames (channel=2)
NUM_CLASSES = 6   # target.elements(15) - 9 = 6 gaps (tx,ty,tz,rx,ry,rz)
IMG_H, IMG_W = 480, 640
BATCH_PAIRS  = 1  # number of frame-pairs to process at once (memory budget)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# Backbone (copy of the class in models/online_baseline_backbone.py)
# ─────────────────────────────────────────────────────────────────────────────
import timm

class Backbone(nn.Module):
    def __init__(self, in_planes, num_classes):
        super().__init__()
        self.efficientnet_b1 = timm.create_model(
            'efficientnet_b1', pretrained=False,
            in_chans=in_planes, num_classes=num_classes
        )
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """x: (B, T, C, H, W)  →  out: (B, T, num_classes)"""
        b, t, c, h, w = x.shape
        x = (x - torch.mean(x, dim=[3, 4], keepdim=True)) / \
            (torch.std(x, dim=[3, 4], keepdim=True) + 1e-6)
        x = x.view(b * t, c, h, w)
        x = self.efficientnet_b1(x)           # (B*T, num_classes)
        x = x.view(b, t, *x.shape[1:])        # (B, T, num_classes)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helpers  (from utils/plot_functions.py)
# ─────────────────────────────────────────────────────────────────────────────
def read_calib_matrices(filename_calib):
    tform_calib = np.empty((8, 4), np.float32)
    with open(filename_calib, 'r') as csv_file:
        txt = [i.strip('\n').split(',') for i in csv_file.readlines()]
        tform_calib[0:4, :] = np.array(txt[1:5]).astype(np.float32)
        tform_calib[4:8, :] = np.array(txt[6:10]).astype(np.float32)
    calib_scale   = torch.tensor(tform_calib[0:4, :])
    calib_R_T     = torch.tensor(tform_calib[4:8, :])
    calib         = torch.tensor(tform_calib[4:8, :] @ tform_calib[0:4, :])
    return calib_scale, calib_R_T, calib


def build_series(tforms_np, calib_path, H_out, W_out):
    """Compute series (N, 3, 3) = [center, lower-left, lower-right] in world mm.

    Replicates TUS_subject._build_series().
    """
    tforms = torch.from_numpy(tforms_np)               # (N, 4, 4)
    N = tforms.shape[0]

    # Transform each frame to frame-0 coordinate
    pairs = torch.tensor([[0, n] for n in range(N)])   # (N, 2)
    tforms_inv = torch.linalg.inv(tforms)
    tforms_world_to_f0 = tforms_inv[pairs[:, 0]]       # (N, 4, 4)
    tforms_fn_to_world = tforms[pairs[:, 1]]           # (N, 4, 4)
    tforms_f2f0 = torch.matmul(tforms_world_to_f0, tforms_fn_to_world)

    _, calib_R_T, calib = read_calib_matrices(calib_path)
    T_combined = torch.matmul(
        torch.linalg.inv(calib_R_T).unsqueeze(0),
        torch.matmul(tforms_f2f0, calib.unsqueeze(0)),
    )  # (N, 4, 4)

    pixel_pts = torch.tensor(
        [[W_out / 2.0, H_out / 2.0, 0.0, 1.0],
         [1.0,         float(H_out), 0.0, 1.0],
         [float(W_out), float(H_out), 0.0, 1.0]],
        dtype=T_combined.dtype,
    ).T  # (4, 3)

    world_pts = torch.bmm(
        T_combined,
        pixel_pts.unsqueeze(0).expand(N, 4, 3)
    )
    return world_pts[:, :3, :].permute(0, 2, 1)   # (N, 3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# dof_to_series  (from utils/simulation.py)
# ─────────────────────────────────────────────────────────────────────────────
def get_axis(series):
    v1 = series[:, 2, :] - series[:, 1, :]
    v2 = series[:, 0, :] - series[:, 1, :]
    v3 = torch.cross(v1, v2, dim=1)
    axis = torch.stack([v1, v2, v3], dim=1)
    norms = torch.norm(axis, dim=2, keepdim=True).clamp(min=1e-8)
    return axis / norms


def euler_matrix_batch(angles):
    """angles: (B, 3) in radians → rotation matrices (B, 4, 4)."""
    ai, aj, ak = angles[:, 0], angles[:, 1], angles[:, 2]
    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    B = angles.shape[0]
    M = torch.zeros(B, 4, 4, dtype=angles.dtype, device=angles.device)
    M[:, 0, 0] = cj * ck
    M[:, 0, 1] = sj * sc - cs
    M[:, 0, 2] = sj * cc + ss
    M[:, 1, 0] = cj * sk
    M[:, 1, 1] = sj * ss + cc
    M[:, 1, 2] = sj * cs - sc
    M[:, 2, 0] = -sj
    M[:, 2, 1] = cj * si
    M[:, 2, 2] = cj * ci
    M[:, 3, 3] = 1.0
    return M


def dof_to_series(start_point, dof):
    """
    start_point : (1, 3, 3)  – first frame's series
    dof         : (1, T, 6)  – predicted gaps
    Returns     : (T+1, 3, 3) full series including the start frame
    """
    old_type = start_point.dtype
    start_point = start_point.double()
    dof = dof.double()

    b, t, _ = dof.shape
    dof_flat = dof.view(b * t, -1)
    matrix = euler_matrix_batch(dof_flat[:, 3:])
    matrix[:, :3, 3] = dof_flat[:, :3]
    matrix = matrix.view(b, t, 4, 4)

    start_axis = get_axis(start_point).permute(0, 2, 1)        # (1, 3, 3)
    start_matrix = torch.cat(
        [start_axis, start_point[:, 0, :].unsqueeze(-1)], dim=-1
    )                                                           # (1, 3, 4)
    start_matrix = F.pad(start_matrix, (0, 0, 0, 1))           # (1, 4, 4)
    start_matrix[:, 3, 3] = 1.0
    start_matrix_inv = torch.linalg.inv(start_matrix)

    matrix_chain = [start_matrix]
    for idx in range(matrix.shape[1]):
        matrix_chain.append(torch.bmm(matrix_chain[-1], matrix[:, idx]))
    matrix_chain = torch.stack(matrix_chain, dim=1)             # (1, T+1, 4, 4)

    start_point_4d = F.pad(start_point, (0, 1))
    start_point_4d[:, :, 3] = 1.0
    # (1, T+1, 3, 3)
    series = torch.einsum(
        'btij,bjk,bkl->btil',
        matrix_chain,
        start_matrix_inv,
        start_point_4d.permute(0, 2, 1),
    ).permute(0, 1, 3, 2)[..., :3]

    return series.squeeze(0).to(old_type)                       # (T+1, 3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load h5 file ──────────────────────────────────────────────────────────
    print(f"Loading: {H5_PATH}")
    with h5py.File(H5_PATH, 'r') as f:
        frames_np = f['frames'][()]    # (N, H_orig, W_orig) uint8
        tforms_np = f['tforms'][()]    # (N, 4, 4) float32
    N, H_orig, W_orig = frames_np.shape
    print(f"  frames: {frames_np.shape}, tforms: {tforms_np.shape}")

    # 2. Build ground-truth series ─────────────────────────────────────────────
    print("Building GT series from calibration ...")
    gt_series = build_series(tforms_np, CALIB_PATH, IMG_H, IMG_W)  # (N, 3, 3)
    print(f"  gt_series: {gt_series.shape}")

    # 3. Prepare source frames ─────────────────────────────────────────────────
    source = torch.from_numpy(frames_np.astype(np.float32) / 255.0)  # (N, H, W)
    if H_orig != IMG_H or W_orig != IMG_W:
        source = F.interpolate(
            source.unsqueeze(1), size=(IMG_H, IMG_W),
            mode='bilinear', align_corners=False,
        ).squeeze(1)
    # source: (N, H, W)  → add channel dim  → (N, 1, H, W)
    source = source.unsqueeze(1)

    # 4. Build pair inputs: (N-1, 2, H, W) ────────────────────────────────────
    # Model expects (B, T, C, H, W) where C=2 = [frame_i, frame_{i+1}]
    pairs = torch.cat([source[:-1], source[1:]], dim=1)  # (N-1, 2, H, W)
    print(f"  pair input shape: {pairs.shape}")

    # 5. Load model ────────────────────────────────────────────────────────────
    print(f"Loading backbone from: {CKPT_PATH}")
    backbone = Backbone(in_planes=IN_PLANES, num_classes=NUM_CLASSES).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    backbone.load_state_dict(state)
    backbone.eval()
    print("  Model loaded.")

    # 6. Inference in mini-batches ─────────────────────────────────────────────
    print(f"Running inference on {N-1} frame pairs (batch_size={BATCH_PAIRS}) ...")
    all_gaps = []
    with torch.no_grad():
        for start in range(0, N - 1, BATCH_PAIRS):
            end = min(start + BATCH_PAIRS, N - 1)
            batch = pairs[start:end].unsqueeze(0).to(device)  # (1, end-start, 2, H, W)
            out = backbone(batch)                              # (1, end-start, 6)
            all_gaps.append(out.squeeze(0).cpu())             # (end-start, 6)

    fake_gaps = torch.cat(all_gaps, dim=0)   # (N-1, 6)
    # De-scale angles (×100 during training → ÷100 here)
    fake_gaps[:, 3:] /= 100.0
    print(f"  predicted gaps shape: {fake_gaps.shape}")

    # 7. Reconstruct series from predicted gaps ────────────────────────────────
    print("Reconstructing predicted series ...")
    start_frame = gt_series[0:1]                              # (1, 3, 3)
    pred_series = dof_to_series(
        start_frame,                                          # (1, 3, 3)
        fake_gaps.unsqueeze(0),                               # (1, N-1, 6)
    )                                                          # (N, 3, 3)
    print(f"  predicted series shape: {pred_series.shape}")

    # 8. Save outputs ──────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "predicted_gaps.npy"),   fake_gaps.numpy())
    np.save(os.path.join(OUT_DIR, "predicted_series.npy"), pred_series.numpy())
    np.save(os.path.join(OUT_DIR, "gt_series.npy"),        gt_series.numpy())
    print(f"\nResults saved to '{OUT_DIR}/':")
    print(f"  predicted_gaps.npy   {fake_gaps.shape}  (tx,ty,tz,rx,ry,rz per pair)")
    print(f"  predicted_series.npy {pred_series.shape}  (world-mm frame positions)")
    print(f"  gt_series.npy        {gt_series.shape}  (ground-truth)")

    # 9. Quick sanity stats ────────────────────────────────────────────────────
    dist_err = torch.norm(
        pred_series[:, 0, :] - gt_series[:, 0, :], dim=-1
    ).mean().item()
    print(f"\nMean center-point distance error: {dist_err:.4f} mm")

    # 10. Visualize ────────────────────────────────────────────────────────────
    visualize(gt_series.numpy(), pred_series.numpy(), OUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _series_to_corners(s):
    """s: (N, 3, 3) → corners (N, 4, 3)  [ll, lr, ur, ul]"""
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


def visualize(gt, pred, out_dir):
    """
    gt, pred : (N, 3, 3) numpy arrays
    Saves trajectory_3d.png and trajectory_2d.png to out_dir.
    """
    n = gt.shape[0]
    gt_center   = gt[:, 0, :]    # (N, 3)
    pred_center = pred[:, 0, :]

    # ── Figure 1: 3D trajectories ─────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6))
    t = np.linspace(0, 1, n)

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

        ax.set_xlabel('X (mm)', labelpad=4)
        ax.set_ylabel('Y (mm)', labelpad=4)
        ax.set_zlabel('Z (mm)', labelpad=4)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=elev, azim=azim)

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

    # ── Figure 2: 2D projections + per-axis error ─────────────────────────────
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
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)

    for col, (ai, lbl, col_err) in enumerate(zip(
            [0, 1, 2], ['X', 'Y', 'Z'], ['#e74c3c', '#2ecc71', '#3498db'])):
        ax = axes[1, col]
        err = pred_center[:, ai] - gt_center[:, ai]
        ax.plot(frame_idx, err, color=col_err, linewidth=0.8, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
        ax.fill_between(frame_idx, err, 0,
                        where=(err >= 0), alpha=0.15, color=col_err)
        ax.fill_between(frame_idx, err, 0,
                        where=(err < 0),  alpha=0.15, color=col_err)
        ax.set_xlabel('Frame index')
        ax.set_ylabel(f'{lbl} error (mm)')
        ax.set_title(f'{lbl}-axis error  (mean={np.mean(np.abs(err)):.2f} mm)')
        ax.grid(True, alpha=0.3)

        if col == 2:
            ax_in = ax.inset_axes([0.55, 0.55, 0.43, 0.40])
            ax_in.plot(frame_idx, dist_err, color='purple', linewidth=0.7)
            ax_in.set_title(f'3D dist  (mean={dist_err.mean():.1f} mm)', fontsize=7)
            ax_in.set_xlabel('frame', fontsize=6)
            ax_in.tick_params(labelsize=6)

    plt.suptitle("2D Projections & Per-Axis Error — GT vs Predicted", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "trajectory_2d.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
