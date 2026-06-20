"""
Memory-efficient inference for Online_RecON_Backbone.

Optimizations vs infer_RecON_baseline.py:
  1. Source frames kept on CPU; edge maps computed in GPU micro-batches then
     immediately moved back to CPU.
  2. Optical flow result collected on CPU (CUDA-OpenCV path already does this;
     RAFT path benefits from slices.device == cpu).
  3. Full input tensor (1, N-1, 6, H, W) built on CPU; only one chunk is
     transferred to GPU at a time during inference.
  4. Backbone.forward now accepts and returns the LSTM hidden state so the
     sequence can be processed in fixed-size chunks without losing context.
  5. Automatic mixed precision (FP16 activations) via torch.amp.autocast.
  6. Intermediate tensors freed with del + torch.cuda.empty_cache() at each step.

Tunable knobs (see constants below):
  INFER_CHUNK  -- frames per LSTM chunk  (lower → less GPU RAM, more iterations)
  EDGE_CHUNK   -- frames per Canny batch (lower → less GPU RAM, more iterations)
"""

import sys
import os

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import contextlib
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
import timm

import models
from utils.image import get_optical_flow, get_edge

# ── Paths ─────────────────────────────────────────────────────────────────────
H5_PATH    = "/media/wu/Extreme SSD/datasets/11178509/train_part1/022/LH_Par_L_PtD.h5"
CALIB_PATH = "../../data/calib_matrix.csv"
CKPT_PATH  = "../../save/online_RecON_bk-hp_bk-TUS_complete/online_RecON_bk_backbone_10.pth"
OUT_DIR    = "../../infer_RecON_baseline_output"

# ── Model config (must match training) ────────────────────────────────────────
IN_PLANES   = 6
NUM_CLASSES = 6
IMG_H, IMG_W = 480, 640

# ── Memory-saving knobs ───────────────────────────────────────────────────────
INFER_CHUNK = 50   # LSTM inference frames per chunk — reduce if still OOM
EDGE_CHUNK  = 32   # Canny edge frames per GPU batch — reduce if still OOM

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# Backbone — identical architecture; forward now accepts/returns LSTM hidden
# state so the caller can chunk the sequence without losing temporal context.
# ─────────────────────────────────────────────────────────────────────────────
class Backbone(nn.Module):

    def __init__(self, in_planes, num_classes):
        super().__init__()
        self.resnet = timm.create_model(
            'resnet18', pretrained=False,
            in_chans=in_planes, num_classes=0, global_pool=''
        )
        self.lstm = models.layers.convolutional_rnn.Conv2dLSTM(
            512, 512, kernel_size=3, batch_first=True
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, hx=None, return_feature=False):
        """
        x   : (B, T, C, H, W)
        hx  : LSTM hidden state from previous chunk, or None for first chunk
        Returns (out, feature, hx_new)
          out     : (B, T, num_classes)
          feature : (B, T, 512) if return_feature else None
          hx_new  : hidden state to pass into the next chunk
        """
        b, t, c, h, w = x.shape
        x = (x - torch.mean(x, dim=[3, 4], keepdim=True)) / \
            (torch.std(x, dim=[3, 4], keepdim=True) + 1e-6)
        x = x.view(b * t, c, h, w)
        x = self.resnet(x)
        x = x.view(b, t, *x.shape[1:])
        if return_feature:
            f = self.avg(x)
            f = f.view(f.size(0), f.size(1), -1)
        else:
            f = None
        x, hx = self.lstm(x, hx)           # carry hidden state across chunks
        x = self.avg(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.fc(x)
        return x, f, hx


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helpers  (unchanged from infer_RecON_baseline.py)
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
         [1.0,         float(H_out), 0.0, 1.0],
         [float(W_out), float(H_out), 0.0, 1.0]],
        dtype=T_combined.dtype,
    ).T

    world_pts = torch.bmm(
        T_combined,
        pixel_pts.unsqueeze(0).expand(N, 4, 3)
    )
    return world_pts[:, :3, :].permute(0, 2, 1)


# ─────────────────────────────────────────────────────────────────────────────
# dof_to_series  (unchanged from infer_RecON_baseline.py)
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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # AMP context: enabled only on CUDA
    if device.type == 'cuda':
        autocast_ctx = torch.amp.autocast(device_type='cuda')
    else:
        autocast_ctx = contextlib.nullcontext()

    # 1. Load h5 ───────────────────────────────────────────────────────────────
    print(f"Loading: {H5_PATH}")
    with h5py.File(H5_PATH, 'r') as f:
        frames_np = f['frames'][()]
        tforms_np = f['tforms'][()]
    N, H_orig, W_orig = frames_np.shape
    print(f"  frames: {frames_np.shape}, tforms: {tforms_np.shape}")

    # 2. Ground-truth series (CPU) ─────────────────────────────────────────────
    print("Building GT series ...")
    gt_series = build_series(tforms_np, CALIB_PATH, IMG_H, IMG_W)
    print(f"  gt_series: {gt_series.shape}")

    # 3. Source frames on CPU ──────────────────────────────────────────────────
    # Optimization: keep source on CPU to avoid holding N × H × W on GPU.
    source_cpu = torch.from_numpy(frames_np.astype(np.float32) / 255.0)  # (N, H, W)
    del frames_np
    if H_orig != IMG_H or W_orig != IMG_W:
        source_cpu = F.interpolate(
            source_cpu.unsqueeze(1), size=(IMG_H, IMG_W),
            mode='bilinear', align_corners=False,
        ).squeeze(1)
    print(f"  source (CPU): {source_cpu.shape}")

    # 4. Edge maps — computed in GPU micro-batches, stored on CPU ──────────────
    # Optimization: never hold all N edge maps on GPU simultaneously.
    print(f"Computing edge maps in chunks of {EDGE_CHUNK} ...")
    edge_chunks = []
    for start in range(0, N, EDGE_CHUNK):
        chunk = source_cpu[start:start + EDGE_CHUNK].to(device)
        with torch.no_grad():
            e = get_edge(chunk, device=device)          # (c, 1, H, W) on GPU
        edge_chunks.append(e.cpu())
        del chunk, e
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    edge_cpu = torch.cat(edge_chunks, dim=0)            # (N, 1, H, W) on CPU
    del edge_chunks
    print(f"  edge (CPU): {edge_cpu.shape}")

    # 5. Optical flow — result on CPU ──────────────────────────────────────────
    # Passing source_cpu (device=cpu) makes get_optical_flow return a CPU tensor.
    # RAFT: flows computed on GPU one-by-one, final cat moved to slices.device=cpu.
    # CUDA-OpenCV: already uses CPU numpy internally.
    print("Computing optical flow ...")
    optical_flow_cpu = get_optical_flow(source_cpu, device=device)   # (N-1, 2, H, W) CPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f"  optical_flow (CPU): {optical_flow_cpu.shape}")

    # 6. Build full input tensor on CPU ────────────────────────────────────────
    # Optimization: (1, N-1, 6, H, W) lives on CPU; GPU sees only one chunk.
    print("Building model input tensor (CPU) ...")
    s  = source_cpu.unsqueeze(1).unsqueeze(0)       # (1, N,   1, H, W)
    e  = edge_cpu.unsqueeze(0)                       # (1, N,   1, H, W)
    of = optical_flow_cpu.unsqueeze(0)               # (1, N-1, 2, H, W)
    inp_cpu = torch.cat([s[:, :-1], s[:, 1:], e[:, :-1], e[:, 1:], of], dim=2)
    print(f"  inp (CPU): {inp_cpu.shape}")           # (1, N-1, 6, H, W)

    del source_cpu, edge_cpu, optical_flow_cpu, s, e, of
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 7. Load model ────────────────────────────────────────────────────────────
    print(f"Loading backbone from: {CKPT_PATH}")
    backbone = Backbone(in_planes=IN_PLANES, num_classes=NUM_CLASSES).to(device)
    state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
    backbone.load_state_dict(state)
    backbone.eval()
    print("  Model loaded.")

    # 8. Chunked LSTM inference ────────────────────────────────────────────────
    # Optimization: send INFER_CHUNK frames at a time; carry LSTM hidden state
    # across chunks so temporal context is never broken.
    print(f"Running inference in chunks of {INFER_CHUNK} frames ...")
    all_gaps = []
    hx = None
    with torch.no_grad(), autocast_ctx:
        for start in range(0, N - 1, INFER_CHUNK):
            chunk = inp_cpu[:, start:start + INFER_CHUNK].to(device)
            out, _, hx = backbone(chunk, hx)        # (1, chunk_t, 6)
            all_gaps.append(out.squeeze(0).float().cpu())
            del chunk
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    fake_gaps = torch.cat(all_gaps, dim=0)          # (N-1, 6)
    del all_gaps, hx, inp_cpu
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    fake_gaps[:, 3:] /= 100.0
    print(f"  predicted gaps: {fake_gaps.shape}")

    # 9. Reconstruct series ────────────────────────────────────────────────────
    print("Reconstructing predicted series ...")
    start_frame = gt_series[0:1]
    pred_series = dof_to_series(start_frame, fake_gaps.unsqueeze(0))
    print(f"  predicted series: {pred_series.shape}")

    # 10. Save outputs ─────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "predicted_gaps.npy"),   fake_gaps.numpy())
    np.save(os.path.join(OUT_DIR, "predicted_series.npy"), pred_series.numpy())
    np.save(os.path.join(OUT_DIR, "gt_series.npy"),        gt_series.numpy())
    print(f"\nResults saved to '{OUT_DIR}/'")

    # 11. Quick sanity stats ───────────────────────────────────────────────────
    dist_err = torch.norm(
        pred_series[:, 0, :] - gt_series[:, 0, :], dim=-1
    ).mean().item()
    print(f"Mean center-point distance error: {dist_err:.4f} mm")

    # 12. Visualize ────────────────────────────────────────────────────────────
    visualize(gt_series.numpy(), pred_series.numpy(), OUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization  (unchanged from infer_RecON_baseline.py)
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


def visualize(gt, pred, out_dir):
    n = gt.shape[0]
    gt_center   = gt[:, 0, :]
    pred_center = pred[:, 0, :]

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