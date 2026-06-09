"""
Test: inference → 3D reconstruction → slice comparison.

For a chosen scan and a chosen frame index:
  1. Load the h5 scan and compute GT series from calibration + sensor transforms.
  2. Run the pre-trained baseline backbone to predict 6-DoF inter-frame gaps.
  3. Integrate the gaps into a full predicted probe trajectory.
  4. Reconstruct a 3D ultrasound volume from the predicted trajectory.
  5. Cut a 2D slice from that volume at the predicted position of the chosen frame.
  6. Save a side-by-side figure: original US frame | reconstructed slice.

Run from the project root:
    python trial/recon/infer_recon_slice.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import h5py
import pyvista as pv

import trial.my_utils.functions as my_utils
from utils.plot_functions import add_series_rects

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  ← adjust these to change the scan / frame
# ─────────────────────────────────────────────────────────────────────────────
H5_PATH     = "data/frames_transfs/001/LH_Per_L_DtP.h5"
CALIB_PATH  = "data/calib_matrix.csv"
CKPT_PATH   = "save/online_baseline_bk-hp_bk-TUS_complete/online_baseline_bk_backbone_230.pth"
QUERY_FRAME = 3        # which frame to slice from the reconstructed volume
DOWN_RATIO  = 1        # spatial downsampling applied before reconstruction
OUT_PATH    = "trial/recon/infer_recon_slice_output.png"

IN_PLANES   = 2
NUM_CLASSES = 6          # target.elements(15) − 9
IMG_H, IMG_W = 480, 640
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Backbone  (copy of models/online_baseline_backbone.py – no project imports)
# ─────────────────────────────────────────────────────────────────────────────
class Backbone(nn.Module):
    def __init__(self, in_planes, num_classes):
        super().__init__()
        self.efficientnet_b1 = timm.create_model(
            'efficientnet_b1', pretrained=False,
            in_chans=in_planes, num_classes=num_classes,
        )

    def forward(self, x):
        """x : (B, T, C, H, W)  →  out : (B, T, num_classes)"""
        b, t, c, h, w = x.shape
        x = (x - x.mean(dim=[3, 4], keepdim=True)) / (x.std(dim=[3, 4], keepdim=True) + 1e-6)
        x = x.view(b * t, c, h, w)
        x = self.efficientnet_b1(x)      # (B*T, num_classes)
        return x.view(b, t, -1)          # (B, T, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_calib(calib_path):
    """Returns calib_scale (4×4), calib_R_T (4×4), calib = calib_R_T @ calib_scale."""
    raw = np.empty((8, 4), np.float32)
    with open(calib_path) as f:
        rows = [r.strip().split(',') for r in f.readlines()]
    raw[0:4] = np.array(rows[1:5],  dtype=np.float32)
    raw[4:8] = np.array(rows[6:10], dtype=np.float32)
    calib_scale = torch.tensor(raw[0:4])
    calib_R_T   = torch.tensor(raw[4:8])
    calib       = torch.tensor(raw[4:8] @ raw[0:4])
    return calib_scale, calib_R_T, calib


def build_series(tforms_np, calib_path):
    """
    Build GT series (N, 3, 3) = [center, lower-left, lower-right] in world-mm,
    relative to frame-0 coordinate, at full resolution (IMG_H × IMG_W).
    """
    tforms      = torch.from_numpy(tforms_np)                              # (N, 4, 4)
    N           = tforms.shape[0]
    tforms_inv  = torch.linalg.inv(tforms)
    tforms_f2f0 = torch.matmul(tforms_inv[0:1].expand(N, 4, 4), tforms)  # (N, 4, 4)

    calib_scale, calib_R_T, calib = read_calib(calib_path)
    T_combined = torch.matmul(
        torch.linalg.inv(calib_R_T).unsqueeze(0),      # (1, 4, 4) → broadcasts
        torch.matmul(tforms_f2f0, calib.unsqueeze(0))  # (N, 4, 4) @ (1, 4, 4)
    )  # (N, 4, 4)

    pixel_pts = torch.tensor(
        [[IMG_W / 2.0, IMG_H / 2.0, 0.0, 1.0],  # center
         [1.0,         float(IMG_H), 0.0, 1.0],  # lower-left
         [float(IMG_W), float(IMG_H), 0.0, 1.0]], # lower-right
        dtype=T_combined.dtype,
    ).T  # (4, 3)

    world_pts = T_combined.bmm(pixel_pts.unsqueeze(0).expand(N, 4, 3))
    return world_pts[:, :3, :].permute(0, 2, 1)   # (N, 3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# dof_to_series  (mirrors utils/simulation.py)
# ─────────────────────────────────────────────────────────────────────────────
def _euler_matrix(angles):
    """angles : (K, 3) in radians  →  rotation matrices (K, 4, 4)"""
    ai, aj, ak = angles[:, 0], angles[:, 1], angles[:, 2]
    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    K = angles.shape[0]
    M = torch.zeros(K, 4, 4, dtype=angles.dtype, device=angles.device)
    M[:, 0, 0] = cj * ck
    M[:, 0, 1] = sj * si * ck - ci * sk
    M[:, 0, 2] = sj * ci * ck + si * sk
    M[:, 1, 0] = cj * sk
    M[:, 1, 1] = sj * si * sk + ci * ck
    M[:, 1, 2] = sj * ci * sk - si * ck
    M[:, 2, 0] = -sj
    M[:, 2, 1] = cj * si
    M[:, 2, 2] = cj * ci
    M[:, 3, 3] = 1.0
    return M


def _get_axis(series):
    v1 = series[:, 2, :] - series[:, 1, :]
    v2 = series[:, 0, :] - series[:, 1, :]
    v3 = torch.cross(v1, v2, dim=1)
    axis = torch.stack([v1, v2, v3], dim=1)
    return axis / axis.norm(dim=2, keepdim=True).clamp(min=1e-8)


def dof_to_series(start_point, dof):
    """
    start_point : (1, 3, 3)  – first frame's world-mm probe position
    dof         : (1, T, 6)  – predicted 6-DoF gaps [tx,ty,tz,rx,ry,rz]
    Returns     : (T+1, 3, 3) full series including the start frame
    """
    sp  = start_point.double()
    d   = dof.double()
    b, t, _ = d.shape

    mat = _euler_matrix(d.view(b * t, -1).double())
    mat[:, :3, 3] = d.view(b * t, -1)[:, :3]
    mat = mat.view(b, t, 4, 4)

    ax = _get_axis(sp).permute(0, 2, 1)                          # (1, 3, 3)
    sm = torch.cat([ax, sp[:, 0:1, :].permute(0, 2, 1)], dim=-1) # (1, 3, 4)
    sm = F.pad(sm, (0, 0, 0, 1))
    sm[:, 3, 3] = 1.0
    sm_inv = torch.linalg.inv(sm)

    chain = [sm]
    for i in range(t):
        chain.append(chain[-1].bmm(mat[:, i]))
    chain = torch.stack(chain, dim=1)                            # (1, T+1, 4, 4)

    sp4 = F.pad(sp, (0, 1))
    sp4[:, :, 3] = 1.0
    series = torch.einsum('btij,bjk,bkl->btil',
                          chain, sm_inv, sp4.permute(0, 2, 1))
    series = series.permute(0, 1, 3, 2)[..., :3].squeeze(0)     # (T+1, 3, 3)
    return series.to(start_point.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # ── 1. Load scan ──────────────────────────────────────────────────────────
    h5_abs = os.path.join(_ROOT, H5_PATH)
    print(f"\n[1] Loading scan: {h5_abs}")
    with h5py.File(h5_abs, 'r') as f:
        frames_np = f['frames'][()]    # (N, H, W) uint8
        tforms_np = f['tforms'][()]    # (N, 4, 4) float32
    N = frames_np.shape[0]
    print(f"    frames {frames_np.shape}  tforms {tforms_np.shape}")
    assert 0 <= QUERY_FRAME < N, \
        f"QUERY_FRAME={QUERY_FRAME} out of range [0, {N - 1}]"

    # ── 2. GT series & calibration ────────────────────────────────────────────
    calib_abs = os.path.join(_ROOT, CALIB_PATH)
    print(f"\n[2] Building GT series from {calib_abs}")
    gt_series = build_series(tforms_np, calib_abs)   # (N, 3, 3)

    calib_scale, _, _ = read_calib(calib_abs)
    scale_w = float(calib_scale[0, 0])   # mm per pixel (width direction)
    scale_h = float(calib_scale[1, 1])   # mm per pixel (height direction)
    print(f"    scale_h={scale_h:.4f} mm/px   scale_w={scale_w:.4f} mm/px")

    # ── 3. Prepare source frames ──────────────────────────────────────────────
    source = torch.from_numpy(frames_np.astype(np.float32) / 255.0)  # (N, H, W)
    H_orig, W_orig = source.shape[1], source.shape[2]
    if H_orig != IMG_H or W_orig != IMG_W:
        source = F.interpolate(
            source.unsqueeze(1), size=(IMG_H, IMG_W),
            mode='bilinear', align_corners=False,
        ).squeeze(1)
    source = source.unsqueeze(1)   # (N, 1, H, W)

    # ── 4. Backbone inference ─────────────────────────────────────────────────
    ckpt_abs = os.path.join(_ROOT, CKPT_PATH)
    print(f"\n[3] Loading backbone checkpoint: {ckpt_abs}")
    backbone = Backbone(IN_PLANES, NUM_CLASSES).to(DEVICE)
    backbone.load_state_dict(torch.load(ckpt_abs, map_location=DEVICE))
    backbone.eval()

    pairs = torch.cat([source[:-1], source[1:]], dim=1)   # (N-1, 2, H, W)
    print(f"    Running inference on {N - 1} frame pairs ...")
    gaps_list = []
    with torch.no_grad():
        for i in range(N - 1):
            batch = pairs[i:i + 1].unsqueeze(0).to(DEVICE)   # (1, 1, 2, H, W)
            gaps_list.append(backbone(batch).squeeze(0).cpu())
    fake_gaps = torch.cat(gaps_list, dim=0)   # (N-1, 6)
    fake_gaps[:, 3:] /= 100.0                 # de-scale rotation angles
    print(f"    predicted gaps: {fake_gaps.shape}")

    # ── 5. Predict series ─────────────────────────────────────────────────────
    print(f"\n[4] Building predicted trajectory ...")
    pred_series = dof_to_series(
        gt_series[0:1],           # (1, 3, 3) — share starting position with GT
        fake_gaps.unsqueeze(0),   # (1, N-1, 6)
    )                              # (N, 3, 3)
    print(f"    predicted series: {pred_series.shape}")

    # ── 6. Reconstruct 3D volume ──────────────────────────────────────────────
    mat_scale = torch.eye(4, dtype=torch.float32, device=DEVICE)
    for i in range(3):
        mat_scale[i, i] = DOWN_RATIO

    source_down = F.interpolate(
        source.squeeze(1).unsqueeze(1), scale_factor=DOWN_RATIO,
        mode='bilinear', align_corners=False,
    ).squeeze(1)   # (N, H*dr, W*dr)

    print(f"\n[5] Reconstructing 3D volume (down_ratio={DOWN_RATIO}) ...")
    volume, bias = my_utils.reco(
        source_down.to(DEVICE),
        pred_series.to(DEVICE),
        scale_w, scale_h,
        mat_scale,
    )
    print(f"    volume: {tuple(volume.shape)}  bias: {bias.cpu().numpy().round(2)}")

    # upsample volume to full resolution for accurate slice extraction
    volume_up = F.interpolate(
        volume.unsqueeze(0).unsqueeze(0),
        scale_factor=1.0 / DOWN_RATIO,
        mode='trilinear', align_corners=False,
    ).squeeze(0).squeeze(0)
    bias = bias.cpu()

    # ── 7. Extract slice at the chosen frame's predicted position ─────────────
    print(f"\n[6] Slicing volume at frame {QUERY_FRAME} ...")
    frame_series_biased = (pred_series[QUERY_FRAME:QUERY_FRAME + 1].to(DEVICE)
                           - bias.to(DEVICE))   # (1, 3, 3) shifted into volume coords

    slice_img = my_utils.get_slice(
        volume_up,
        frame_series_biased,
        (IMG_H, IMG_W),
        scale_h=scale_h,
        scale_w=scale_w,
    )  # (1, 1, 1, H, W)
    slice_img = slice_img.squeeze().cpu()   # (H, W)

    # ── 8. 2D side-by-side comparison (interactive matplotlib) ───────────────
    orig_frame = source[QUERY_FRAME, 0].cpu()   # (H, W)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"Frame {QUERY_FRAME} / {N - 1} — Original US frame vs. slice from predicted volume\n"
        f"Scan: {os.path.basename(H5_PATH)}   Backbone: epoch-230",
        fontsize=11,
    )
    axes[0].imshow(orig_frame.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original frame")
    axes[0].axis('off')

    axes[1].imshow(slice_img.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Slice from predicted volume")
    axes[1].axis('off')

    plt.tight_layout()
    out_abs = os.path.join(_ROOT, OUT_PATH)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    plt.savefig(out_abs, dpi=150, bbox_inches='tight')
    print(f"Saved 2D figure → {out_abs}")
    plt.show()   # interactive window — close to continue to 3D viewer

    # ── 9. 3D interactive volume viewer (PyVista) ─────────────────────────────
    print("\n[8] Opening 3D PyVista viewer ...")

    # series in volume coordinate system (subtract the volume origin bias)
    source_np          = source.squeeze(1).cpu().numpy()                   # (N, H, W) float32
    pred_series_biased = (pred_series.cpu() - bias).numpy()                # (N, 3, 3)
    query_biased       = pred_series_biased[QUERY_FRAME:QUERY_FRAME + 1]  # (1, 3, 3)

    # display every ~step-th frame as a textured rectangle
    step            = max(N // 10, 1)
    display_indices = list(range(0, N, step))
    if display_indices[-1] != N - 1:
        display_indices.append(N - 1)

    vol_np = volume_up.cpu().numpy()   # (D0, D1, D2)
    grid   = pv.ImageData()
    grid.dimensions = np.array(vol_np.shape)
    grid.spacing    = (1, 1, 1)
    grid.point_data["Intensity"] = vol_np.flatten(order="F")

    plotter = pv.Plotter(title="Predicted 3D Reconstruction")
    plotter.add_volume(grid, scalars="Intensity", cmap="bone", opacity="sigmoid")

    # predicted trajectory — red textured rectangles
    add_series_rects(plotter, pred_series_biased,
                     indices=display_indices, colors='red',
                     opacity=0.8, frames=source_np)

    # query frame — solid blue rectangle
    add_series_rects(plotter, query_biased,
                     indices=[0], colors='blue', opacity=1.0)

    plotter.show_axes()
    plotter.set_background('black')
    plotter.show()


if __name__ == '__main__':
    main()