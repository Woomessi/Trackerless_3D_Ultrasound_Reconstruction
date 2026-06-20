"""visualize_jagged_mask.py

对比不同旋转角度下 loss_jagged 中内容掩码的效果，帮助判断是否需要加掩码。

输出到 visualize_jagged_mask_output/jagged_mask.png ，每一行对应一个旋转角度，
每列依次展示：
  旋转切片 | 内容掩码 | drow 热图 | dcol 热图 | 三种 loss 值对比柱状图

最后一行为汇总曲线：有效像素比例 + 三种 loss 值随角度的变化。

三种 loss 公式：
  A) drow.mean() + dcol.mean()            （当前公式，各向同性 TV，无掩码）
  B) relu(drow.mean() - dcol.mean())       （RecON 公式，无掩码）
  C) relu(drow_m.mean() - dcol_m.mean())   （RecON 公式，有掩码）
"""

import sys
import os

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# 将项目根目录插入 sys.path[0]，并移除可能与本项目 utils/models 冲突的其他路径
sys.path = [_PROJ_ROOT] + [p for p in sys.path if _PROJ_ROOT not in p]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

import models
import utils
import utils.image
import utils.simulation
import trial.my_utils.functions as my_utils
from utils.plot_functions import read_calib_matrices

# ── 路径配置 ──────────────────────────────────────────────────────────────────
H5_PATH   = "/media/wu/Extreme SSD/datasets/11178509/train_part1/049/LH_Par_L_PtD.h5"
CALIB_PATH = os.path.join(_PROJ_ROOT, "data", "calib_matrix.csv")
CKPT_PATH = "/media/wu/Extreme SSD/online_LSTM_edge_bk-hp_bk-TUS_LSTM_edge_complete/online_LSTM_edge_bk_backbone_21.pth"
OUT_DIR   = os.path.join(_PROJ_ROOT, "visualize_jagged_mask_output")

# ── 参数 ──────────────────────────────────────────────────────────────────────
IMG_H, IMG_W = 480, 640
IN_PLANES    = 4
NUM_CLASSES  = 6
MAX_FRAMES   = 64      # 用于 reco 的帧数（太多会很慢）
CRIT_SCALE   = 0.5     # 与训练配置保持一致
ANGLES       = [0.3, 0.6, 0.9, 1.5]   # 待测旋转角度（弧度）
MASK_THRESH  = 0.02    # 低于此值视为背景（越界/无数据区域）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# ── Backbone 定义（与训练时一致）────────────────────────────────────────────
class Backbone(nn.Module):
    def __init__(self, in_planes, num_classes):
        super().__init__()
        import timm
        self.resnet = timm.create_model(
            'resnet18', pretrained=False,
            in_chans=in_planes, num_classes=0, global_pool=''
        )
        self.lstm = models.layers.convolutional_rnn.Conv2dLSTM(
            512, 512, kernel_size=3, batch_first=True
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(512, num_classes)

    def forward(self, x, return_feature=False):
        b, t, c, h, w = x.shape
        x = (x - x.mean(dim=[3, 4], keepdim=True)) / \
            (x.std(dim=[3, 4], keepdim=True) + 1e-6)
        x = x.view(b * t, c, h, w)
        x = self.resnet(x)
        x = x.view(b, t, *x.shape[1:])
        f = None
        x = self.lstm(x)[0]
        x = self.avg(x).view(x.size(0), x.size(1), -1)
        x = self.fc(x)
        return x, f


# ── GT series 构建（与 infer_LSTM_edge_baseline.py 相同）────────────────────
def build_series(tforms_np, calib_path, H_out, W_out):
    tforms = torch.from_numpy(tforms_np)
    N = tforms.shape[0]
    tforms_inv = torch.linalg.inv(tforms)
    tforms_w2f0 = tforms_inv[0:1].expand(N, 4, 4)
    tforms_fn2w = tforms
    tforms_f2f0 = torch.bmm(tforms_w2f0, tforms_fn2w)

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
    world_pts = torch.bmm(T_combined, pixel_pts.unsqueeze(0).expand(N, 4, 3))
    return world_pts[:, :3, :].permute(0, 2, 1)   # (N, 3, 3)


# ── 旋转切片提取 ─────────────────────────────────────────────────────────────
def rotated_slice(volume, gt_pos, bias_origin, H, W, scale_h, scale_w, theta, crit_scale):
    """在 gt_pos 处提取绕 ax_x 旋转 theta 角的切片。

    Returns:
        slice_np  (H_c, W_c)  numpy float32
        gt_pos_rot  (1, 3, 3) tensor
    """
    H_c = max(1, int(H * crit_scale))
    W_c = max(1, int(W * crit_scale))

    axis = my_utils.get_axis(gt_pos.float())
    ax_x = axis[0, 0]
    ax_y = axis[0, 1]
    ax_z = axis[0, 2]

    dev, dt = gt_pos.device, gt_pos.dtype
    cos_t    = torch.tensor(theta, device=dev, dtype=dt).cos()
    sin_t    = torch.tensor(theta, device=dev, dtype=dt).sin()
    ax_y_rot = cos_t * ax_y + sin_t * ax_z

    half_w = (W - 1) / 2.0 * scale_w
    half_h = (H - 1) / 2.0 * scale_h
    center = gt_pos[0, 0]
    ll_rot = center - ax_x * half_w - ax_y_rot * half_h
    lr_rot = center + ax_x * half_w - ax_y_rot * half_h
    gt_pos_rot = torch.stack([center, ll_rot, lr_rot]).unsqueeze(0)

    gt_pos_rot_vol = gt_pos_rot - bias_origin.view(1, 1, 3)

    with torch.no_grad():
        sl = my_utils.get_slice(
            volume, gt_pos_rot_vol, (H_c, W_c),
            scale_h=scale_h / crit_scale,
            scale_w=scale_w / crit_scale,
        ).squeeze(0).squeeze(0)   # (1, H_c, W_c) → (H_c, W_c) after squeeze below
    sl = sl.squeeze(0)            # (H_c, W_c)
    return sl.cpu().float().numpy(), gt_pos_rot


# ── 指标计算 ─────────────────────────────────────────────────────────────────
def compute_metrics(sl_np, mask_thresh):
    """
    sl_np: (H_c, W_c) float32 in [0, 1]
    返回 dict{content_frac, loss_A, loss_B, loss_C, drow, dcol, mask}
    """
    sl = torch.from_numpy(sl_np).float()
    mask = (sl > mask_thresh)   # (H_c, W_c)
    content_frac = mask.float().mean().item()

    drow = (sl[1:, :] - sl[:-1, :]).abs()    # (H_c-1, W_c)
    dcol = (sl[:, 1:] - sl[:, :-1]).abs()    # (H_c, W_c-1)

    loss_A = (drow.mean() + dcol.mean()).item()
    loss_B = F.relu(drow.mean() - dcol.mean()).item()

    # 掩码版：仅统计相邻两侧都是有效内容的差分
    drow_mask = mask[1:, :] & mask[:-1, :]   # (H_c-1, W_c)
    dcol_mask = mask[:, 1:] & mask[:, :-1]   # (H_c, W_c-1)
    if drow_mask.any() and dcol_mask.any():
        drow_m = drow[drow_mask].mean()
        dcol_m = dcol[dcol_mask].mean()
        loss_C = F.relu(drow_m - dcol_m).item()
    else:
        loss_C = 0.0

    return {
        'content_frac': content_frac,
        'loss_A':       loss_A,
        'loss_B':       loss_B,
        'loss_C':       loss_C,
        'drow':         drow.numpy(),
        'dcol':         dcol.numpy(),
        'mask':         mask.numpy(),
    }


# ── 可视化 ────────────────────────────────────────────────────────────────────
def make_figure(gt_slice_np, angle_data, angles, mask_thresh, scale_h, scale_w, H, W):
    """
    angle_data: list of (angle, slice_np, metrics_dict)
    """
    n_angles = len(angles)
    n_cols   = 6   # slice | mask | drow | dcol | loss_bar | z_shift_text

    fig = plt.figure(figsize=(22, 3.8 * (n_angles + 2)))
    outer = gridspec.GridSpec(
        n_angles + 2, 1,
        figure=fig,
        hspace=0.55,
    )

    # ── Row 0: 参考切片 + 总说明 ─────────────────────────────────────────────
    ref_gs = gridspec.GridSpecFromSubplotSpec(
        1, n_cols, subplot_spec=outer[0], wspace=0.15
    )
    H_c = max(1, int(H * CRIT_SCALE))
    W_c = max(1, int(W * CRIT_SCALE))
    half_h_mm = (H - 1) / 2.0 * scale_h

    ax0 = fig.add_subplot(ref_gs[0, :2])
    ax0.imshow(gt_slice_np, cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax0.set_title('Reference GT slice  (theta=0, no rotation)', fontsize=11)
    ax0.axis('off')

    ax_txt = fig.add_subplot(ref_gs[0, 2:])
    ax_txt.axis('off')
    info = (
        f"Image size:  H={H}, W={W}  ->  crit_scale={CRIT_SCALE}  "
        f"->  H_c={H_c}, W_c={W_c}\n"
        f"half_h = (H-1)/2 * scale_h = {half_h_mm:.1f} mm\n"
        f"Out-of-volume shift:  dz = half_h * sin(theta)\n"
        f"  theta=0.02 rad -> dz~{half_h_mm*np.sin(0.02):.1f} mm\n"
        f"  theta=0.05 rad -> dz~{half_h_mm*np.sin(0.05):.1f} mm\n"
        f"  theta=0.10 rad -> dz~{half_h_mm*np.sin(0.10):.1f} mm\n"
        f"  theta=0.18 rad -> dz~{half_h_mm*np.sin(0.18):.1f} mm\n\n"
        f"Content mask threshold: {mask_thresh}\n\n"
        f"Three loss formulas (weight=1):\n"
        f"  A (current): drow.mean() + dcol.mean()\n"
        f"  B (RecON):   relu(drow.mean() - dcol.mean())\n"
        f"  C (RecON+mask): relu(drow_m.mean() - dcol_m.mean())"
    )
    ax_txt.text(0.02, 0.98, info, transform=ax_txt.transAxes,
                fontsize=9, va='top', fontfamily='monospace',
                bbox=dict(facecolor='lightyellow', alpha=0.7, edgecolor='gray'))

    # ── Rows 1..N: 每个角度一行 ──────────────────────────────────────────────
    drow_vmax = max(
        m['drow'].max() for _, _, m in angle_data if m['drow'].size > 0
    )
    drow_vmax = max(drow_vmax, 1e-4)

    loss_max = max(
        max(m['loss_A'], m['loss_B'], m['loss_C'])
        for _, _, m in angle_data
    )
    loss_max = max(loss_max, 1e-4)

    for row_i, (theta, sl_np, metrics) in enumerate(angle_data, start=1):
        row_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=outer[row_i], wspace=0.12,
            width_ratios=[3, 2, 2, 2, 2, 1],
        )

        theta_deg = np.degrees(theta)
        dz_mm     = half_h_mm * np.sin(theta)

        # 列 0: 旋转切片
        ax_sl = fig.add_subplot(row_gs[0, 0])
        ax_sl.imshow(sl_np, cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax_sl.set_title(
            f'Rotated slice  theta={theta:.2f} rad ({theta_deg:.1f} deg)\n'
            f'dz~{dz_mm:.1f} mm',
            fontsize=9,
        )
        ax_sl.axis('off')

        # col 1: content mask
        ax_mk = fig.add_subplot(row_gs[0, 1])
        ax_mk.imshow(metrics['mask'].astype(float), cmap='Greens', vmin=0, vmax=1, aspect='auto')
        frac_pct = metrics['content_frac'] * 100
        ax_mk.set_title(f'Content mask\nvalid px: {frac_pct:.1f}%', fontsize=9)
        ax_mk.axis('off')

        # col 2: drow heatmap
        ax_dr = fig.add_subplot(row_gs[0, 2])
        im_dr = ax_dr.imshow(
            metrics['drow'], cmap='hot', vmin=0, vmax=drow_vmax, aspect='auto'
        )
        ax_dr.set_title(f'|drow|\nmean={metrics["drow"].mean():.4f}', fontsize=9)
        ax_dr.axis('off')
        plt.colorbar(im_dr, ax=ax_dr, fraction=0.06, pad=0.02)

        # col 3: dcol heatmap
        ax_dc = fig.add_subplot(row_gs[0, 3])
        im_dc = ax_dc.imshow(
            metrics['dcol'], cmap='hot', vmin=0, vmax=drow_vmax, aspect='auto'
        )
        ax_dc.set_title(f'|dcol|\nmean={metrics["dcol"].mean():.4f}', fontsize=9)
        ax_dc.axis('off')
        plt.colorbar(im_dc, ax=ax_dc, fraction=0.06, pad=0.02)

        # col 4: three loss bar chart
        ax_bar = fig.add_subplot(row_gs[0, 4])
        bars = ax_bar.bar(
            ['A\nno mask\ndr+dc', 'B\nno mask\nrelu(dr-dc)', 'C\nmasked\nrelu(dr-dc)'],
            [metrics['loss_A'], metrics['loss_B'], metrics['loss_C']],
            color=['#e74c3c', '#e67e22', '#2ecc71'],
            width=0.6,
        )
        ax_bar.set_ylim(0, loss_max * 1.15)
        ax_bar.set_ylabel('loss value', fontsize=8)
        ax_bar.tick_params(axis='x', labelsize=7)
        ax_bar.tick_params(axis='y', labelsize=7)
        for bar, val in zip(bars, [metrics['loss_A'], metrics['loss_B'], metrics['loss_C']]):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + loss_max * 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7
            )

        # 列 5: 角度标签（侧栏）
        ax_lbl = fig.add_subplot(row_gs[0, 5])
        ax_lbl.axis('off')
        ax_lbl.text(
            0.5, 0.5,
            f'θ={theta:.2f}\n({theta_deg:.1f}°)',
            transform=ax_lbl.transAxes,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='navy',
        )

    # ── Row N+1: 汇总曲线 ────────────────────────────────────────────────────
    sum_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[n_angles + 1], wspace=0.35
    )
    angles_arr = np.array(angles)
    content_fracs = np.array([m['content_frac'] * 100 for _, _, m in angle_data])
    loss_As       = np.array([m['loss_A'] for _, _, m in angle_data])
    loss_Bs       = np.array([m['loss_B'] for _, _, m in angle_data])
    loss_Cs       = np.array([m['loss_C'] for _, _, m in angle_data])

    # 左图：有效像素比例
    ax_frac = fig.add_subplot(sum_gs[0, 0])
    ax_frac.plot(np.degrees(angles_arr), content_fracs, 'go-', linewidth=2, markersize=7)
    ax_frac.axhline(100, color='gray', linestyle='--', linewidth=0.8)
    for x, y in zip(np.degrees(angles_arr), content_fracs):
        ax_frac.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                         xytext=(0, 6), ha='center', fontsize=9)
    ax_frac.set_xlabel('Rotation angle (deg)', fontsize=10)
    ax_frac.set_ylabel('Valid pixel fraction (%)', fontsize=10)
    ax_frac.set_title('Content mask coverage vs rotation angle', fontsize=10)
    ax_frac.set_ylim(0, 115)
    ax_frac.grid(True, alpha=0.35)

    # right: three loss values
    ax_loss = fig.add_subplot(sum_gs[0, 1])
    ax_loss.plot(np.degrees(angles_arr), loss_As, 'rs-',  linewidth=2, markersize=7, label='A: drow+dcol (no mask)')
    ax_loss.plot(np.degrees(angles_arr), loss_Bs, 'o-',   linewidth=2, markersize=7, color='orange', label='B: relu(dr-dc) (no mask)')
    ax_loss.plot(np.degrees(angles_arr), loss_Cs, 'g^--', linewidth=2, markersize=7, label='C: relu(dr-dc) (masked)')
    ax_loss.set_xlabel('Rotation angle (deg)', fontsize=10)
    ax_loss.set_ylabel('loss value', fontsize=10)
    ax_loss.set_title('Three loss formulas vs rotation angle', fontsize=10)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.35)

    fig.suptitle(
        'loss_jagged content-mask effect visualization\n'
        f'(mask_thresh={MASK_THRESH}, crit_scale={CRIT_SCALE}, frames={MAX_FRAMES})',
        fontsize=13, y=1.005,
    )
    return fig


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 加载数据 ──────────────────────────────────────────────────────────────
    print(f"Loading: {H5_PATH}")
    with h5py.File(H5_PATH, 'r') as f:
        frames_np = f['frames'][()]   # (N, H, W) uint8
        tforms_np = f['tforms'][()]   # (N, 4, 4) float32
    N_full = frames_np.shape[0]
    print(f"  N={N_full} frames, using first {MAX_FRAMES}")

    # 取前 MAX_FRAMES 帧
    frames_np = frames_np[:MAX_FRAMES]
    tforms_np = tforms_np[:MAX_FRAMES]
    N = frames_np.shape[0]

    # 2. 校准参数 ──────────────────────────────────────────────────────────────
    T_calib_scale, _, _ = read_calib_matrices(CALIB_PATH)
    scale_w = float(T_calib_scale[0, 0])
    scale_h = float(T_calib_scale[1, 1])
    print(f"  scale_w={scale_w:.4f} mm/px, scale_h={scale_h:.4f} mm/px")

    mat_scale = torch.eye(4, dtype=torch.float32)   # down_ratio=1

    # 3. 构建 GT series ────────────────────────────────────────────────────────
    gt_series = build_series(tforms_np, CALIB_PATH, IMG_H, IMG_W).float()
    print(f"  gt_series: {gt_series.shape}")

    # 4. 准备源帧 ──────────────────────────────────────────────────────────────
    source = torch.from_numpy(frames_np.astype(np.float32) / 255.0)   # (N, H, W)
    H_orig, W_orig = source.shape[1], source.shape[2]
    if H_orig != IMG_H or W_orig != IMG_W:
        source = F.interpolate(
            source.unsqueeze(1), size=(IMG_H, IMG_W),
            mode='bilinear', align_corners=False,
        ).squeeze(1)
    source = source.unsqueeze(1)   # (N, 1, H, W)

    # 5. Backbone 推理 → 预测 gaps ────────────────────────────────────────────
    print("Loading backbone ...")
    backbone = Backbone(IN_PLANES, NUM_CLASSES).to(device)
    state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    backbone.load_state_dict(state)
    backbone.eval()

    print("Computing edge maps ...")
    with torch.no_grad():
        edge = utils.image.get_edge(source.squeeze(1).to(device), device=device)

    s0 = source[:-1].to(device)
    s1 = source[1:].to(device)
    e0 = edge[:-1]
    e1 = edge[1:]
    inp = torch.cat([s0, s1, e0, e1], dim=1).unsqueeze(0)   # (1, N-1, 4, H, W)

    print("Running backbone inference ...")
    with torch.no_grad():
        raw_gaps, _ = backbone(inp, return_feature=False)   # (1, N-1, 6)
    raw_gaps = raw_gaps.squeeze(0)                           # (N-1, 6)
    fake_gaps = torch.cat([raw_gaps[:, :3], raw_gaps[:, 3:] / 100.0], dim=-1)

    # 6. 重建轨迹 + 体积 ───────────────────────────────────────────────────────
    fake_series = utils.simulation.dof_to_series(
        gt_series[0:1].to(device),
        fake_gaps.unsqueeze(0),
    ).squeeze(0)   # (N, 3, 3)

    print("Reconstructing 3D volume (this may take a moment) ...")
    source_down = source.to(device).squeeze(1)   # (N, H, W)

    with torch.no_grad():
        volume, bias = my_utils.reco(
            source_down, fake_series,
            scale_w, scale_h, mat_scale.to(device),
        )
    print(f"  volume shape: {tuple(volume.shape)}")

    bias_origin = bias.detach()

    # 7. 寻找与体积相交的 GT 帧 ───────────────────────────────────────────────
    GRID = 20
    print("Searching for an intersecting GT frame ...")

    def overlap_frac(series_pos):
        pos    = series_pos.detach().cpu().float()
        b      = bias_origin.cpu().float()
        pos_vol = pos - b.view(1, 1, 3)
        xs = torch.linspace(-(IMG_H-1)/2, (IMG_H-1)/2, GRID)
        ys = torch.linspace(-(IMG_W-1)/2, (IMG_W-1)/2, GRID)
        gx, gy = torch.meshgrid(xs, ys, indexing='ij')
        local = torch.stack([gy * scale_w, -gx * scale_h, torch.zeros_like(gx)], dim=-1)
        axis   = my_utils.get_axis(pos_vol).permute(0, 2, 1)
        center = pos_vol[:, 0, :]
        mesh   = torch.einsum('ij,HWj->HWi', axis[0], local) + center.view(1, 1, 3)
        vs     = volume.shape
        in_b   = (
            (mesh[..., 0] >= 0) & (mesh[..., 0] < vs[0]) &
            (mesh[..., 1] >= 0) & (mesh[..., 1] < vs[1]) &
            (mesh[..., 2] >= 0) & (mesh[..., 2] < vs[2])
        )
        return in_b.float().mean().item()

    best_idx, best_frac = 1, 0.0
    for idx in range(1, N):
        frac = overlap_frac(gt_series[idx:idx+1].to(device))
        if frac > best_frac:
            best_frac = frac
            best_idx  = idx
            if best_frac >= 0.9:
                break

    gt_pos = gt_series[best_idx:best_idx+1].to(device)
    print(f"  best frame idx={best_idx}, overlap={best_frac:.1%}")

    # 8. 提取原始 GT 切片（无旋转，作为参考）──────────────────────────────────
    H_c = max(1, int(IMG_H * CRIT_SCALE))
    W_c = max(1, int(IMG_W * CRIT_SCALE))
    gt_pos_vol = gt_pos - bias_origin.view(1, 1, 3)
    with torch.no_grad():
        sl_gt = my_utils.get_slice(
            volume, gt_pos_vol, (H_c, W_c),
            scale_h=scale_h / CRIT_SCALE,
            scale_w=scale_w / CRIT_SCALE,
        ).squeeze()   # → (H_c, W_c)
    gt_slice_np = sl_gt.cpu().float().numpy()

    # 9. 各旋转角度的切片 + 指标 ──────────────────────────────────────────────
    angle_data = []
    for theta in ANGLES:
        print(f"  angle={theta:.2f} rad ({np.degrees(theta):.1f}°) ...", end=' ')
        sl_np, _ = rotated_slice(
            volume, gt_pos, bias_origin,
            IMG_H, IMG_W, scale_h, scale_w, theta, CRIT_SCALE,
        )
        metrics = compute_metrics(sl_np, MASK_THRESH)
        print(f"content={metrics['content_frac']:.1%}  "
              f"loss_A={metrics['loss_A']:.4f}  "
              f"loss_B={metrics['loss_B']:.4f}  "
              f"loss_C={metrics['loss_C']:.4f}")
        angle_data.append((theta, sl_np, metrics))

    # 10. 生成并保存图像 ────────────────────────────────────────────────────────
    print("Generating figure ...")
    fig = make_figure(
        gt_slice_np, angle_data, ANGLES, MASK_THRESH, scale_h, scale_w, IMG_H, IMG_W
    )
    out_path = os.path.join(OUT_DIR, "jagged_mask.png")
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n保存至: {out_path}")

    # 11. 打印汇总表 ───────────────────────────────────────────────────────────
    half_h_mm = (IMG_H - 1) / 2.0 * scale_h
    print("\n--- Summary -----------------------------------------------------------")
    print(f"{'angle(rad)':>11}  {'angle(deg)':>10}  {'dz(mm)':>7}  "
          f"{'valid%':>6}  {'loss_A':>8}  {'loss_B':>8}  {'loss_C':>8}")
    print("-" * 72)
    for theta, _, m in angle_data:
        dz = half_h_mm * np.sin(theta)
        print(f"{theta:>11.2f}  {np.degrees(theta):>10.1f}  {dz:>7.1f}  "
              f"{m['content_frac']*100:>5.1f}%  "
              f"{m['loss_A']:>8.4f}  {m['loss_B']:>8.4f}  {m['loss_C']:>8.4f}")


if __name__ == '__main__':
    main()