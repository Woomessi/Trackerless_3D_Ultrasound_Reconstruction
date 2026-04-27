#!/usr/bin/env python3
"""
analyse_ol.py – Post-hoc analysis of saved Online-Learning (OL) results.

Loads source_{idx}.pth / value_{idx}.pth produced by Online_My_Framework.test()
and provides two modes:

  1. Metric analysis  : table + plots of MEA/FDR/ADR/MD/SD/HD before vs. after OL
  2. 3D reconstruction: mid-slice comparison (before OL / after OL / GT),
                        with optional PyVista interactive volume rendering

Usage examples
--------------
  # Metric table + convergence curves for all scans
  python analyse_ol.py

  # Only scans 0, 5, 10
  python analyse_ol.py --idx 0 5 10

  # Metric analysis + 3D reconstruction for scan 3, with PyVista render
  python analyse_ol.py --idx 3 --recon --render3d

  # Custom directory
  python analyse_ol.py --data_dir /path/to/RecON/ --idx 0
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ── project root on path ───────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import utils.reconstruction as reconstruction

# ── defaults ───────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = os.path.join(ROOT, 'save', 'online_my_fm-hp_fm-TUS_subject', 'RecON')
DOWN_RATIO = 0.3        # must match online_my_fm.json → down_ratio
METRICS = ['MEA', 'FDR', 'ADR', 'MD', 'SD', 'HD']


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def available_indices(data_dir: str) -> List[int]:
    """Return sorted list of scan indices found in data_dir."""
    indices = []
    for fname in os.listdir(data_dir):
        if fname.startswith('value_') and fname.endswith('.pth'):
            try:
                indices.append(int(fname[len('value_'):-len('.pth')]))
            except ValueError:
                pass
    return sorted(indices)


def load_scan(data_dir: str, idx: int):
    """
    Load one scan's saved tensors.

    Returns
    -------
    source : (1, N, 1, H, W) float tensor  – raw US frames
    value  : dict with keys:
               real_gaps    (N-1, 6)
               real_series  (N, 3, 3)
               fake_gaps    list[(N-1, 6)]  length = ol_epochs+1
               fake_series  list[(N, 3, 3)] length = ol_epochs+1
               loss         list[dict]      length = ol_epochs+1
    """
    src_path = os.path.join(data_dir, f'source_{idx}.pth')
    val_path = os.path.join(data_dir, f'value_{idx}.pth')
    source = torch.load(src_path, map_location='cpu', weights_only=False)
    value  = torch.load(val_path,  map_location='cpu', weights_only=False)
    return source, value


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 – Metric analysis
# ══════════════════════════════════════════════════════════════════════════════

def _scalar(v) -> float:
    if isinstance(v, torch.Tensor):
        return v.item()
    return float(v)


def collect_metrics(data_dir: str, indices: List[int]) -> dict:
    """
    Returns a dict: metric_name → {'before': [float, …], 'after': [float, …]}
    """
    result = {m: {'before': [], 'after': []} for m in METRICS}
    for idx in indices:
        _, value = load_scan(data_dir, idx)
        loss_before = value['loss'][0]
        loss_after  = value['loss'][-1]
        for m in METRICS:
            if m in loss_before:
                result[m]['before'].append(_scalar(loss_before[m]))
                result[m]['after'].append(_scalar(loss_after[m]))
    return result


def print_metric_table(metrics: dict, indices: List[int]) -> None:
    """Print a per-scan table + aggregate statistics."""
    col_w = 10
    header = f"{'Scan':>5} " + ' '.join(
        f"{'Before':>{col_w}} {'After':>{col_w}} {'Δ':>{col_w}}"
        for _ in METRICS
    )
    metric_header = '       ' + ' '.join(
        f"{'── ' + m + ' ──':^{col_w*3+2}}" for m in METRICS
    )
    print(metric_header)
    print(header)
    print('-' * (6 + len(METRICS) * (col_w * 3 + 3)))

    for i, idx in enumerate(indices):
        row = f"{idx:>5} "
        for m in METRICS:
            if not metrics[m]['before']:
                row += ' ' * (col_w * 3 + 3)
                continue
            b = metrics[m]['before'][i]
            a = metrics[m]['after'][i]
            d = a - b
            row += f"{b:{col_w}.4f} {a:{col_w}.4f} {d:+{col_w}.4f} "
        print(row)

    print('-' * (6 + len(METRICS) * (col_w * 3 + 3)))
    # mean / std rows
    for label, key in [('Mean', None), ('Std', None)]:
        row = f"{'mean':>5} " if label == 'Mean' else f"{'std':>5} "
        for m in METRICS:
            if not metrics[m]['before']:
                row += ' ' * (col_w * 3 + 3)
                continue
            bs = np.array(metrics[m]['before'])
            as_ = np.array(metrics[m]['after'])
            ds = as_ - bs
            if label == 'Mean':
                row += f"{bs.mean():{col_w}.4f} {as_.mean():{col_w}.4f} {ds.mean():+{col_w}.4f} "
            else:
                row += f"{bs.std():{col_w}.4f} {as_.std():{col_w}.4f} {ds.std():+{col_w}.4f} "
        print(row)


def plot_metric_comparison(metrics: dict, indices: List[int], save_path: Optional[str] = None) -> None:
    """Bar chart: before vs. after for each metric (mean ± std across scans)."""
    available = [m for m in METRICS if metrics[m]['before']]
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), squeeze=False)

    for ax, m in zip(axes[0], available):
        bs = np.array(metrics[m]['before'])
        as_ = np.array(metrics[m]['after'])
        means = [bs.mean(), as_.mean()]
        stds  = [bs.std(),  as_.std()]
        bars = ax.bar(['Before', 'After'], means, yerr=stds, capsize=5,
                      color=['#4C72B0', '#DD8452'], alpha=0.85, width=0.5)
        ax.set_title(m, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        delta = as_.mean() - bs.mean()
        ax.text(0.5, 0.97, f'Δ = {delta:+.4f}', ha='center', va='top',
                transform=ax.transAxes, fontsize=9, color='green' if delta < 0 else 'red')

    fig.suptitle('Online Learning: Before vs. After (mean ± std)', fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved metric comparison → {save_path}')
    plt.show()


def plot_convergence(data_dir: str, indices: List[int],
                     metric: str = 'FDR', save_path: Optional[str] = None) -> None:
    """Plot per-epoch loss curve for each scan and their average."""
    fig, ax = plt.subplots(figsize=(8, 4))
    all_curves = []

    for idx in indices:
        _, value = load_scan(data_dir, idx)
        if metric not in value['loss'][0]:
            continue
        curve = [_scalar(d[metric]) for d in value['loss']]
        all_curves.append(curve)
        ax.plot(curve, color='steelblue', alpha=0.3, linewidth=0.8)

    if all_curves:
        min_len = min(len(c) for c in all_curves)
        arr = np.array([c[:min_len] for c in all_curves])
        ax.plot(arr.mean(axis=0), color='tomato', linewidth=2, label='Mean')

    ax.set_xlabel('Online Learning Epoch')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} convergence during Online Learning ({len(indices)} scans)')
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved convergence plot → {save_path}')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 – 3D reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def _build_mat_scale(down_ratio: float, device: torch.device) -> torch.Tensor:
    mat = torch.eye(4, dtype=torch.float32, device=device)
    mat[0, 0] = down_ratio
    mat[1, 1] = down_ratio
    mat[2, 2] = down_ratio
    return mat


def _clear_reco_cache() -> None:
    """Remove cached mesh from utils.reconstruction.reco so it is rebuilt."""
    for attr in ('reco_mesh', 'reco_size', 'bias'):
        if hasattr(reconstruction.reco, attr):
            delattr(reconstruction.reco, attr)


def build_volume(frames_NHW: torch.Tensor, series: torch.Tensor,
                 down_ratio: float = DOWN_RATIO) -> torch.Tensor:
    """
    Reconstruct a 3-D volume from US frames and a position series.

    Parameters
    ----------
    frames_NHW : (N, H, W) float tensor in [0, 1]
    series     : (N, 3, 3) world-mm position tensor
    down_ratio : spatial downsampling factor (must match training config)

    Returns
    -------
    volume : 3-D float tensor on CPU
    """
    device = frames_NHW.device
    mat_scale = _build_mat_scale(down_ratio, device)
    # Downsample frames to match the mat_scale used during training
    src_down = F.interpolate(
        frames_NHW.unsqueeze(1),        # (N, 1, H, W)
        scale_factor=down_ratio,
        mode='bilinear',
        align_corners=False,
    ).squeeze(1)                         # (N, H_d, W_d)

    _clear_reco_cache()
    volume, _ = reconstruction.reco(src_down, series.to(device), mat_scale=mat_scale)
    return volume.cpu()


def show_midslices(volumes: Dict[str, torch.Tensor],
                   idx: int, save_path: Optional[str] = None) -> None:
    """
    Plot three orthogonal mid-slices for each volume side-by-side.

    Parameters
    ----------
    volumes  : {'Before OL': tensor, 'After OL': tensor, 'GT': tensor}
    """
    labels = list(volumes.keys())
    n = len(labels)
    planes = ['Axial (XY)', 'Coronal (XZ)', 'Sagittal (YZ)']
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 9))

    def _mid_slice(vol, plane):
        v = vol.numpy()
        if plane == 0:   return v[:, :, v.shape[2] // 2]
        elif plane == 1: return v[:, v.shape[1] // 2, :]
        else:            return v[v.shape[0] // 2, :, :]

    for col, label in enumerate(labels):
        vol = volumes[label]
        for row, plane_name in enumerate(planes):
            ax = axes[row, col]
            sl = _mid_slice(vol, row)
            ax.imshow(sl, cmap='gray', origin='lower')
            if row == 0:
                ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_ylabel(plane_name if col == 0 else '')
            ax.axis('off')

    fig.suptitle(f'Scan {idx}: Mid-slice 3D reconstruction comparison', fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved mid-slice figure → {save_path}')
    plt.show()


def render_pyvista(volume: torch.Tensor, title: str = '3D US Volume') -> None:
    """Interactive PyVista volume rendering (requires pyvista)."""
    try:
        import pyvista as pv
    except ImportError:
        print('PyVista not installed – skipping 3D render. pip install pyvista')
        return

    vol_np = volume.numpy()
    grid = pv.ImageData()
    grid.dimensions = np.array(vol_np.shape)
    grid.spacing    = (1, 1, 1)
    grid.point_data['Intensity'] = vol_np.flatten(order='F')

    plotter = pv.Plotter(title=title)
    plotter.add_volume(grid, scalars='Intensity', cmap='bone', opacity='sigmoid')
    plotter.show_axes()
    plotter.set_background('black')
    plotter.show()


def analyse_reconstruction(data_dir: str, idx: int,
                            render3d: bool = False,
                            save_dir: Optional[str] = None) -> None:
    """Full reconstruction analysis for one scan."""
    print(f'\n── Reconstruction: scan {idx} ──')
    source, value = load_scan(data_dir, idx)

    # frames: (N, H, W) from source (1, N, 1, H, W)
    frames = source[0, :, 0, :, :]          # (N, H, W)

    series_gt     = value['real_series']     # (N, 3, 3)
    series_before = value['fake_series'][0]  # initial prediction
    series_after  = value['fake_series'][-1] # after OL

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames = frames.to(device)

    print('  Building GT volume …')
    vol_gt     = build_volume(frames, series_gt.to(device))
    print('  Building Before-OL volume …')
    vol_before = build_volume(frames, series_before.to(device))
    print('  Building After-OL volume …')
    vol_after  = build_volume(frames, series_after.to(device))

    volumes = {'Before OL': vol_before, 'After OL': vol_after, 'GT': vol_gt}

    save_path = None
    if save_dir:
        save_path = os.path.join(save_dir, f'reconstruction_{idx}.png')
    show_midslices(volumes, idx, save_path=save_path)

    if render3d:
        render_pyvista(vol_before, title=f'Scan {idx} – Before OL')
        render_pyvista(vol_after,  title=f'Scan {idx} – After OL')
        render_pyvista(vol_gt,     title=f'Scan {idx} – GT')


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyse saved Online-Learning results (source/value .pth files).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR,
                        help='Directory containing source_*.pth / value_*.pth')
    parser.add_argument('--idx', nargs='+', type=int, default=None,
                        help='Scan indices to analyse (default: all found)')
    parser.add_argument('--metric', default='FDR',
                        help='Metric to plot in convergence curve (default: FDR)')
    parser.add_argument('--recon', action='store_true',
                        help='Also run 3D reconstruction comparison')
    parser.add_argument('--render3d', action='store_true',
                        help='Show PyVista 3D volume render (implies --recon)')
    parser.add_argument('--save_dir', default=None,
                        help='Directory to save output figures (default: show only)')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.render3d:
        args.recon = True

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        sys.exit(f'ERROR: data_dir not found: {data_dir}')

    all_idx = available_indices(data_dir)
    if not all_idx:
        sys.exit(f'ERROR: no value_*.pth files in {data_dir}')

    indices = args.idx if args.idx is not None else all_idx
    missing = [i for i in indices if i not in all_idx]
    if missing:
        print(f'WARNING: indices not found and will be skipped: {missing}')
        indices = [i for i in indices if i in all_idx]
    if not indices:
        sys.exit('ERROR: no valid scan indices.')

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ── Part 1: metric analysis ───────────────────────────────────────────────
    print(f'\n{"═"*60}')
    print(f'  Metric analysis  ({len(indices)} scans)')
    print(f'{"═"*60}')
    metrics = collect_metrics(data_dir, indices)
    print_metric_table(metrics, indices)

    metric_fig = os.path.join(save_dir, 'metric_comparison.png') if save_dir else None
    plot_metric_comparison(metrics, indices, save_path=metric_fig)

    conv_fig = os.path.join(save_dir, f'convergence_{args.metric}.png') if save_dir else None
    plot_convergence(data_dir, indices, metric=args.metric, save_path=conv_fig)

    # ── Part 2: 3D reconstruction ─────────────────────────────────────────────
    if args.recon:
        print(f'\n{"═"*60}')
        print('  3D Reconstruction')
        print(f'{"═"*60}')
        for idx in indices:
            analyse_reconstruction(data_dir, idx, render3d=args.render3d, save_dir=save_dir)


if __name__ == '__main__':
    main()
