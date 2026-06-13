"""
Interactive edge visualization for Canny2D.

Usage:
    python scripts/visualize_edge.py --h5 <path.h5> [--threshold 0.1] [--filter_size 5]

Controls:
    Left/Right arrow  : prev/next frame
    Scroll slider     : frame index
    Threshold slider  : Canny2D threshold
    Click on image    : add annotation point (saved to annotations.json)
    Press 'd'         : delete last annotation
    Press 's'         : save annotations
    Press 'c'         : clear all annotations for current frame
"""

import argparse
import json
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.layers.canny import Canny2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frames(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, 'r') as f:
        frames = f['frames'][()]   # (N, H, W) uint8
    return frames


def compute_edge(frame_np: np.ndarray, threshold: float, filter_size: int, device: torch.device) -> np.ndarray:
    """Return edge map (H, W) in [0, 1]."""
    canny = Canny2D(threshold=threshold, filter_size=filter_size).to(device)
    t = torch.from_numpy(frame_np.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, H, W)
    with torch.no_grad():
        edge = canny(t)
    return edge.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def load_annotations(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_annotations(path: str, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'[saved] {path}  ({sum(len(v) for v in data.values())} points total)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Interactive Canny edge visualizer')
    parser.add_argument('--h5', type=str,
                        default='/media/wu/Extreme SSD/datasets/11178509/train_part1/010/LH_Par_C_DtP.h5',
                        help='Path to .h5 file')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Initial Canny2D threshold (0–1 range after normalisation)')
    parser.add_argument('--filter_size', type=int, default=5,
                        help='Gaussian filter size (odd integer)')
    parser.add_argument('--annot', type=str, default='scripts/annotations.json',
                        help='Annotation save path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    print(f'[loading] {args.h5}')
    frames = load_frames(args.h5)
    N, H, W = frames.shape
    print(f'[frames] {N} × {H}×{W}')

    annot_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.annot,
    )
    annotations: dict = load_annotations(annot_path)   # { "frame_idx": [[x,y,label], ...] }

    state = {
        'frame_idx': 0,
        'threshold': args.threshold,
        'filter_size': args.filter_size,
    }

    # ---- build figure ----
    fig = plt.figure(figsize=(13, 7))
    fig.suptitle('Edge Visualizer  |  ←/→: frame   s: save   d: del last   c: clear frame', fontsize=10)

    ax_orig = fig.add_axes([0.02, 0.22, 0.45, 0.72])
    ax_edge = fig.add_axes([0.52, 0.22, 0.45, 0.72])

    ax_slider_frame = fig.add_axes([0.12, 0.12, 0.76, 0.03])
    ax_slider_thresh = fig.add_axes([0.12, 0.06, 0.76, 0.03])

    slider_frame = Slider(ax_slider_frame, 'Frame', 0, N - 1, valinit=0, valstep=1)
    slider_thresh = Slider(ax_slider_thresh, 'Threshold', 0.0, 1.0, valinit=args.threshold)

    # ---- render function ----
    scatter_orig = [None]
    scatter_edge = [None]

    def render():
        idx = state['frame_idx']
        thr = state['threshold']
        fs  = state['filter_size']

        frame = frames[idx]
        edge  = compute_edge(frame, thr, fs, device)

        ax_orig.cla()
        ax_edge.cla()

        ax_orig.imshow(frame, cmap='gray', vmin=0, vmax=255)
        ax_orig.set_title(f'Original  [frame {idx}/{N-1}]', fontsize=9)
        ax_orig.axis('off')

        ax_edge.imshow(edge, cmap='hot', vmin=0, vmax=1)
        ax_edge.set_title(f'Canny2D edge  (threshold={thr:.3f})', fontsize=9)
        ax_edge.axis('off')

        # draw annotations for this frame
        key = str(idx)
        pts = annotations.get(key, [])
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            labels = [p[2] if len(p) > 2 else '' for p in pts]
            for ax in (ax_orig, ax_edge):
                ax.scatter(xs, ys, s=40, c='cyan', edgecolors='blue', linewidths=0.8, zorder=5)
                for x, y, lbl in zip(xs, ys, labels):
                    if lbl:
                        ax.text(x + 4, y - 4, lbl, color='cyan', fontsize=7, zorder=6)

        fig.canvas.draw_idle()

    render()

    # ---- slider callbacks ----
    def on_frame_slider(val):
        state['frame_idx'] = int(slider_frame.val)
        render()

    def on_thresh_slider(val):
        state['threshold'] = float(slider_thresh.val)
        render()

    slider_frame.on_changed(on_frame_slider)
    slider_thresh.on_changed(on_thresh_slider)

    # ---- keyboard callbacks ----
    def on_key(event):
        if event.key == 'right':
            new_idx = min(state['frame_idx'] + 1, N - 1)
            state['frame_idx'] = new_idx
            slider_frame.set_val(new_idx)
        elif event.key == 'left':
            new_idx = max(state['frame_idx'] - 1, 0)
            state['frame_idx'] = new_idx
            slider_frame.set_val(new_idx)
        elif event.key == 's':
            save_annotations(annot_path, annotations)
        elif event.key == 'd':
            key = str(state['frame_idx'])
            if annotations.get(key):
                removed = annotations[key].pop()
                print(f'[deleted] frame {key}: {removed}')
                render()
        elif event.key == 'c':
            key = str(state['frame_idx'])
            if key in annotations:
                del annotations[key]
                print(f'[cleared] frame {key}')
                render()

    # ---- click annotation ----
    def on_click(event):
        if event.inaxes not in (ax_orig, ax_edge):
            return
        if event.button != 1:   # left click only
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        key = str(state['frame_idx'])
        if key not in annotations:
            annotations[key] = []
        # ask for a short label via terminal (non-blocking: use empty string if no stdin)
        try:
            label = input(f'Label for point ({x:.0f},{y:.0f}) [enter to skip]: ').strip()
        except EOFError:
            label = ''
        annotations[key].append([round(x, 1), round(y, 1), label])
        print(f'[annotated] frame {key}: ({x:.0f},{y:.0f}) "{label}"')
        render()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()


if __name__ == '__main__':
    main()