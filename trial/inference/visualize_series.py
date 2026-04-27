"""
Visualize gt_series vs predicted_series from infer_baseline_output/.

Produces two figures saved to infer_baseline_output/:
  1. trajectory_3d.png   – 3D center-point trajectories + every 50th frame quad
  2. trajectory_2d.png   – three 2D projections (XY / XZ / YZ) with per-axis
                           error subplot
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT_DIR = "../../infer_baseline_output"

# ── Load data ──────────────────────────────────────────────────────────────────
gt   = np.load(f"{OUT_DIR}/gt_series.npy")    # (N, 3, 3)
pred = np.load(f"{OUT_DIR}/predicted_series.npy")

N = gt.shape[0]
gt_center   = gt[:, 0, :]    # (N, 3)  world-mm centre points
pred_center = pred[:, 0, :]


# ── Helper: series → 4 corners ────────────────────────────────────────────────
def series_to_corners(s):
    """s: (N, 3, 3) → corners (N, 4, 3)  [ll, lr, ur, ul]"""
    center, ll, lr = s[:, 0], s[:, 1], s[:, 2]
    ur = 2 * center - ll
    ul = 2 * center - lr
    return np.stack([ll, lr, ur, ul], axis=1)


def add_quads(ax, series, color, alpha=0.25, step=50):
    corners = series_to_corners(series)
    for i in range(0, len(corners), step):
        quad = [corners[i]]                 # list of one (4,3) polygon
        poly = Poly3DCollection(quad, facecolors=color, edgecolors=color,
                                linewidths=0.5, alpha=alpha)
        ax.add_collection3d(poly)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – 3D trajectories + sampled frame quads
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 6))

for col, (elev, azim, title) in enumerate(
        [(20, 45, "View 1  (elev=20°, azim=45°)"),
         (10, 135, "View 2  (elev=10°, azim=135°)")]):

    ax = fig.add_subplot(1, 2, col + 1, projection='3d')

    t = np.linspace(0, 1, N)
    cmap = plt.get_cmap('Blues')

    # GT trajectory ────────────────────────────────────────────────────────
    for i in range(N - 1):
        c = cmap(0.3 + 0.7 * t[i])
        ax.plot(gt_center[i:i+2, 0], gt_center[i:i+2, 1], gt_center[i:i+2, 2],
                color=c, linewidth=1.2, zorder=2)

    # Pred trajectory ──────────────────────────────────────────────────────
    cmap_p = plt.get_cmap('Reds')
    for i in range(N - 1):
        c = cmap_p(0.3 + 0.7 * t[i])
        ax.plot(pred_center[i:i+2, 0], pred_center[i:i+2, 1], pred_center[i:i+2, 2],
                color=c, linewidth=1.2, zorder=2)

    # Sampled frame quads ──────────────────────────────────────────────────
    add_quads(ax, gt,   color='steelblue',   alpha=0.20, step=50)
    add_quads(ax, pred, color='tomato',      alpha=0.20, step=50)

    # Start / end markers ──────────────────────────────────────────────────
    ax.scatter(*gt_center[0],   color='blue',  s=60, marker='o', zorder=5, label='GT start')
    ax.scatter(*gt_center[-1],  color='blue',  s=60, marker='*', zorder=5, label='GT end')
    ax.scatter(*pred_center[0], color='red',   s=60, marker='o', zorder=5, label='Pred start')
    ax.scatter(*pred_center[-1],color='red',   s=60, marker='*', zorder=5, label='Pred end')

    ax.set_xlabel('X (mm)', labelpad=4)
    ax.set_ylabel('Y (mm)', labelpad=4)
    ax.set_zlabel('Z (mm)', labelpad=4)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)

    if col == 0:
        legend_elements = [
            Patch(facecolor='steelblue', label='GT trajectory'),
            Patch(facecolor='tomato',    label='Predicted trajectory'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

plt.suptitle("3D Frame Trajectories — GT vs Predicted", fontsize=13, y=1.01)
plt.tight_layout()
path3d = f"{OUT_DIR}/trajectory_3d.png"
plt.savefig(path3d, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path3d}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – 2D projections + per-axis error
# ══════════════════════════════════════════════════════════════════════════════
frame_idx = np.arange(N)
dist_err  = np.linalg.norm(pred_center - gt_center, axis=-1)  # (N,)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
proj_pairs = [(0, 1, 'X (mm)', 'Y (mm)', 'XY plane'),
              (0, 2, 'X (mm)', 'Z (mm)', 'XZ plane'),
              (1, 2, 'Y (mm)', 'Z (mm)', 'YZ plane')]

for col, (xi, yi, xlabel, ylabel, title) in enumerate(proj_pairs):
    ax = axes[0, col]
    ax.plot(gt_center[:, xi],   gt_center[:, yi],
            color='steelblue', linewidth=1.2, label='GT')
    ax.plot(pred_center[:, xi], pred_center[:, yi],
            color='tomato',    linewidth=1.2, label='Pred', linestyle='--')
    # Mark every 100 frames
    for k in range(0, N, 100):
        ax.annotate(str(k),
                    xy=(gt_center[k, xi], gt_center[k, yi]),
                    fontsize=6, color='steelblue', ha='center')
        ax.annotate(str(k),
                    xy=(pred_center[k, xi], pred_center[k, yi]),
                    fontsize=6, color='tomato', ha='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)

# Per-axis position error ──────────────────────────────────────────────────────
axis_labels = ['X', 'Y', 'Z']
colors_err  = ['#e74c3c', '#2ecc71', '#3498db']
for col, (ai, lbl, col_err) in enumerate(zip([0, 1, 2], axis_labels, colors_err)):
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

# Overall distance error as inset in last subplot ─────────────────────────────
ax_last = axes[1, 2]
ax_in = ax_last.inset_axes([0.55, 0.55, 0.43, 0.40])
ax_in.plot(frame_idx, dist_err, color='purple', linewidth=0.7)
ax_in.set_title(f'3D dist  (mean={dist_err.mean():.1f} mm)', fontsize=7)
ax_in.set_xlabel('frame', fontsize=6)
ax_in.tick_params(labelsize=6)

plt.suptitle("2D Projections & Per-Axis Error — GT vs Predicted", fontsize=13)
plt.tight_layout()
path2d = f"{OUT_DIR}/trajectory_2d.png"
plt.savefig(path2d, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path2d}")
