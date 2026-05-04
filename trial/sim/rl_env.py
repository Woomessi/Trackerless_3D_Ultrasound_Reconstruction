import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import h5py
import numpy as np
import torch.nn.functional as F
import pyvista as pv
import matplotlib.pyplot as plt
import trial.my_utils.functions as my_utils
from utils.plot_functions import (
    data_pairs_adjacent, transform_t2t, read_calib_matrices,
)
from utils.reconstruction import reco
from scipy.interpolate import CubicSpline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProbeSurface:
    """Quadratic-cross-section swept surface parameterized by
    s ∈ [0, 1]  (left boundary → contact centre at 0.5 → right boundary) and
    t ∈ [0, 1]  (first valid frame → last valid frame).

    The two boundary curves arc_vox_left_top (s=0) and arc_vox_right_top (s=1)
    are hard constraints; contact_vox (s=0.5) is the third interpolation knot
    that makes the cross-section quadratic instead of linear.

    Public API
    ----------
    evaluate(s, t) -> np.ndarray (..., 3)   point on surface
    normal(s, t)   -> np.ndarray (..., 3)   unit normal  (∂S/∂s × ∂S/∂t)
    """

    def __init__(self, left_top, contact, right_top):
        valid  = ~np.isnan(left_top[:, 0])
        idx    = np.where(valid)[0].astype(float)
        t_norm = (idx - idx[0]) / (idx[-1] - idx[0])   # map frame index → [0,1]
        self._spl_L = CubicSpline(t_norm, left_top[valid])
        self._spl_C = CubicSpline(t_norm, contact[valid])
        self._spl_R = CubicSpline(t_norm, right_top[valid])

    @staticmethod
    def _basis(s):
        """Lagrange quadratic basis through s={0, 0.5, 1} and their derivatives."""
        s    = np.asarray(s, dtype=float)
        B_L  = 2*s**2 - 3*s + 1;  dB_L = 4*s - 3   # s=0 → 1,  s=0.5 → 0,  s=1 → 0
        B_C  = -4*s**2 + 4*s;     dB_C = -8*s + 4   # s=0 → 0,  s=0.5 → 1,  s=1 → 0
        B_R  = 2*s**2 - s;        dB_R = 4*s - 1    # s=0 → 0,  s=0.5 → 0,  s=1 → 1
        return B_L, B_C, B_R, dB_L, dB_C, dB_R

    def evaluate(self, s, t):
        """Point on the surface.  s, t may be scalars or broadcastable arrays."""
        s, t = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
        B_L, B_C, B_R, *_ = self._basis(s)
        L = self._spl_L(t); C = self._spl_C(t); R = self._spl_R(t)
        return B_L[..., None]*L + B_C[..., None]*C + B_R[..., None]*R

    def normal(self, s, t):
        """Unit outward normal, computed analytically from ∂S/∂s × ∂S/∂t."""
        s, t = np.broadcast_arrays(np.asarray(s, float), np.asarray(t, float))
        B_L, B_C, B_R, dB_L, dB_C, dB_R = self._basis(s)
        L  = self._spl_L(t);    C  = self._spl_C(t);    R  = self._spl_R(t)
        dL = self._spl_L(t, 1); dC = self._spl_C(t, 1); dR = self._spl_R(t, 1)
        dS_ds = dB_L[..., None]*L  + dB_C[..., None]*C  + dB_R[..., None]*R
        dS_dt =  B_L[..., None]*dL +  B_C[..., None]*dC +  B_R[..., None]*dR
        n     = np.cross(dS_ds, dS_dt)
        norm  = np.linalg.norm(n, axis=-1, keepdims=True)
        norm  = np.where(norm < 1e-12, 1.0, norm)
        return n / norm

DATA_DIR       = '/home/wu/Documents/projects/cloned_repositories/RecON/datasets'
FILENAME_CALIB = '/home/wu/Documents/projects/cloned_repositories/RecON/datasets/calib_matrix.csv'

# ── Load data ──────────────────────────────────────────────────────────────────
with h5py.File(os.path.join(DATA_DIR, "LH_Per_L_DtP.h5"), 'r') as f:
    example_scan           = f["frames"][()]   # (N, H, W) uint8
    example_transformation = f["tforms"][()]   # (N, 4, 4) float32

# with h5py.File("/home/wu/Documents/projects/cloned_repositories/RecON/data/frames_transfs/005/RH_Per_S_PtD.h5", 'r') as f:
#     example_scan           = f["frames"][()]   # (N, H, W) uint8
#     example_transformation = f["tforms"][()]   # (N, 4, 4) float32

H_img, W_img = example_scan.shape[1], example_scan.shape[2]

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

# ── Build series (N, 3, 3): [centre, lower-left, lower-right] in world mm ─────
# T_combined[i]: pixel [u, v, 0, 1] → world mm
#   centre      pixel: (W/2, H/2, 0, 1)
#   lower-left  pixel: (1,   H,   0, 1)
#   lower-right pixel: (W,   H,   0, 1)
# Note: series is in world mm.  source is downsampled so that 1 pixel ≈ 1 mm,
# making reco() consistent with the mm-unit series without any mat_scale.
#
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

# ── Extract topmost arc endpoints (fan-shaped: narrow at top) ─────────────────
# For each frame find the topmost non-zero row, then take its leftmost and
# rightmost non-zero pixel and project to world mm via T_combined.
arc_world_left      = np.full((N, 3), np.nan, dtype=np.float32)
arc_world_right     = np.full((N, 3), np.nan, dtype=np.float32)
arc_world_left_top  = np.full((N, 3), np.nan, dtype=np.float32)  # translated to v=0
arc_world_right_top = np.full((N, 3), np.nan, dtype=np.float32)
arc_pixels          = []   # [(i, u_left, u_right, v), ...]
for i in range(N):
    frame   = example_scan[i]                         # (H, W) uint8
    nz_rows = np.where(frame.any(axis=1))[0]
    if len(nz_rows) == 0:
        continue
    top_v   = int(nz_rows[0])
    nz_cols = np.where(frame[top_v] > 0)[0]
    u_left, u_right = float(nz_cols[0]), float(nz_cols[-1])
    arc_pixels.append((i, u_left, u_right, float(top_v)))
    for u_val, arr_arc, arr_top in [
        (u_left,  arc_world_left,  arc_world_left_top),
        (u_right, arc_world_right, arc_world_right_top),
    ]:
        # arc endpoint at actual topmost row
        pix = torch.tensor([u_val, float(top_v), 0.0, 1.0], dtype=T_combined.dtype)
        w   = (T_combined[i] @ pix.unsqueeze(-1)).squeeze(-1)
        arr_arc[i] = w[:3].numpy()
        # same u, translated up to v=0 (top frame boundary) → collinear with contact_vox
        pix_top = torch.tensor([u_val, 0.0, 0.0, 1.0], dtype=T_combined.dtype)
        w_top   = (T_combined[i] @ pix_top.unsqueeze(-1)).squeeze(-1)
        arr_top[i] = w_top[:3].numpy()

print(f"Extracted arc endpoints for {len(arc_pixels)} frames")
print(f"Frame 0: top_row={int(arc_pixels[0][3])}, u_left={arc_pixels[0][1]:.0f}, u_right={arc_pixels[0][2]:.0f}")
print(f"  world_left[0]       = {arc_world_left[0]}")
print(f"  world_left_top[0]   = {arc_world_left_top[0]}")
print(f"  world_right_top[0]  = {arc_world_right_top[0]}")

# ── Prepare source frames: (N, H, W) float32 in [0, 1] ────────────────────────
source = torch.from_numpy(example_scan.astype(np.float32) / 255.0)  # (N, H, W)

# ── Downsample ──────────────────────────────────

down_ratio = 1
mat_scale = torch.eye(4, dtype=torch.float32, device=DEVICE)
mat_scale[0, 0] = down_ratio
mat_scale[1, 1] = down_ratio
mat_scale[2, 2] = down_ratio

source_down = F.interpolate(
    source.unsqueeze(1), scale_factor=down_ratio, mode='bilinear', align_corners=False
).squeeze(1)  # (N, H*dr, W*dr)

scale_w = float(tform_calib_scale[0, 0])  # u → W dimension
scale_h = float(tform_calib_scale[1, 1])  # v → H dimension

# ── Reconstruct 3-D volume ─────────────────────────────────────────────────────

volume, bias = my_utils.reco(source_down.to(DEVICE), series.to(DEVICE), scale_w, scale_h, mat_scale)
# volume, bias = my_utils.reco(source.to(DEVICE), series.to(DEVICE), scale_w, scale_h)

volume = volume.cpu()

# ── 3D volume rendering with PyVista ──────────────────────────────────────────
vol_np = volume.numpy()
grid = pv.ImageData()
grid.dimensions = np.array(vol_np.shape)
grid.spacing = (1, 1, 1)
grid.point_data["Intensity"] = vol_np.flatten(order="F")

plotter = pv.Plotter(title="3D US reconstruction")
plotter.add_volume(grid, scalars="Intensity", cmap="bone", opacity="sigmoid")

# ── World coordinate frame (origin + X/Y/Z axes) ──────────────────────────────
# voxel = world_mm - bias  →  world origin in PyVista space is at -bias
bias_np = bias.cpu().numpy()
world_origin_v = (-bias_np).tolist()
axis_len = max(vol_np.shape) * 0.15

plotter.add_mesh(
    pv.Sphere(radius=axis_len * 0.06, center=world_origin_v),
    color="white",
)
for direction, color, label in [
    ([1, 0, 0], "red",   "X"),
    ([0, 1, 0], "green", "Y"),
    ([0, 0, 1], "blue",  "Z"),
]:
    d = np.array(direction, dtype=float)
    plotter.add_mesh(
        pv.Arrow(start=world_origin_v, direction=d, scale=axis_len,
                 tip_length=0.25, tip_radius=0.10, shaft_radius=0.04),
        color=color,
    )
    plotter.add_point_labels(
        [(-bias_np + d * axis_len * 1.15).tolist()],
        [label],
        font_size=16,
        text_color=color,
        always_visible=True,
        shape=None,
        show_points=False,
    )

# ── Visualise example_transformation: frame planes + probe trajectory ──────────
stride = 50  # one frame outline per this many frames

# 4 image-corner pixels (columns = homogeneous points [u, v, 0, 1]^T)
corners_pix = torch.tensor(
    [
        [0.0,          0.0,          0.0, 1.0],  # upper-left
        [float(W_img), 0.0,          0.0, 1.0],  # upper-right
        [float(W_img), float(H_img), 0.0, 1.0],  # lower-right
        [0.0,          float(H_img), 0.0, 1.0],  # lower-left
    ],
    dtype=T_combined.dtype,
).T  # (4, 4)

corners_world = torch.bmm(
    T_combined, corners_pix.unsqueeze(0).expand(N, -1, -1)
)  # (N, 4, 4)
bias_t = torch.from_numpy(bias_np)
corners_vox = (
    corners_world[:, :3, :].permute(0, 2, 1) - bias_t  # (N, 4, 3) voxel space
).numpy()
centers_vox = (series[:, 0, :] - bias_t).numpy()  # (N, 3)

# Arc top-edge endpoints in voxel space
arc_vox_left      = arc_world_left      - bias_np  # (N, 3)
arc_vox_right     = arc_world_right     - bias_np  # (N, 3)
arc_vox_left_top  = arc_world_left_top  - bias_np  # (N, 3)  v=0 boundary
arc_vox_right_top = arc_world_right_top - bias_np  # (N, 3)

# # Probe-centre trajectory (yellow polyline)
# traj_poly = pv.PolyData()
# traj_poly.points = centers_vox
# traj_poly.lines = np.hstack([[N], np.arange(N)])
# plotter.add_mesh(traj_poly, color="yellow", line_width=2)

# example_transformation stores the tracker-sensor pose; it does NOT directly
# encode the probe-skin contact point.  The contact point is the top-centre of
# the image plane (v = 0, near-field / transducer face), which requires the
# calibration matrix to reach world mm:
#   P_contact = T_combined @ [W/2, 0, 0, 1]^T
contact_pix   = torch.tensor([W_img / 2.0, 0.0, 0.0, 1.0], dtype=T_combined.dtype)
contact_world = (T_combined @ contact_pix.unsqueeze(-1)).squeeze(-1)  # (N, 4)
contact_vox   = (contact_world[:, :3] - bias_t).numpy()               # (N, 3)
probe_surface = ProbeSurface(arc_vox_left_top, contact_vox, arc_vox_right_top)

# Contact-point trajectory (magenta) — distinct from the centre trajectory
contact_traj        = pv.PolyData()
contact_traj.points = contact_vox
contact_traj.lines  = np.hstack([[N], np.arange(N)])
plotter.add_mesh(contact_traj, color="magenta", line_width=2)

# # Arc top-edge endpoints at actual topmost row: left = cyan, right = orange
# valid = ~np.isnan(arc_vox_left[:, 0])
# plotter.add_points(arc_vox_left[valid],  color="cyan",   point_size=6, render_points_as_spheres=True)
# plotter.add_points(arc_vox_right[valid], color="orange", point_size=6, render_points_as_spheres=True)

# Same endpoints translated to v=0 (frame top boundary, collinear with contact_vox):
# left = deep sky blue, right = gold
valid_top = ~np.isnan(arc_vox_left_top[:, 0])
plotter.add_points(arc_vox_left_top[valid_top],  color="deepskyblue", point_size=8, render_points_as_spheres=True)
plotter.add_points(arc_vox_right_top[valid_top], color="gold",        point_size=8, render_points_as_spheres=True)

# Fitted bounded contact surface (left/right boundaries are hard constraints)
_ns, _nt = 40, 120
_SS, _TT = np.meshgrid(np.linspace(0, 1, _ns), np.linspace(0, 1, _nt))  # (_nt, _ns)
_pts     = probe_surface.evaluate(_SS, _TT)                               # (_nt, _ns, 3)
surf_mesh = pv.StructuredGrid(_pts[..., 0], _pts[..., 1], _pts[..., 2])
plotter.add_mesh(surf_mesh, color="cyan", opacity=0.35, show_edges=False)

# Probe rotation axes (columns of T_combined's rotation block, normalised):
#   col 0 = u-axis (image width / lateral)  → red
#   col 1 = v-axis (image depth / axial)    → green
#   col 2 = normal (out-of-plane / elevational) → blue
rot_cols  = T_combined[:, :3, :3]
axis_dirs = (rot_cols / rot_cols.norm(dim=1, keepdim=True)).numpy()  # (N, 3, 3)
probe_axis_len = float(max(vol_np.shape)) * 0.05

# First and last frame: yellow outline + textured US image
# corners order: UL(0), UR(1), LR(2), LL(3)
# UV: UL→(0,1), UR→(1,1), LR→(1,0), LL→(0,0)  [VTK: t=0 bottom, t=1 top]
# corners order: UL(0), UR(1), LR(2), LL(3)
# UL/UR: pixel v=0 (near-field)  → numpy row 0 → VTK t=0
# LL/LR: pixel v=H (far-field)   → numpy row H → VTK t=1
uv_coords = np.array([[0., 1.], [1., 1.], [1., 0.], [0., 0.]], dtype=np.float32)

for i in [0, N - 1]:
    pts = corners_vox[i]
    quad = pv.PolyData()
    quad.points = pts
    quad.lines = np.array([2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0])
    plotter.add_mesh(quad, color="yellow", line_width=2, opacity=0.9)

    img_gray = example_scan[i]                                       # (H, W) uint8
    img_rgb  = np.stack([img_gray, img_gray, img_gray], axis=-1)     # (H, W, 3) uint8
    tex      = pv.Texture(img_rgb)

    quad_tex = pv.PolyData()
    quad_tex.points = pts
    quad_tex.faces  = np.array([4, 0, 1, 2, 3])
    quad_tex.active_texture_coordinates = uv_coords
    plotter.add_mesh(quad_tex, texture=tex, opacity=0.85)

for i in range(0, N, stride):
    # Probe pose axes at the contact point (v=0 top-centre of image)
    origin = contact_vox[i]
    for j, ax_color in enumerate(["red", "green", "blue"]):
        plotter.add_mesh(
            pv.Arrow(
                start=origin,
                direction=axis_dirs[i, :, j],
                scale=probe_axis_len,
                tip_length=0.25,
                tip_radius=0.05,
                shaft_radius=0.01,
            ),
            color=ax_color,
        )

# ─── Probe surface environment (RL-compatible) ───────────────────────────────

volume_up = F.interpolate(
    volume.unsqueeze(0).unsqueeze(0), scale_factor=1.0 / down_ratio
).squeeze(0).squeeze(0)

# ── Module-level helpers ──────────────────────────────────────────────────────

def _ax_x_ref(s, t, surf=None):
    """Unit s-tangent of the probe surface projected onto the tangent plane.
    Serves as the zero-angle reference direction for ax_x."""
    _s = surf if surf is not None else probe_surface
    B_L, B_C, B_R, dB_L, dB_C, dB_R = ProbeSurface._basis(float(s))
    L  = _s._spl_L(float(t))
    Cm = _s._spl_C(float(t))
    R  = _s._spl_R(float(t))
    dS_ds = dB_L * L + dB_C * Cm + dB_R * R
    n     = _s.normal(s, t)
    ref   = dS_ds - n * np.dot(n, dS_ds)
    nrm   = np.linalg.norm(ref)
    if nrm < 1e-12:
        ref = np.array([1., 0., 0.]) - n * n[0]
        nrm = np.linalg.norm(ref)
    return (ref / nrm).astype(np.float32)


def _rotate_around(v, axis, angle):
    """Rodrigues: rotate unit vector v (⊥ axis) around unit axis by angle (rad)."""
    ca, sa = float(np.cos(angle)), float(np.sin(angle))
    return (ca * v + sa * np.cross(axis, v)).astype(np.float32)


# ── Environment ───────────────────────────────────────────────────────────────

class ProbeEnv:
    """RL environment for simulated freehand ultrasound acquisition.

    Probe pose is constrained to the curved contact surface (surf_mesh):
      origin  – on the surface, parameterised by (s, t) ∈ [0, 1]²
      y-axis  – outward surface normal  (display convention: -normal points INTO volume)
      x-axis  – lateral direction in the tangent plane   (ax_x)
      z-axis  – elevational direction                    (ax_z = normal × ax_x)

    State  : (s, t, angle)
    Action : (dx, dz, dθ) – translate along x / z (mm), rotate around normal (rad)
    Obs    : torch.Tensor (H, W) float32 – simulated US slice

    After every step the displaced origin P_new = origin + ax_x*dx + ax_z*dz
    is re-projected onto surf_mesh via L-BFGS-B minimisation so that the pose
    constraint is always satisfied.
    """

    def __init__(
        self,
        probe_surface,
        volume_up: torch.Tensor,
        scale_h: float,
        scale_w: float,
        H_img: int,
        W_img: int,
        plotter=None,
        axis_len: float = 50.0,
    ):
        self._surf  = probe_surface
        self._vol   = volume_up
        self._sh    = scale_h
        self._sw    = scale_w
        self._H     = H_img
        self._W     = W_img
        self._plt   = plotter
        self._alen  = axis_len
        self._actors: list = []

        self.s     = 0.5
        self.t     = 0.5
        self.angle = 0.0

        plt.ion()
        self._fig2d, self._ax2d = plt.subplots(figsize=(4, 3))
        self._img_hdl = self._ax2d.imshow(
            np.zeros((H_img, W_img), dtype=np.float32), cmap='gray', vmin=0, vmax=1
        )
        self._ax2d.axis('off')
        self._fig2d.tight_layout()

    # ── internal ──────────────────────────────────────────────────────────────

    def _frame(self):
        """Return (origin_np, normal_np, ax_x_np, ax_z_np) for current state.
        All arrays are float32 (3,).
        ax_z = normal × ax_x  (elevational / scanning direction)."""
        origin_np = self._surf.evaluate(self.s, self.t).astype(np.float32)
        normal_np = self._surf.normal(self.s, self.t).astype(np.float32)
        ax_x_np   = _rotate_around(
            _ax_x_ref(self.s, self.t, surf=self._surf), normal_np, self.angle
        )
        ax_z_np   = np.cross(normal_np, ax_x_np).astype(np.float32)
        return origin_np, normal_np, ax_x_np, ax_z_np

    def _get_obs(self, origin_np, normal_np, ax_x_np) -> torch.Tensor:
        """Sample the (H, W) US slice at the given pose."""
        ax_y   = torch.from_numpy(normal_np)
        ax_x   = torch.from_numpy(ax_x_np)
        origin = torch.from_numpy(origin_np)
        # Image centre is (H-1)/2 depth-steps below the contact/top-edge point
        center = origin - ax_y * (self._H - 1) / 2.0 * self._sh
        sv = torch.stack(
            [center, center - ax_x - ax_y, center + ax_x - ax_y], dim=0
        ).unsqueeze(0)
        return my_utils.get_slice(
            self._vol, sv, (self._H, self._W), scale_h=self._sh, scale_w=self._sw
        ).squeeze(0, 1, 2)   # (H, W)

    def _project(self, P_target: np.ndarray):
        """Find (s, t) ∈ [0, 1]² on surf_mesh closest to P_target."""
        from scipy.optimize import minimize
        def obj(st):
            p = self._surf.evaluate(float(st[0]), float(st[1]))
            d = p - P_target
            return float(np.dot(d, d))
        res = minimize(
            obj, [self.s, self.t], method='L-BFGS-B',
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            options={'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 200},
        )
        return float(np.clip(res.x[0], 0.0, 1.0)), float(np.clip(res.x[1], 0.0, 1.0))

    def _do_render(self, origin_np, normal_np, ax_x_np, obs: torch.Tensor) -> None:
        # 2-D matplotlib
        self._img_hdl.set_data(obs.numpy())
        self._ax2d.set_title(
            f"s={self.s:.3f}  t={self.t:.3f}  θ={np.degrees(self.angle):.1f}°",
            fontsize=9,
        )
        self._fig2d.canvas.draw_idle()
        self._fig2d.canvas.flush_events()

        if self._plt is None:
            return

        # 3-D PyVista
        for a in self._actors:
            self._plt.remove_actor(a)
        self._actors.clear()

        ax_x   = torch.from_numpy(ax_x_np)
        ax_y   = torch.from_numpy(normal_np)
        origin = torch.from_numpy(origin_np)

        # vox(h, w) = origin + ax_x*(w-(W-1)/2)*sw - ax_y*h*sh
        def _c(h, w):
            return (origin + ax_x * (w - (self._W - 1) / 2.0) * self._sw
                           - ax_y * h * self._sh).numpy()

        corners = np.stack([
            _c(0,          0          ),   # UL  near-field, left
            _c(0,          self._W - 1),   # UR  near-field, right
            _c(self._H - 1, self._W - 1),  # LR  far-field,  right
            _c(self._H - 1, 0         ),   # LL  far-field,  left
        ])

        _ol        = pv.PolyData()
        _ol.points = corners
        _ol.lines  = np.array([2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 0])
        self._actors.append(self._plt.add_mesh(_ol, color='lime', line_width=3))

        _u8  = (obs.numpy() * 255).clip(0, 255).astype(np.uint8)
        _tex = pv.Texture(np.stack([_u8] * 3, axis=-1))
        _q   = pv.PolyData()
        _q.points = corners
        _q.faces  = np.array([4, 0, 1, 2, 3])
        _q.active_texture_coordinates = np.array(
            [[0., 1.], [1., 1.], [1., 0.], [0., 0.]], dtype=np.float32
        )
        self._actors.append(self._plt.add_mesh(_q, texture=_tex, opacity=0.85))

        self._actors.append(self._plt.add_mesh(
            pv.Sphere(radius=self._alen * 0.04, center=origin_np.tolist()),
            color='lime',
        ))
        for _d, _clr in [(ax_x_np, 'red'), (-normal_np, 'limegreen')]:
            self._actors.append(self._plt.add_mesh(
                pv.Arrow(
                    start=origin_np, direction=_d, scale=self._alen * 0.5,
                    tip_length=0.25, tip_radius=0.05, shaft_radius=0.02,
                ),
                color=_clr,
            ))
        self._plt.render()

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self, s: float = 0.5, t: float = 0.5, angle: float = 0.0) -> torch.Tensor:
        """Reset probe pose.  Returns initial observation (H, W) float32."""
        self.s, self.t, self.angle = float(s), float(t), float(angle)
        o, n, x, _ = self._frame()
        obs = self._get_obs(o, n, x)
        self._do_render(o, n, x, obs)
        return obs

    def step(self, dx: float, dz: float, dtheta: float):
        """Apply a 3-DoF rigid body action in the current probe frame.

        The new origin P_new = origin + ax_x*dx + ax_z*dz is projected back
        onto surf_mesh so the pose constraint is always satisfied.

        Parameters
        ----------
        dx     : translation along ax_x  (lateral,      mm)
        dz     : translation along ax_z  (elevational,  mm)
        dtheta : rotation  around ax_y   (outward normal, rad)

        Returns
        -------
        obs   : torch.Tensor (H, W)  simulated US slice at new pose
        state : dict  {'s': float, 't': float, 'angle': float}
        """
        origin_np, normal_np, ax_x_np, ax_z_np = self._frame()

        P_new        = origin_np + ax_x_np * float(dx) + ax_z_np * float(dz)
        s_new, t_new = self._project(P_new)
        angle_new    = self.angle + float(dtheta)

        self.s, self.t, self.angle = s_new, t_new, angle_new

        o, n, x, _ = self._frame()
        obs = self._get_obs(o, n, x)
        self._do_render(o, n, x, obs)
        return obs, {'s': s_new, 't': t_new, 'angle': angle_new}

    def render(self) -> None:
        """Re-render the current state (no state change)."""
        o, n, x, _ = self._frame()
        obs = self._get_obs(o, n, x)
        self._do_render(o, n, x, obs)

    @property
    def state(self) -> dict:
        return {'s': self.s, 't': self.t, 'angle': self.angle}


# # ── Instantiate & show ────────────────────────────────────────────────────────
# env = ProbeEnv(
#     probe_surface, volume_up, scale_h, scale_w, H_img, W_img,
#     plotter=plotter, axis_len=axis_len,
# )
# _ = env.reset()
#
# plotter.show_axes()
# plotter.set_background("black")
# plotter.show()