"""Online_Finetuning_Backbone — scan-level self-supervised fine-tuning.

Training data flow (one step = one complete scan)
--------------------------------------------------
1. Backbone processes all N-1 adjacent frame pairs  → fake_gaps  (N-1, 6)
2. dof_to_series reconstructs predicted positions   → fake_series (N, 3, 3)
3. my_utils.reco reconstructs the 3-D volume        → volume (D, H', W')
4. custom_criterion defines the self-supervised loss on the volume.

Memory strategy
---------------
Two techniques are combined to keep GPU memory under control:

A) Gradient checkpointing (backbone)
   The backbone is called in small chunks (backbone_chunk_size, default 8).
   Each chunk uses torch.utils.checkpoint so intermediate activations are
   NOT stored; they are recomputed during backward.  This trades a small
   compute overhead for a large activation-memory saving.
   Backbone BN runs in eval mode (frozen stats) to avoid the double-update
   artefact that occurs when BN + checkpointing interact.

B) No-grad volume reconstruction
   The 3-D volume is built with torch.no_grad() so reco() stores no
   gradients.  Gradient flow for the self-supervised loss (Loss 1) comes
   through the volume's *bounding-box origin* (bias), not through voxel
   intensities.

   Gradient chain for Loss 1 (slice–real-image similarity):
       backbone → fake_gaps → fake_series
           → bias_grad  (via get_reco_size: min of trajectory corners)
           → (gt_pos - bias_grad)  [slice extraction coordinate]
           → F.grid_sample ∂grid  [bilinear-interp gradient w.r.t. grid]
           → loss_slice

   Interpretation: the network is encouraged to predict a trajectory
   whose bounding-box origin (bias) places a known real-world scan
   position correctly inside the volume, yielding a slice that matches
   the real ultrasound image.

   ⚠ This gradient is sparse — only the trajectory frame(s) that sit
   on the bounding-box boundary receive non-zero signal.  For a denser
   gradient through voxel intensities, remove torch.no_grad() from
   _reconstruct_volume (at the cost of higher activation memory).

Config keys (all optional, in res/models/online_finetune_bk.json)
------------------------------------------------------------------
pretrained_weight      str    path to pre-trained backbone checkpoint
max_train_frames       int    maximum frames per training window (default 64)
                              if the scan has more frames a random contiguous
                              window of this length is chosen each step
backbone_chunk_size    int    pairs per backbone forward chunk (default 8)
backbone_input_scale   float  spatial downscale applied to frames BEFORE the
                              backbone (default 0.5).  Does NOT affect the
                              frames used for reco() or _find_intersecting_frame.
                              EfficientNet-B1 backward peak at 480×640:
                                  scale=1.0 → 6040 MB  (OOM on 8 GB GPU)
                                  scale=0.5 → 1672 MB  ✓
                                  scale=0.25→  550 MB  ✓
down_ratio             float  spatial downscale factor for reco() (default 1.0)

# Loss 1 — slice–real-image similarity
intersect_threshold    float  minimum fraction of a candidate frame's pixels
                              that must fall inside the volume to count as an
                              intersection (default 0.5)
max_reco_attempts      int    how many window re-samples to try when no
                              intersecting frame is found (default 5)
max_dataset_trials     int    random training-set frames to probe as fallback
                              before giving up (default 20)
criterion_scale        float  downsample ratio for the slice extracted in
                              custom_criterion (default 0.5); halving the
                              resolution cuts gradient-tensor memory ~4×
                              (~12 MiB → ~3 MiB at 480×640 full resolution)
weight_loss1           float  weight applied to loss_slice (default 1.0)
weight_loss_rot        float  weight applied to the rotation-consistency loss
                              (default 0.0; set > 0 to enable Loss 2)
rotation_max_angle     float  maximum tilt angle in radians for Loss 2 / 3
                              (default π/6 ≈ 30°)
weight_loss_jagged     float  weight applied to the jaggedness loss on the
                              rotated slice (default 0.0; set > 0 to enable
                              Loss 3).  Measured as isotropic total variation
                              of the rotated slice: TV = mean|Δrow| + mean|Δcol|
weight_loss_ssim       float  weight applied to the adjacent-slice SSIM loss
                              (default 0.0; set > 0 to enable Loss 4).
                              ⚠ Gradient only when dense_grad=True.
ssim_window_size       int    Gaussian window width for SSIM (default 11)
ssim_num_pairs         int    adjacent slice pairs evaluated per step
                              (-1 = all D-1 pairs, default -1)
dense_grad             bool   enable dense gradients through voxel intensities
                              (default False).  When True, _reconstruct_volume
                              runs with gradient tracking so the backward pass
                              receives signal through every voxel that contributes
                              to slice_vol, not just through the bounding-box
                              origin.  Gradient chain with dense_grad=True:
                                  backbone → fake_gaps → fake_series
                                      → reco() → volume voxels
                                      → get_slice / F.grid_sample ∂input
                                      → loss_slice
                              This provides a much richer training signal but
                              stores the full reco() computation graph, which
                              can be 5–20× larger than the sparse-grad path.
                              _reconstruct_volume automatically wraps reco()
                              in gradient checkpointing when dense_grad=True,
                              so reco intermediates are recomputed during
                              backward rather than stored.  This prevents the
                              OOM caused by the reco graph and backbone
                              recomputation coexisting in GPU memory during
                              backward, and eliminates progressive fragmentation
                              from differently-sized reco graphs across scans.
                              Additional recommended mitigations:
                                  down_ratio         0.25 – 0.5
                                  max_train_frames   16 – 32
                                  criterion_scale    0.25 – 0.5
                                  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Memory tuning (environment variable, not a config key)
------------------------------------------------------
Set before launching training to eliminate GPU memory fragmentation from
the repeated small alloc/free pattern in _find_intersecting_frame:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

import configs
import models
import utils
import trial.my_utils.functions as my_utils
from utils.plot_functions import read_calib_matrices


# ── Module-level helper for gradient checkpointing ────────────────────────────
# Must be a plain function (not a method) so checkpoint can pickle / re-run it.

def _ckpt_backbone_fwd(backbone, chunk):
    """Single backbone chunk forward — used as the checkpointed function.

    Args:
        backbone  nn.Module   the backbone (passed explicitly for checkpointing)
        chunk     (1, k, 2, H, W)  frame-pair batch

    Returns:
        (k, 6)  raw gap predictions
    """
    out, _ = backbone(chunk, return_feature=False)  # (1, k, 6)
    return out.squeeze(0)                           # (k, 6)


# ── Module-level SSIM helpers (used by Loss 4) ────────────────────────────────

def _gaussian_window(size, sigma, device, dtype):
    """2-D separable Gaussian kernel of shape (1, 1, size, size)."""
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-coords.pow(2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, size, 1) * g.view(1, 1, 1, size)   # outer product


def _batch_ssim(x, y, window_size=11, sigma=1.5, C1=1e-4, C2=9e-4):
    """Mean SSIM between two batches of single-channel images.

    Standard SSIM formula with a Gaussian weighting window.
    C1 = (0.01·L)², C2 = (0.03·L)² for L=1 (images in [0, 1]).

    Args:
        x, y        (N, 1, H, W)  float32 in [0, 1]
        window_size int            Gaussian kernel width; auto-clamped to
                                   the smaller of H and W (must be ≥ 3)
        sigma       float          Gaussian std
        C1, C2      float          stability constants

    Returns:
        scalar  mean SSIM in [-1, 1] over all pixels and N pairs
    """
    H, W = x.shape[-2], x.shape[-1]
    ws = min(window_size, H, W)
    if ws % 2 == 0:
        ws -= 1
    ws = max(ws, 3)
    pad = ws // 2

    k = _gaussian_window(ws, sigma, x.device, x.dtype)   # (1, 1, ws, ws)

    mu_x  = F.conv2d(x,        k, padding=pad)
    mu_y  = F.conv2d(y,        k, padding=pad)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sg_x2 = F.conv2d(x.pow(2), k, padding=pad) - mu_x2
    sg_y2 = F.conv2d(y.pow(2), k, padding=pad) - mu_y2
    sg_xy = F.conv2d(x * y,    k, padding=pad) - mu_xy

    num   = (2.0 * mu_xy + C1) * (2.0 * sg_xy + C2)
    denom = (mu_x2 + mu_y2 + C1) * (sg_x2 + sg_y2 + C2)
    return (num / denom).mean()


# ──────────────────────────────────────────────────────────────────────────────

class Online_Finetuning_Backbone(models.BaseModel):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)

        # ── Backbone ──────────────────────────────────────────────────
        self.backbone = models.online_baseline_backbone.Backbone(
            self.data_cfg.source.channel,
            self.data_cfg.target.elements - 9,
        ).to(self.device)

        pretrained_path  = configs.env.getdir(self.cfg.pretrained_weight)
        pretrained_state = torch.load(pretrained_path, map_location=self.device)
        self.backbone.load_state_dict(pretrained_state)

        # ── Optimiser / scheduler ─────────────────────────────────────
        self.optimizer = torch.optim.Adam(
            self.backbone.parameters(),
            lr=self.run.lr,
            betas=self.run.betas,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.run.step_size,
            gamma=self.run.gamma,
        )

        # ── Calibration for 3-D reconstruction ────────────────────────
        calib_path           = configs.env.getdir(self.data_cfg.paths.calib)
        T_calib_scale, _, _  = read_calib_matrices(calib_path)
        self.scale_w = float(T_calib_scale[0, 0])   # mm / pixel  (u → width)
        self.scale_h = float(T_calib_scale[1, 1])   # mm / pixel  (v → height)

        self.down_ratio = float(getattr(self.cfg, 'down_ratio', 1.0))
        self.mat_scale  = torch.eye(4, dtype=torch.float32, device=self.device)
        self.mat_scale[0, 0] = self.down_ratio
        self.mat_scale[1, 1] = self.down_ratio
        self.mat_scale[2, 2] = self.down_ratio

        # ── Memory-management parameters ──────────────────────────────
        # backbone_chunk_size: pairs per backbone forward (gradient checkpointing)
        self.backbone_chunk_size = int(getattr(self.cfg, 'backbone_chunk_size', 8))
        # max_train_frames: maximum frames used per training step; longer scans
        # are randomly windowed.  Set to 0 or None to use the full scan.
        self.max_train_frames    = int(getattr(self.cfg, 'max_train_frames', 64) or 0)

        # ── Gradient clipping ──────────────────────────────────────────
        # reco()'s _get_weight uses softmax(w / T=0.001) whose Jacobian has
        # magnitude ~1/T = 1000, which causes gradient explosion without
        # clipping.  grad_clip_norm caps the L2 norm of all backbone
        # parameter gradients before each optimizer.step().
        # Set to 0 to disable clipping (not recommended with dense_grad=True).
        self.grad_clip_norm = float(getattr(self.cfg, 'grad_clip_norm', 1.0) or 0.0)

        # ── Scan context (populated by train(), consumed by custom_criterion) ──
        # Holds the full-scan tensors and the current training-window index set.
        self._scan_ctx: dict = {}

        # ── Consistent loss-key tracking ──────────────────────────────────────
        # train() must return the same set of keys for every scan so that
        # BaseModel.train_return_hook can align _count (one entry per scan)
        # with each per-scan loss value via dot product.
        # Initialized to ['loss_slice'] (the base custom_criterion output);
        # updated after the first successful criterion call so that subclasses
        # returning a different set of keys (e.g. extra regularisation terms)
        # are also handled correctly.
        self._component_loss_keys: list = ['loss_slice', 'grad_norm']

    # ------------------------------------------------------------------
    # 3-D reconstruction helper
    # ------------------------------------------------------------------

    def _reconstruct_volume(self, source, series, enable_grad=False):
        """Reconstruct a 3-D intensity volume from a scan window.

        By default runs without gradient tracking (torch.no_grad) so the volume
        is used purely as a lookup table and only the bounding-box origin
        (bias) carries gradient — see module docstring (sparse-grad path).

        When enable_grad=True (dense_grad config key) the full reco()
        computation graph is kept so gradients flow through every voxel that
        contributes to the extracted slice.  bias is returned with gradient
        in this mode and does NOT need to be re-derived via get_reco_size.

        Args:
            source       (N, 1, H, W)  float32 frames in [0, 1], on self.device
                                        always pass source.detach() — we never
                                        want gradient w.r.t. pixel intensities
            series       (N, 3, 3)     world-mm positions [center, LL, LR]
                                        pass fake_series (with grad) when
                                        enable_grad=True; fake_series.detach()
                                        otherwise
            enable_grad  bool          False (default) → torch.no_grad() wrapper
                                        True            → gradient tracking kept

        Returns:
            volume  (D, H', W')  reconstructed 3-D volume
                                  requires_grad=False when enable_grad=False
                                  requires_grad=True  when enable_grad=True
            bias    (3,)         world-mm origin offset;
                                  detached when enable_grad=False,
                                  has gradient through series when enable_grad=True
        """
        def _body(source_, series_):
            # source_ and series_ are explicit args so that grad_ckpt can
            # identify which tensors need gradient tracking during recompute.
            # Non-tensor attributes (scale_w, scale_h, mat_scale, down_ratio)
            # are captured from self via closure — that is safe.

            # ── Volume-size guard ─────────────────────────────────────────
            # A degenerate predicted trajectory (e.g. accumulated backbone
            # errors early in fine-tuning) can produce a bounding box whose
            # voxel count overflows torch's 32-bit numel limit (~2.1 B),
            # raising "RuntimeError: numel: integer multiplication overflow"
            # inside reco()'s torch.meshgrid call.
            # Pre-checking with get_reco_size is cheap (no GPU allocation)
            # and lets train() catch the error and retry with a new window.
            reco_size_chk, _ = my_utils.get_reco_size(series_, self.mat_scale)
            n_voxels = 1
            for s in reco_size_chk:
                n_voxels *= int(s)
            max_voxels = int(getattr(self.cfg, 'max_reco_voxels', 500_000_000))
            if n_voxels > max_voxels:
                raise ValueError(
                    f'Predicted volume too large: {n_voxels:,} voxels '
                    f'(reco_size={[int(s) for s in reco_size_chk]}, '
                    f'max={max_voxels:,}). '
                    f'Trajectory is likely degenerate; will retry.'
                )

            source_down = F.interpolate(
                source_,
                scale_factor=self.down_ratio,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)                   # (N, H·dr, W·dr)

            volume, bias = my_utils.reco(
                source_down, series_,
                self.scale_w, self.scale_h,
                self.mat_scale,
            )

            if self.down_ratio != 1.0:
                volume = F.interpolate(
                    volume.unsqueeze(0).unsqueeze(0),
                    scale_factor=1.0 / self.down_ratio,
                ).squeeze(0).squeeze(0)

            return volume, bias

        if enable_grad:
            # Gradient checkpointing for reco() is critical when dense_grad=True.
            #
            # Without it, reco() stores its full computation graph (proportional
            # to N × H × W × volume_size) throughout the entire backward pass.
            # When backbone gradient checkpointing then RECOMPUTES backbone
            # activations (in the same backward call), both graphs must coexist
            # in GPU memory simultaneously — causing OOM on typical GPUs.
            #
            # With checkpointing: reco() intermediates are discarded after
            # forward and recomputed on-demand during backward, so they are
            # freed before backbone recomputation begins.  This also eliminates
            # the progressive memory fragmentation from repeated alloc/free of
            # differently-sized reco graphs across scans.
            return grad_ckpt(_body, source, series, use_reentrant=False)
        else:
            with torch.no_grad():
                return _body(source, series)

    # ------------------------------------------------------------------
    # Backbone inference
    # ------------------------------------------------------------------

    def _run_backbone(self, source, use_checkpoint=True):
        """Run backbone on all N-1 adjacent frame pairs.

        During training (use_checkpoint=True) each chunk is wrapped in
        torch.utils.checkpoint so intermediate activations are discarded and
        recomputed during backward — trading compute for activation memory.

        During evaluation (use_checkpoint=False) the outer torch.no_grad()
        context (set by Main.test) already prevents activation storage.

        Args:
            source          (N, 1, H, W)  float32 frames in [0, 1]
            use_checkpoint  bool          enable gradient checkpointing

        Returns:
            fake_gaps  (N-1, 6)  predicted gaps [tx,ty,tz, rx,ry,rz] (descaled)

        Memory note — backbone_input_scale
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        EfficientNet-B1 backward on (chunk_size, 2, 480, 640) requires ~6 GB
        of intermediate activations, which exceeds a 7.6 GB GPU when combined
        with source_full and optimizer states.  backbone_input_scale (default
        0.5) downsamples frames to (chunk_size, 2, 240, 320) before the
        backbone, cutting backward peak from 6040 MB → 1672 MB.  The reco()
        call and _find_intersecting_frame always use the original 480×640
        frames, so reconstruction quality is unaffected.
        """
        N = source.shape[0]

        # Spatial downscale for backbone input only.
        # reco() uses the original-resolution source; only the backbone sees
        # the downsampled version.
        bb_scale = float(getattr(self.cfg, 'backbone_input_scale', 0.5))

        chunk_outputs = []

        for start in range(0, N - 1, self.backbone_chunk_size):
            end = min(start + self.backbone_chunk_size, N - 1)

            # Build pair chunk on-the-fly to avoid allocating the full pairs tensor
            s0 = source[start:end]           # (k, 1, H, W)
            s1 = source[start + 1:end + 1]  # (k, 1, H, W)
            if bb_scale != 1.0:
                s0 = F.interpolate(s0, scale_factor=bb_scale, mode='bilinear', align_corners=False)
                s1 = F.interpolate(s1, scale_factor=bb_scale, mode='bilinear', align_corners=False)
            chunk = torch.cat([s0, s1], dim=1).unsqueeze(0)   # (1, k, 2, H_s, W_s)

            if use_checkpoint:
                # Recompute activations during backward — saves ~90 % of backbone memory
                out = grad_ckpt(
                    _ckpt_backbone_fwd, self.backbone, chunk,
                    use_reentrant=False,
                )                          # (k, 6)
            else:
                out, _ = self.backbone(chunk, return_feature=False)
                out = out.squeeze(0)       # (k, 6)

            chunk_outputs.append(out)

        raw_gaps  = torch.cat(chunk_outputs, dim=0)         # (N-1, 6)

        # Descale rotation angles (training target stores angles × 100)
        fake_gaps = torch.cat(
            [raw_gaps[:, :3], raw_gaps[:, 3:] / 100.0],
            dim=-1,
        )                                                    # (N-1, 6)
        return fake_gaps

    # ------------------------------------------------------------------
    # Intersection helpers (used by custom_criterion)
    # ------------------------------------------------------------------

    def _overlap_fraction(self, series_pos, volume, bias, H, W):
        """Fraction of a scan plane that falls inside the volume bounding box.

        Uses a coarse 20 × 20 CPU grid instead of the full (H × W) GPU mesh.
        This avoids allocating multi-megabyte tensors on the GPU inside the
        frame-search loop and eliminates the memory fragmentation that causes
        CUDA OOM errors.

        The grid covers the same (H × W) pixel extent as the real image,
        but is sampled at 1/20 density.  For the purpose of the binary
        inside/outside test this gives essentially the same fraction as the
        full grid.

        Args:
            series_pos  (1, 3, 3)    one scan position in world-mm
                                     [center, lower-left, lower-right]
            volume      (D, H', W')  reconstructed volume (shape used only)
            bias        (3,)         world-mm origin offset from reco()
            H, W        int          image height / width in pixels

        Returns:
            float  fraction of coarse-grid points inside the volume, in [0, 1]
        """
        GRID = 20

        # Move everything to CPU — avoids any GPU allocation in this function
        pos   = series_pos.detach().cpu().float()   # (1, 3, 3)
        b     = bias.detach().cpu().float()          # (3,)
        pos_vol = pos - b.view(1, 1, 3)             # volume-mm coords (1, 3, 3)

        # Coarse pixel grid — same span as the full image, fewer sample points
        xs = torch.linspace(-(H - 1) / 2, (H - 1) / 2, GRID, dtype=torch.float32)
        ys = torch.linspace(-(W - 1) / 2, (W - 1) / 2, GRID, dtype=torch.float32)
        gx, gy = torch.meshgrid(xs, ys, indexing='ij')   # (GRID, GRID)

        # Local mm offset vectors in the probe frame (matching my_utils.transform)
        # dim-0 = mesh_y * scale_w,  dim-1 = -mesh_x * scale_h,  dim-2 = 0
        local = torch.stack(
            [gy * self.scale_w, -gx * self.scale_h, torch.zeros_like(gx)],
            dim=-1,
        )   # (GRID, GRID, 3)

        # Rotate local offsets by probe-frame axes, then add the centre position
        axis   = my_utils.get_axis(pos_vol).permute(0, 2, 1)  # (1, 3, 3) — CPU
        center = pos_vol[:, 0, :]                              # (1, 3)
        mesh   = torch.einsum('ij,HWj->HWi', axis[0], local) + center.view(1, 1, 3)
        # mesh: (GRID, GRID, 3) — 3-D world/volume positions of each grid point

        vs   = volume.shape   # (D, H', W')
        in_b = (
            (mesh[..., 0] >= 0) & (mesh[..., 0] < vs[0]) &
            (mesh[..., 1] >= 0) & (mesh[..., 1] < vs[1]) &
            (mesh[..., 2] >= 0) & (mesh[..., 2] < vs[2])
        )
        return in_b.float().mean().item()

    def _find_intersecting_frame(self, volume, bias, H, W):
        """Search for a real frame whose GT position intersects the volume.

        Searches two candidate pools in priority order:

        1. **Current-scan outside-window frames** — frames of the *same* full
           scan that were not included in the training window.  These share the
           same anatomy and have the highest chance of intersection.

        2. **Training-dataset fallback** — random frames from ``self.dataset``
           (training scans), probed up to ``max_dataset_trials`` times.

        A frame is accepted as soon as its overlap fraction (see
        ``_overlap_fraction``) is ≥ ``intersect_threshold``.

        Config keys read:
            intersect_threshold  (default 0.5)
            max_dataset_trials   (default 20)

        Args:
            volume  (D, H', W')  reconstructed 3-D volume
            bias    (3,)          world-mm origin offset
            H, W    int           image dimensions (pixels)

        Returns:
            (real_image, gt_pos, overlap_frac, source_tag)  if a frame is found
                real_image   (1, 1, H, W)  float32 in [0, 1]
                gt_pos       (1, 3, 3)     GT world-mm position
                overlap_frac float         fraction of pixels inside volume
                source_tag   str           one of ``'window'``, ``'out_of_window'``,
                                           or ``'training_dataset'``
            None  if no qualifying frame is found within the search budget

        Search order
        ~~~~~~~~~~~~
        **Pool 1a — non-anchor window frames** (preferred):
            The anchor frame (``window_start``) is excluded because
            ``fake_series[0] == gt_series[window_start]`` exactly, making the
            volume slice at that GT position identical to the source image
            (trivial loss, zero gradient).  All other window frames are valid
            candidates.  Search starts from the **middle of the window** and
            spreads outward — middle-to-end first (larger accumulated prediction
            error, non-trivial loss), then middle-to-anchor+1 (smaller error
            but guaranteed intersection due to proximity to the predicted
            trajectory).

        **Pool 1b — outside-window frames of the same scan**:
            These extend the trajectory beyond the training window.  Due to
            accumulated prediction error the predicted volume may not align
            with the real trajectory beyond the window, so overlap is not
            guaranteed.  Searched sequentially.

        **Pool 2 — random training-dataset frames** (fallback):
            Probed randomly up to ``max_dataset_trials`` times.
        """
        threshold     = float(getattr(self.cfg, 'intersect_threshold', 0.5))
        max_ds_trials = int(getattr(self.cfg, 'max_dataset_trials', 20))
        ctx           = self._scan_ctx

        source_full    = ctx['source_full']         # (N_full, 1, H, W)
        gt_series_full = ctx['gt_series_full']      # (N_full, 3, 3)
        window_set     = ctx['window_indices_set']
        win_start      = ctx['window_start']        # absolute index of anchor frame
        win_size       = ctx['window_size']         # number of frames in window
        N_full         = source_full.shape[0]

        # ── Pool 1a: non-anchor window frames ─────────────────────────
        # Build the search order: start from the middle of the window,
        # go toward the end first (more accumulated error → non-trivial loss),
        # then back toward anchor+1 (less error but guaranteed intersection).
        #
        # Example for win_start=10, win_size=8 (frames 10..17):
        #   mid = 14  →  [14,15,16,17, 13,12,11]   (anchor 10 excluded)
        win_end  = win_start + win_size                    # exclusive
        mid      = win_start + win_size // 2
        # Guard: when win_size <= 1, mid == win_start (anchor), so the first
        # range would include the anchor.  max(..., win_start+1) prevents that.
        pool_1a  = (
            list(range(max(mid, win_start + 1), win_end))  # mid → end  (anchor-safe)
            + list(range(win_start + 1, mid))[::-1]        # mid-1 → anchor+1
        )

        with torch.no_grad():
            for idx in pool_1a:
                pos  = gt_series_full[idx:idx + 1]          # (1, 3, 3)
                frac = self._overlap_fraction(pos, volume, bias, H, W)
                if frac >= threshold:
                    return source_full[idx:idx + 1], pos, frac, 'window'

        # ── Pool 1b: outside-window frames of the current scan ─────────
        with torch.no_grad():
            for idx in range(N_full):
                if idx in window_set:
                    continue
                pos  = gt_series_full[idx:idx + 1]
                frac = self._overlap_fraction(pos, volume, bias, H, W)
                if frac >= threshold:
                    return source_full[idx:idx + 1], pos, frac, 'out_of_window'

        # ── Pool 2: random training-dataset frames ─────────────────────
        with torch.no_grad():
            for _ in range(max_ds_trials):
                d_idx  = torch.randint(
                    self.dataset.trainset_length, (1,),
                    dtype=torch.long, device=self.device,
                )
                r_data = self.dataset[d_idx[0]][0]
                r_src  = r_data['source'].to(self.device)   # (N, 1, H, W)
                r_tgt  = r_data['target'].to(self.device)   # (N, 15)
                r_ser  = r_tgt[:, -9:].view(-1, 3, 3)       # (N, 3, 3)

                f_idx  = torch.randint(r_ser.shape[0], (1,)).item()
                pos    = r_ser[f_idx:f_idx + 1]             # (1, 3, 3)
                frac   = self._overlap_fraction(pos, volume, bias, H, W)
                if frac >= threshold:
                    return r_src[f_idx:f_idx + 1], pos, frac, 'training_dataset'

        return None  # no qualifying frame found

    # ------------------------------------------------------------------
    # Self-supervised loss — Loss 1 + extension hooks
    # ------------------------------------------------------------------

    def custom_criterion(self, source, fake_gaps, fake_series, volume, bias):
        """Self-supervised loss based on slice–real-image similarity (Loss 1).

        Algorithm
        ~~~~~~~~~
        1. Search for a real-world scan frame (from outside the training
           window or from the training dataset) whose GT position intersects
           the reconstructed 3-D volume.
        2. If none found → return ``None`` (signals ``train()`` to resample
           the training window and retry).
        3. Extract a 2-D slice from the volume at the found GT position.
        4. Compare the slice with the corresponding real ultrasound frame
           → **Loss 1** (``loss_slice``).

        Gradient path (with volume built under torch.no_grad)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``bias`` returned by ``_reconstruct_volume`` is detached.  To enable
        backprop, ``bias_grad`` is re-derived *with gradient tracking* from
        ``fake_series`` via ``my_utils.get_reco_size``:

            backbone → fake_gaps → fake_series
                → bias_grad  (torch.min of trajectory corners, sparse grad)
                → gt_pos - bias_grad  (slice extraction coordinate in vol)
                → F.grid_sample ∂grid
                → loss_slice

        Only the trajectory frame(s) sitting on the volume boundary receive
        non-zero gradient through ``bias_grad``.  For a denser gradient
        through voxel intensities, remove ``torch.no_grad()`` from
        ``_reconstruct_volume`` (higher memory cost).

        Extending with additional losses
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Subclass ``Online_Finetuning_Backbone`` and override this method,
        or append entries to the returned dict.  All values are summed in
        ``train()`` to form the total loss.  Assign different weights via
        config keys (e.g. ``weight_loss1``, ``weight_loss2``, …).

        Example::

            def custom_criterion(self, source, fake_gaps, fake_series, volume, bias):
                losses = super().custom_criterion(...)
                if losses is None:
                    return None
                # Loss 2: trajectory smoothness
                losses['loss_smooth'] = (
                    F.mse_loss(fake_gaps[1:], fake_gaps[:-1])
                    * float(getattr(self.cfg, 'weight_loss2', 0.1))
                )
                return losses

        Args:
            source      (N, 1, H, W)   raw input frames in [0, 1]  (with grad)
            fake_gaps   (N-1, 6)       predicted 6-DoF transforms
                                        [tx, ty, tz, rx, ry, rz] (descaled, with grad)
            fake_series (N, 3, 3)      predicted probe positions in world-mm
                                        [center, lower-left, lower-right]  (with grad)
            volume      (D, H', W')    reconstructed 3-D volume  (NO grad — reference only)
            bias        (3,)           world-mm origin offset from reco() (detached)

        Returns:
            dict[str, Tensor]  {loss_name: scalar_tensor, ...}
                All values are summed to form the total loss in ``train()``.
            None
                Signals that no qualifying intersecting frame was found;
                ``train()`` will resample the window and retry.
        """
        H, W = source.shape[-2], source.shape[-1]

        # ── 1. Find an intersecting real frame ────────────────────────
        result = self._find_intersecting_frame(volume, bias, H, W)
        if result is None:
            return None   # caller (train) resamples the window

        real_image, gt_pos, overlap_frac, _src_tag = result  # (1,1,H,W), (1,3,3), float, str

        # ── 2. Volume-origin for coordinate transformation ────────────
        dense_grad = bool(getattr(self.cfg, 'dense_grad', False))
        if dense_grad:
            # bias from _reconstruct_volume already carries gradient through
            # fake_series (reco graph was kept).  Use it directly — no need
            # to recompute via get_reco_size.
            # Gradient now flows through TWO paths simultaneously:
            #   Path A (coordinate): bias → gt_pos_vol → grid coords → loss
            #   Path B (intensity):  fake_series → reco → volume voxels → loss
            bias_origin = bias
        else:
            # Sparse path (default): volume has no gradient.
            # Re-derive a differentiable bias from fake_series so at least
            # the coordinate path (Path A) carries gradient signal.
            _, bias_origin = my_utils.get_reco_size(fake_series, self.mat_scale)
            # bias_origin: (3,)  — gradient via torch.min of trajectory corners

        # ── 3. Extract slice from volume at the GT position ───────────
        # criterion_scale reduces the slice resolution to cut GPU memory.
        # At 0.5× scale the three gradient tensors inside get_slice drop from
        # ~12 MiB to ~3 MiB (at 480×640 full resolution), eliminating the
        # "Tried to allocate 12.00 MiB" OOM that occurs in the retry loop.
        crit_scale = float(getattr(self.cfg, 'criterion_scale', 1))
        H_c = max(1, int(H * crit_scale))
        W_c = max(1, int(W * crit_scale))

        # gt_pos - bias_origin: shifts the GT world-mm position into
        # volume-mm coordinates using the differentiable origin.
        gt_pos_vol = gt_pos - bias_origin.view(1, 1, 3)  # (1, 3, 3), grad via bias_origin
        # scale / crit_scale: each pixel now spans more mm so the extracted
        # slice covers the *same physical probe area* as the full (H × W)
        # image, just at lower sampling density.  Without this correction the
        # slice would cover only the central (crit_scale)² fraction of the
        # probe — a zoomed-in centre crop that does not match real_small.
        slice_vol  = my_utils.get_slice(
            volume, gt_pos_vol, (H_c, W_c),
            scale_h=self.scale_h / crit_scale,
            scale_w=self.scale_w / crit_scale,
        )                                                  # (1, 1, 1, H_c, W_c)
        slice_vol  = slice_vol.squeeze(0)                  # (1, 1, H_c, W_c)

        # Downsample real image to match the reduced slice resolution
        real_small = F.interpolate(
            real_image, size=(H_c, W_c), mode='bilinear', align_corners=False,
        )                                                  # (1, 1, H_c, W_c)

        # ── 4. Loss 1: slice – real-image L1 similarity ───────────────
        weight_loss1 = float(getattr(self.cfg, 'weight_loss1', 1.0))
        loss_slice   = F.l1_loss(slice_vol, real_small) * weight_loss1

        # # ╔══════════════════════════════════════════════════════════════════╗
        # # ║  DEBUG: criterion visualisation — comment out for training      ║
        # # ╚══════════════════════════════════════════════════════════════════╝
        # self._debug_visualise_criterion(
        #     volume        = volume,
        #     bias          = bias,
        #     fake_series   = fake_series.detach(),
        #     source        = source.detach(),
        #     gt_pos        = gt_pos,
        #     real_image    = real_small.detach(),           # same resolution as slice_vol
        #     slice_vol     = slice_vol.detach(),
        #     overlap_frac  = overlap_frac,
        #     loss_val      = loss_slice.item() / weight_loss1,   # unweighted L1
        #     source_tag    = _src_tag,
        #     epoch_info    = self._scan_ctx.get('epoch_info'),
        # )
        # # ╚══════════════════════════════════════════════════════════════════╝

        losses = {'loss_slice': loss_slice}

        # ── 5. Loss 2 + Loss 3: rotation losses ──────────────────────────────
        # Both losses share the same rotated slice; compute together when either
        # weight is non-zero.
        weight_loss_rot    = float(getattr(self.cfg, 'weight_loss_rot',    0.0))
        weight_loss_jagged = float(getattr(self.cfg, 'weight_loss_jagged', 0.0))
        if weight_loss_rot > 0 or weight_loss_jagged > 0:
            losses.update(self._rotation_consistency_loss(
                volume, bias_origin, gt_pos, real_small, H, W, H_c, W_c, crit_scale,
            ))

        # ── 6. Loss 4: adjacent trajectory-frame SSIM ────────────────────────
        # Extracts volume slices at consecutive predicted scan positions and
        # penalises their dissimilarity, encouraging volumetric smoothness
        # along the actual scan direction.
        weight_loss_ssim = float(getattr(self.cfg, 'weight_loss_ssim', 0.0))
        if weight_loss_ssim > 0:
            losses['loss_ssim'] = self._volume_ssim_loss(
                volume, bias_origin, fake_series, H, W, crit_scale, weight_loss_ssim,
            )

        return losses

    # ------------------------------------------------------------------
    # Volume SSIM loss helper (Loss 4)
    # ------------------------------------------------------------------

    def _volume_ssim_loss(self, volume, bias_origin, fake_series, H, W, crit_scale, weight):
        """Loss 4: 1 − mean SSIM between adjacent trajectory-frame slices.

        Instead of axis-aligned volume planes, slices are extracted at the
        **predicted probe positions** (fake_series), so adjacent pairs
        correspond to consecutive frames along the actual scan direction.
        High SSIM between adjacent slices encourages a smooth, consistent
        reconstruction along the trajectory.

        Loss 4 = weight × (1 − mean_SSIM)

        Slice extraction
        ~~~~~~~~~~~~~~~~
        For trajectory indices a and b = a+1 (or a random subset):

            pos_vol = fake_series[a] − bias_origin      (volume-mm coords)
            slice   = get_slice(volume, pos_vol, (H_c, W_c), ...)

        The slices are at the same physical orientation as the real US frames
        (oblique planes matching the probe geometry), not axis-aligned.

        Gradient paths
        ~~~~~~~~~~~~~~
        **Sparse path** (dense_grad=False, default):
            volume is detached.  Gradient flows through the *coordinates*:
                fake_series[a] → series_vol → get_slice grid → SSIM → loss
                bias_origin    → series_vol → get_slice grid → SSIM → loss
            Signal: adjust predicted positions so adjacent frames' volume
            content is similar (smooth trajectory).

        **Dense path** (dense_grad=True):
            Additional gradient through voxel intensities:
                fake_series → reco → volume voxels → SSIM → loss

        Config keys consumed
        ~~~~~~~~~~~~~~~~~~~~
        ssim_window_size  int    Gaussian window width (default 11)
        ssim_num_pairs    int    frame pairs evaluated per step
                                  (-1 = all N-1 pairs, default -1).
                                  Reduce (e.g. 16) to cap memory cost.

        Args:
            volume       (D, H', W')   reconstructed 3-D volume
            bias_origin  (3,)           differentiable vol-mm origin
            fake_series  (N, 3, 3)     predicted probe positions (world-mm)
            H, W         int            full-resolution image dimensions
            crit_scale   float          criterion downscale factor
            weight       float          weight_loss_ssim read by caller

        Returns:
            scalar Tensor  weighted (1 − mean_SSIM)
        """
        N = fake_series.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=volume.device, dtype=volume.dtype)

        window_size = int(getattr(self.cfg, 'ssim_window_size', 11))
        num_pairs   = int(getattr(self.cfg, 'ssim_num_pairs',   -1))

        H_c = max(1, int(H * crit_scale))
        W_c = max(1, int(W * crit_scale))

        # ── Select adjacent trajectory index pairs ─────────────────────
        if 0 < num_pairs < N - 1:
            perm  = torch.randperm(N - 1, device=fake_series.device)[:num_pairs]
            idx_a = perm
        else:
            idx_a = torch.arange(N - 1, device=fake_series.device)
        idx_b = idx_a + 1

        # ── Convert predicted positions to volume coordinates ──────────
        # bias_origin carries gradient in both sparse and dense paths.
        series_a_vol = fake_series[idx_a] - bias_origin.view(1, 1, 3)  # (M, 3, 3)
        series_b_vol = fake_series[idx_b] - bias_origin.view(1, 1, 3)  # (M, 3, 3)

        # ── Extract slices at trajectory positions ─────────────────────
        # get_slice returns (1, M, 1, H_c, W_c); squeeze batch dim → (M, 1, H_c, W_c)
        slices_a = my_utils.get_slice(
            volume, series_a_vol, (H_c, W_c),
            scale_h=self.scale_h / crit_scale,
            scale_w=self.scale_w / crit_scale,
        ).squeeze(0)   # (M, 1, H_c, W_c)

        slices_b = my_utils.get_slice(
            volume, series_b_vol, (H_c, W_c),
            scale_h=self.scale_h / crit_scale,
            scale_w=self.scale_w / crit_scale,
        ).squeeze(0)   # (M, 1, H_c, W_c)

        ssim_val = _batch_ssim(slices_a, slices_b, window_size=window_size)
        return (1.0 - ssim_val) * weight

    # ------------------------------------------------------------------
    # Rotation losses helper (Loss 2 + Loss 3)
    # ------------------------------------------------------------------

    def _rotation_consistency_loss(
        self, volume, bias_origin, gt_pos, real_small, H, W, H_c, W_c, crit_scale,
    ):
        """Weighted rotation losses sharing a single rotated slice.

        Computes Loss 2 and/or Loss 3 from the same rotated-frame slice so
        that ``get_slice`` is called only once regardless of which losses are
        enabled.

        Algorithm
        ~~~~~~~~~
        1. Derive the local frame axes (ax_x, ax_y, ax_z) from gt_pos.
        2. Sample a random tilt θ ∈ [-rotation_max_angle, +rotation_max_angle]
           around ax_x (the horizontal in-plane axis of the probe).
        3. Build a rotated frame: same center and physical size, but ax_y is
           replaced by  cos(θ)·ax_y + sin(θ)·ax_z.
        4. Extract a volume slice at the rotated position  → slice_rot.

        Loss 2 — 3-D rotation consistency (``loss_rot``)
            The original and rotated planes intersect along the line
            center + ax_x·t, which corresponds to the **center row**
            (mesh_x = 0) of both slices and of the real image.
            loss_rot = L1(center_row(slice_rot), center_row(real_small))

        Loss 3 — jaggedness of the rotated slice (``loss_jagged``)
            A consistent volume produces smooth slices at any orientation.
            Trajectory errors cause staircase artifacts (horizontal stripes)
            in oblique slices.  Isotropic total variation penalises this:
            loss_jagged = mean|Δrow(slice_rot)| + mean|Δcol(slice_rot)|

        Gradient path (sparse, same as loss_slice)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        bias_origin → gt_pos_rot_vol → get_slice / F.grid_sample ∂grid
            → loss_rot / loss_jagged

        Args:
            volume      (D, H', W')        reconstructed 3-D volume
            bias_origin (3,)               differentiable vol-mm origin
            gt_pos      (1, 3, 3)          GT world-mm position (detached)
            real_small  (1, 1, H_c, W_c)  real US frame at gt_pos
            H, W        int                full-resolution image dims
            H_c, W_c    int                criterion-scale dims
            crit_scale  float              criterion downscale factor

        Returns:
            dict[str, Tensor]  subset of {loss_rot, loss_jagged} — only keys
            whose config weight is > 0 are included, already weighted.
        """
        weight_loss_rot    = float(getattr(self.cfg, 'weight_loss_rot',    0.0))
        weight_loss_jagged = float(getattr(self.cfg, 'weight_loss_jagged', 0.0))
        max_angle          = float(getattr(self.cfg, 'rotation_max_angle', torch.pi / 6))

        # ── Frame axes from gt_pos ─────────────────────────────────────
        axis = my_utils.get_axis(gt_pos.float())  # (1, 3, 3) rows: ax_x, ax_y, ax_z
        ax_x = axis[0, 0]                          # (3,) horizontal in-plane axis
        ax_y = axis[0, 1]                          # (3,)
        ax_z = axis[0, 2]                          # (3,) normal to the frame

        # ── Random tilt around ax_x ────────────────────────────────────
        theta = (torch.rand(1, device=gt_pos.device).item() * 2.0 - 1.0) * max_angle
        dev, dt = gt_pos.device, gt_pos.dtype
        cos_t    = torch.tensor(theta, device=dev, dtype=dt).cos()
        sin_t    = torch.tensor(theta, device=dev, dtype=dt).sin()
        ax_y_rot = cos_t * ax_y + sin_t * ax_z   # rotated ax_y; ax_x unchanged

        # ── Rotated frame in world-mm (same center, same physical size) ─
        half_w = (W - 1) / 2.0 * self.scale_w
        half_h = (H - 1) / 2.0 * self.scale_h
        center = gt_pos[0, 0]                      # (3,) world-mm center
        ll_rot = center - ax_x * half_w - ax_y_rot * half_h
        lr_rot = center + ax_x * half_w - ax_y_rot * half_h
        gt_pos_rot = torch.stack([center, ll_rot, lr_rot]).unsqueeze(0)  # (1, 3, 3)

        # ── Shift into volume coordinates (gradient via bias_origin) ───
        gt_pos_rot_vol = gt_pos_rot - bias_origin.view(1, 1, 3)

        # ── Extract rotated slice (shared by Loss 2 and Loss 3) ────────
        slice_rot = my_utils.get_slice(
            volume, gt_pos_rot_vol, (H_c, W_c),
            scale_h=self.scale_h / crit_scale,
            scale_w=self.scale_w / crit_scale,
        ).squeeze(0)   # (1, 1, H_c, W_c)

        losses = {}

        # ── Loss 2: center-row consistency ─────────────────────────────
        if weight_loss_rot > 0:
            h_mid           = H_c // 2
            real_center_row = real_small[:, :, h_mid, :]   # (1, 1, W_c) — no grad
            rot_center_row  = slice_rot[:, :, h_mid, :]    # (1, 1, W_c) — grad via bias_origin
            losses['loss_rot'] = (
                F.l1_loss(rot_center_row, real_center_row.detach()) * weight_loss_rot
            )

        # ── Loss 3: jaggedness (isotropic TV of the rotated slice) ────
        # A trajectory-consistent volume yields smooth oblique slices.
        # Staircase artifacts from misaligned scan frames produce sharp
        # row-to-row discontinuities → high TV → gradient pushes the
        # predicted trajectory toward a smoother reconstruction.
        if weight_loss_jagged > 0:
            drow = (slice_rot[:, :, 1:, :] - slice_rot[:, :, :-1, :]).abs()  # (1,1,H_c-1,W_c)
            dcol = (slice_rot[:, :, :, 1:] - slice_rot[:, :, :, :-1]).abs()  # (1,1,H_c,W_c-1)
            losses['loss_jagged'] = (drow.mean() + dcol.mean()) * weight_loss_jagged

        # # ╔══════════════════════════════════════════════════════════════════╗
        # # ║  DEBUG: rotation-consistency visualisation                      ║
        # # ╚══════════════════════════════════════════════════════════════════╝
        # self._debug_visualise_rotation_consistency(
        #     volume      = volume,
        #     bias        = bias_origin.detach(),
        #     gt_pos      = gt_pos,
        #     gt_pos_rot  = gt_pos_rot,
        #     ax_x        = ax_x,
        #     real_small  = real_small,
        #     slice_rot   = slice_rot.detach(),
        #     theta       = theta,
        #     loss_rot    = losses.get('loss_rot'),
        #     loss_jagged = losses.get('loss_jagged'),
        # )
        # # ╚══════════════════════════════════════════════════════════════════╝

        return losses

    # ------------------------------------------------------------------
    # Debug visualisation for rotation consistency loss
    # ------------------------------------------------------------------

    def _debug_visualise_rotation_consistency(
        self,
        volume,
        bias,
        gt_pos,
        gt_pos_rot,
        ax_x,
        real_small,
        slice_rot,
        theta,
        loss_rot=None,
        loss_jagged=None,
    ):
        """Visualise Loss 2 (rotation consistency) and Loss 3 (jaggedness).

        Shows two blocking windows (close each to continue):

        **Window 1 — Matplotlib** (2-row layout):

        Top row (4 panels):
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        │  Real image  │ Rotated slice│  Jaggedness  │ |Diff| at    │
        │  (center row │ (center row  │  |∇slice_rot|│  center row  │
        │  highlighted)│  highlighted)│  heatmap     │  bar chart   │
        └──────────────┴──────────────┴──────────────┴──────────────┘
        Bottom row (full width):
        ┌─────────────────────────────────────────────────────────────┐
        │  1-D signal comparison along the intersection line          │
        │  real (green) vs rotated-slice (orange)                     │
        └─────────────────────────────────────────────────────────────┘

        **Window 2 — PyVista** (3-D):
        • Reconstructed volume (bone colormap, sigmoid opacity)
        • Original GT frame — green quad with real-image texture
        • Rotated frame — cyan wireframe
        • Intersection line — red line through center along ax_x

        Args:
            volume      (D, H', W')        reconstructed volume (no grad)
            bias        (3,)               world-mm origin offset (detached)
            gt_pos      (1, 3, 3)          original GT position (world-mm)
            gt_pos_rot  (1, 3, 3)          rotated position (world-mm)
            ax_x        (3,)               horizontal in-plane axis (unit vec)
            real_small  (1, 1, H_c, W_c)  real US image at criterion scale
            slice_rot   (1, 1, H_c, W_c)  rotated slice (detached)
            theta       float              rotation angle used (radians)
            loss_rot    Tensor | None      weighted Loss 2 value (for display)
            loss_jagged Tensor | None      weighted Loss 3 value (for display)
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import pyvista as pv
        from utils.plot_functions import add_series_rects

        # ── Numpy arrays ───────────────────────────────────────────────
        real_np  = real_small.squeeze().cpu().float().numpy()    # (H_c, W_c)
        rot_np   = slice_rot.squeeze().cpu().float().numpy()     # (H_c, W_c)
        H_c, W_c = real_np.shape
        h_mid    = H_c // 2

        real_row = real_np[h_mid]          # (W_c,) — Loss 2 reference
        rot_row  = rot_np[h_mid]           # (W_c,)
        diff_row = np.abs(real_row - rot_row)

        # Jaggedness map: |Δrow| + |Δcol| (Loss 3 signal)
        drow_np = np.abs(np.diff(rot_np, axis=0))   # (H_c-1, W_c)
        dcol_np = np.abs(np.diff(rot_np, axis=1))   # (H_c, W_c-1)
        # Pad to (H_c, W_c) for display (replicate last row/col)
        jagged_np = np.pad(drow_np, ((0, 1), (0, 0)), mode='edge') \
                  + np.pad(dcol_np, ((0, 0), (0, 1)), mode='edge')

        # Highlight the center row
        real_hi = real_np.copy(); real_hi[h_mid] = 1.0
        rot_hi  = rot_np.copy();  rot_hi[h_mid]  = 1.0

        theta_deg   = float(np.degrees(theta))
        rot_str     = f'loss_rot={loss_rot.item():.4f}'    if loss_rot    is not None else 'loss_rot=disabled'
        jagged_str  = f'loss_jagged={loss_jagged.item():.4f}' if loss_jagged is not None else 'loss_jagged=disabled'

        # ══════════════════════════════════════════════════════════════
        # Window 1 — Matplotlib
        # ══════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(22, 8))
        gs  = gridspec.GridSpec(
            2, 4, figure=fig,
            height_ratios=[2, 1],
            hspace=0.38, wspace=0.12,
        )

        ax_real   = fig.add_subplot(gs[0, 0])
        ax_rot    = fig.add_subplot(gs[0, 1])
        ax_jagged = fig.add_subplot(gs[0, 2])
        ax_diff   = fig.add_subplot(gs[0, 3])
        ax_sig    = fig.add_subplot(gs[1, :])   # full-width 1-D comparison

        # — Real US image (center row highlighted) —
        ax_real.imshow(real_hi, cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax_real.axhline(h_mid, color='lime', linewidth=1.5, linestyle='--', alpha=0.8)
        ax_real.set_title('Real US image\n(GT pos, center row = intersection)', fontsize=10)
        ax_real.axis('off')

        # — Rotated slice (center row highlighted) —
        ax_rot.imshow(rot_hi, cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax_rot.axhline(h_mid, color='orange', linewidth=1.5, linestyle='--', alpha=0.8)
        ax_rot.set_title(
            f'Rotated slice  θ={theta_deg:+.1f}°\n(center row = same intersection)',
            fontsize=10,
        )
        ax_rot.axis('off')

        # — Jaggedness heatmap: |∇slice_rot| (Loss 3) —
        jmax = max(jagged_np.max(), 1e-6)
        im_j = ax_jagged.imshow(jagged_np, cmap='hot', vmin=0, vmax=jmax, aspect='auto')
        ax_jagged.axhline(h_mid, color='cyan', linewidth=1.0, linestyle='--', alpha=0.7)
        ax_jagged.set_title(
            f'Jaggedness  |Δrow|+|Δcol|\n{jagged_str}',
            fontsize=10,
        )
        ax_jagged.axis('off')
        plt.colorbar(im_j, ax=ax_jagged, fraction=0.046, pad=0.02)

        # — |Diff| at center row (bar chart, Loss 2) —
        xs = np.arange(W_c)
        ax_diff.bar(xs, diff_row, color='crimson', width=1.0, linewidth=0)
        ax_diff.set_xlim(0, W_c)
        ax_diff.set_ylim(0, max(diff_row.max() * 1.1, 0.05))
        ax_diff.set_title(f'|Real−Rot| at center row\n{rot_str}', fontsize=10)
        ax_diff.set_xlabel('pixel col'); ax_diff.set_ylabel('|diff|')

        # — 1-D signal comparison (full width, Loss 2) —
        ax_sig.plot(xs, real_row, color='limegreen',  linewidth=1.5, label='Real image (center row)')
        ax_sig.plot(xs, rot_row,  color='darkorange', linewidth=1.5,
                    label=f'Rotated slice center row (θ={theta_deg:+.1f}°)')
        ax_sig.fill_between(xs, real_row, rot_row, alpha=0.2, color='crimson', label='|diff|')
        ax_sig.set_xlim(0, W_c); ax_sig.set_ylim(-0.05, 1.05)
        ax_sig.set_xlabel('pixel column (along ax_x = 3-D intersection line)')
        ax_sig.set_ylabel('intensity')
        ax_sig.set_title('Center-row 1-D comparison (Loss 2)', fontsize=10)
        ax_sig.legend(loc='upper right', fontsize=9)
        ax_sig.grid(True, alpha=0.3)

        fig.suptitle(
            f'DEBUG rotation losses   θ={theta_deg:+.1f}°   {rot_str}   {jagged_str}',
            fontsize=12,
        )
        plt.show()   # ← blocks until closed

        # ══════════════════════════════════════════════════════════════
        # Window 2 — PyVista: 3-D view
        # ══════════════════════════════════════════════════════════════
        vol_np       = volume.cpu().float().numpy()                 # (D, H', W')
        bias_cpu     = bias.cpu()                                   # (3,)
        ax_x_np      = ax_x.detach().cpu().float().numpy()          # (3,)

        # Volume-coordinate positions (subtract bias)
        gt_pos_vol     = (gt_pos.cpu()     - bias_cpu).float().numpy()   # (1, 3, 3)
        gt_pos_rot_vol = (gt_pos_rot.cpu() - bias_cpu).float().numpy()   # (1, 3, 3)
        center_vol     = gt_pos_vol[0, 0]                                 # (3,)

        # Intersection line: center ± half_w along ax_x
        ll_vol = gt_pos_vol[0, 1]; lr_vol = gt_pos_vol[0, 2]
        half_w = float(np.linalg.norm(lr_vol - ll_vol)) / 2.0
        line_a = center_vol - ax_x_np * half_w
        line_b = center_vol + ax_x_np * half_w

        real_u8 = (np.clip(real_np, 0, 1) * 255).astype(np.uint8)

        pv_title = (
            f'DEBUG rotation losses   θ={theta_deg:+.1f}°   '
            f'{rot_str}   {jagged_str}'
        )
        plotter = pv.Plotter(title=pv_title)

        # Volume rendering
        grid = pv.ImageData()
        grid.dimensions = np.array(vol_np.shape)
        grid.spacing    = (1.0, 1.0, 1.0)
        grid.point_data['Intensity'] = vol_np.flatten(order='F')
        plotter.add_volume(grid, scalars='Intensity', cmap='bone', opacity='sigmoid')

        # Original GT frame — green, textured with real image
        add_series_rects(
            plotter, gt_pos_vol, indices=[0],
            colors='green', opacity=0.15, edge_width=3,
            frames=real_u8[np.newaxis],
        )

        # Rotated frame — cyan wireframe, no fill
        add_series_rects(
            plotter, gt_pos_rot_vol, indices=[0],
            colors='cyan', opacity=0, edge_width=3,
        )

        # Intersection line — red
        line_pts  = np.array([line_a, line_b], dtype=np.float32)
        line_mesh = pv.Spline(line_pts, n_points=2)
        plotter.add_mesh(line_mesh, color='red', line_width=4)

        # Center point
        plotter.add_points(
            center_vol.reshape(1, 3).astype(np.float32),
            color='yellow', point_size=12, render_points_as_spheres=True,
        )

        plotter.add_text(
            f'green  = original GT frame (real image texture)\n'
            f'cyan   = rotated frame (θ={theta_deg:+.1f}° around ax_x)\n'
            f'red    = intersection line (Loss 2: center row)\n'
            f'yellow = shared center point\n'
            f'{rot_str}   {jagged_str}',
            position='upper_left', font_size=9, color='white',
        )
        plotter.show_axes()
        plotter.set_background('black')
        plotter.show()   # ← blocks until closed

    # ------------------------------------------------------------------
    # Debug visualisation for custom_criterion
    # ------------------------------------------------------------------

    def _debug_visualise_criterion(
        self,
        volume,
        bias,
        fake_series,
        source,
        gt_pos,
        real_image,
        slice_vol,
        overlap_frac,
        loss_val,
        source_tag='unknown',
        epoch_info=None,
    ):
        """Visualise the criterion pipeline for one training step.

        Shows two blocking windows (close each to continue):

        **Window 1 — Matplotlib** (2-row layout):

        ┌──────────────────┬──────────────────┬──────────────────┐
        │  Volume slice    │  Real US image   │  |Slice − Real|  │
        │  at GT position  │  at GT position  │  (difference)    │
        └──────────────────┴──────────────────┴──────────────────┘
        Sub-title: overlap fraction · unweighted L1 · source pool

        **Window 2 — PyVista** (3-D):
        • Reconstructed volume (bone colormap, sigmoid opacity)
        • Training-window frame positions — red wireframe quads (sparse)
        • Found intersecting GT frame — green quad with real-image texture

        This method is **not called during real training**.  Activate it by
        un-commenting the debug block inside ``custom_criterion``.

        Args:
            volume        (D, H', W')   reconstructed 3-D volume (no grad)
            bias          (3,)          world-mm origin offset (detached)
            fake_series   (N, 3, 3)     predicted window positions (detached)
            source        (N, 1, H, W)  training-window frames (detached)
            gt_pos        (1, 3, 3)     found GT world-mm position
            real_image    (1, 1, H, W)  real US frame at gt_pos  (float32 [0,1])
            slice_vol     (1, 1, H, W)  volume slice at gt_pos   (float32 [0,1])
            overlap_frac  float         fraction of pixels inside the volume
            loss_val      float         unweighted L1 value (for display)
            source_tag    str           ``'same_scan'`` or ``'training_dataset'``
            epoch_info    dict | None   for window/figure titles
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import pyvista as pv
        from utils.plot_functions import add_series_rects

        # ── Prepare numpy arrays ───────────────────────────────────────
        vol_np    = volume.cpu().float().numpy()                 # (D, H', W')
        ser_cpu   = fake_series.cpu()                            # (N, 3, 3)
        bias_cpu  = bias.cpu()                                   # (3,)
        ser_vol   = ser_cpu - bias_cpu                           # (N, 3, 3) vol-coords
        gt_vol    = gt_pos.cpu() - bias_cpu                      # (1, 3, 3) vol-coords

        slice_np  = slice_vol.squeeze().cpu().float().numpy()    # (H, W)
        real_np   = real_image.squeeze().cpu().float().numpy()   # (H, W)
        diff_np   = np.abs(slice_np - real_np)                   # (H, W)
        real_u8   = (np.clip(real_np, 0, 1) * 255).astype(np.uint8)  # for texture

        # Middle frame of the training window — used as context in the figure
        N          = ser_cpu.shape[0]
        mid_frame  = source[N // 2].squeeze().cpu().float().numpy()  # (H, W)

        # ── Build title strings ────────────────────────────────────────
        ep_str  = ''
        idx_str = ''
        if epoch_info is not None:
            ep_str  = f"epoch={epoch_info.get('epoch', '?')}  "
            idx_str = f"scan={epoch_info.get('index', '?')}  "

        metric_str = (
            f"overlap={overlap_frac:.1%}   "
            f"L1={loss_val:.4f}   "
            f"source='{source_tag}'"
        )

        # ══════════════════════════════════════════════════════════════
        # Window 1 — Matplotlib: 2-D slice comparison
        # ══════════════════════════════════════════════════════════════
        fig = plt.figure(figsize=(20, 5.5))
        gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.08)

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[3])

        # — Extracted slice from volume (at GT position) —
        ax0.imshow(slice_np, cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax0.set_title('Volume slice\n(at GT position)', fontsize=11)
        ax0.axis('off')

        # — Real ultrasound image (at same GT position) —
        ax1.imshow(real_np, cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax1.set_title(f'Real US image\n(at same position, source: {source_tag})', fontsize=11)
        ax1.axis('off')

        # — Absolute difference map —
        vmax_diff = max(float(diff_np.max()), 1e-6)
        im = ax2.imshow(diff_np, cmap='hot', vmin=0, vmax=vmax_diff, aspect='auto')
        ax2.set_title(f'|Slice − Real|\nmax={vmax_diff:.3f}   L1={loss_val:.4f}', fontsize=11)
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)

        # — Reference: middle training-window frame —
        ax3.imshow(mid_frame, cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax3.set_title(f'Training window frame [N/2]\n(context, N={N})', fontsize=11)
        ax3.axis('off')

        fig.suptitle(
            f'DEBUG custom_criterion   {ep_str}{idx_str}\n{metric_str}',
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        plt.show()   # ← blocks until closed

        # ══════════════════════════════════════════════════════════════
        # Window 2 — PyVista: 3-D volume + frame positions
        # ══════════════════════════════════════════════════════════════
        pv_title = (
            f'DEBUG criterion   {ep_str}{idx_str}\n'
            f'window={N} frames (red)   found GT frame (green)   {metric_str}'
        )
        plotter = pv.Plotter(title=pv_title)

        # Volume rendering
        grid = pv.ImageData()
        grid.dimensions = np.array(vol_np.shape)
        grid.spacing    = (1.0, 1.0, 1.0)
        grid.point_data['Intensity'] = vol_np.flatten(order='F')
        plotter.add_volume(
            grid,
            scalars='Intensity',
            cmap='bone',
            opacity='sigmoid',
        )

        # Training-window frames — red wireframe, no fill, sparse selection
        step       = max(1, N // 8)
        win_idx    = sorted(set([0] + list(range(0, N, step)) + [N - 1]))
        add_series_rects(
            plotter, ser_vol,
            indices=win_idx,
            colors='red',
            opacity=0,
            edge_width=2,
        )

        # Found GT frame — green, with real-image texture
        gt_vol_np = gt_vol.float().numpy()                       # (1, 3, 3)
        add_series_rects(
            plotter, gt_vol_np,
            indices=[0],
            colors='green',
            opacity=0.15,
            edge_width=4,
            frames=real_u8[np.newaxis],                          # (1, H, W) uint8
        )

        # Text overlay
        plotter.add_text(
            f'{metric_str}\nred = training window ({N} frames)\ngreen = found GT frame',
            position='upper_left',
            font_size=9,
            color='white',
        )
        plotter.show_axes()
        plotter.set_background('black')
        plotter.show()   # ← blocks until closed

    # ------------------------------------------------------------------
    # Training step — one scan (or window) per call
    # ------------------------------------------------------------------

    def train(self, epoch_info, sample_dict):
        """One training step on a complete scan (or a windowed sub-sequence).

        Expected sample_dict keys (batch_size=1, squeezed inside here):
            source  (1, N, 1, H, W)   float32 ultrasound frames
            target  (1, N, 15)        gaps_padded | series_flat

        Memory and retry strategy
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        * If the full scan has more than ``max_train_frames`` frames, a
          random contiguous window is selected each attempt.
        * ``custom_criterion`` returns ``None`` when no real frame with
          sufficient overlap is found.  In that case the window is re-sampled
          (up to ``max_reco_attempts`` times).  If all attempts fail, a zero
          loss is returned and the optimizer is **not** stepped.
        * Backbone runs in eval mode (frozen BN stats) so gradient
          checkpointing does not cause BatchNorm double-update.
        * Volume reconstruction runs under ``torch.no_grad()``.

        Returns:
            dict[str, Tensor]  {loss_name: detached scalar, ...}
        """
        # ── Unpack full scan (windowing deferred to retry loop) ────────
        source_full    = sample_dict['source'].to(self.device).squeeze(0)   # (N_full, 1, H, W)
        target_full    = sample_dict['target'].to(self.device).squeeze(0)   # (N_full, 15)
        gt_series_full = target_full[:, -9:].view(-1, 3, 3)                 # (N_full, 3, 3)
        N_full         = source_full.shape[0]

        self.backbone.eval()

        max_reco_attempts = int(getattr(self.cfg, 'max_reco_attempts', 5))
        losses = None

        for _attempt in range(max_reco_attempts):

            # ── Window selection ─────────────────────────────────────────
            if self.max_train_frames > 0 and N_full > self.max_train_frames:
                t0 = torch.randint(
                    0, N_full - self.max_train_frames + 1, (1,)
                ).item()
                source    = source_full[t0:t0 + self.max_train_frames]
                gt_series = gt_series_full[t0:t0 + self.max_train_frames]
                window_indices_set = set(range(t0, t0 + self.max_train_frames))
                win_start = t0
            else:
                source    = source_full
                gt_series = gt_series_full
                window_indices_set = set(range(N_full))
                win_start = 0

            # ── Store full-scan context for custom_criterion ─────────────
            # Populated here so that _find_intersecting_frame can access
            # frame metadata and _debug_visualise_criterion can access
            # epoch_info.
            #
            # window_start  — absolute index of the first frame of the window
            #                 in the full scan (= t0 after random selection,
            #                 or 0 when the whole scan fits in the window)
            # window_size   — number of frames in the window
            #
            # The *anchor frame* (absolute index = window_start) is the frame
            # whose GT pose is used as the starting point for dof_to_series,
            # so fake_series[0] == gt_series_full[window_start] exactly.
            # Querying the volume at that GT position is trivial (the volume
            # was built directly from that image) and must be excluded from
            # the criterion frame search.
            self._scan_ctx = {
                'source_full':        source_full,
                'gt_series_full':     gt_series_full,
                'window_indices_set': window_indices_set,
                'window_start':       win_start,
                'window_size':        source.shape[0],
                'epoch_info':         epoch_info,
            }

            self.optimizer.zero_grad()

            # ── 1. Predict inter-frame gaps (gradient checkpointing) ─────
            fake_gaps = self._run_backbone(source, use_checkpoint=True)    # (N-1, 6)

            # ── 2. Reconstruct predicted probe trajectory ─────────────────
            fake_series = utils.simulation.dof_to_series(
                gt_series[0:1],             # (1, 3, 3)  GT first-frame pose
                fake_gaps.unsqueeze(0),     # (1, N-1, 6)
            ).squeeze(0)                    # (N, 3, 3)  — with gradient

            # ── 3. Reconstruct 3-D volume ─────────────────────────────────
            # dense_grad=True : reco() graph is retained → gradient flows
            #                   through every voxel in the extracted slice
            # dense_grad=False: reco() runs under no_grad → gradient flows
            #                   only through the bounding-box origin (bias)
            dense_grad = bool(getattr(self.cfg, 'dense_grad', False))
            try:
                volume, bias = self._reconstruct_volume(
                    source.detach().clone(),
                    fake_series if dense_grad else fake_series.detach(),
                    enable_grad=dense_grad,
                )                                                           # (D,H',W'), (3,)
            except (ValueError, RuntimeError) as _reco_err:
                # Degenerate trajectory: either the pre-check raised ValueError
                # (too many voxels) or meshgrid raised RuntimeError (overflow).
                # Discard this window and let the retry loop pick a new one.
                del fake_gaps, fake_series
                torch.cuda.empty_cache()
                continue

            # # ╔══════════════════════════════════════════════════════════════════╗
            # # ║  DEBUG: 3-D volume visualisation — comment out for training     ║
            # # ╚══════════════════════════════════════════════════════════════════╝
            # self._debug_visualise_volume(
            #     volume, bias,
            #     fake_series.detach(),
            #     source.detach(),
            #     epoch_info,
            # )
            # # ╚══════════════════════════════════════════════════════════════════╝

            # ── 4. Self-supervised loss (retry if no intersection) ────────
            losses = self.custom_criterion(source, fake_gaps, fake_series, volume, bias)
            if losses is not None:
                # Record the component keys so that failed scans can return
                # zeros for the same keys, keeping _count aligned with every
                # loss value in loss_all for BaseModel.train_return_hook.
                self._component_loss_keys = list(losses.keys()) + ['grad_norm']

                # ── Pre-backward memory release ───────────────────────────
                # source_full (N_full × 1 × H × W, ~700 MB for long scans)
                # is no longer needed: _find_intersecting_frame has already
                # completed inside custom_criterion.  We free it here — before
                # loss.backward() — to give the CUDA allocator ~700 MB of
                # headroom for the reco() recomputation that grad_ckpt triggers
                # during backward.
                #
                # Why .clone() above matters:
                #   source.detach() shares storage with source_full (view).
                #   Without .clone(), the checkpoint would hold a reference
                #   to that shared storage, keeping all 700 MB alive even
                #   after we delete source and source_full here.
                #   source.detach().clone() gives the checkpoint an independent
                #   19 MB copy, so deleting source + source_full below truly
                #   releases the 700 MB before backward.
                del source, source_full
                self._scan_ctx.clear()
                torch.cuda.empty_cache()
                break   # valid window found; exit retry loop

            # No qualifying frame found — release GPU tensors before the next
            # attempt so that repeated alloc/free of different-sized tensors
            # does not fragment the reserved-but-unallocated pool.
            del fake_gaps, fake_series, volume, bias
            torch.cuda.empty_cache()

        if losses is None:
            # All window re-sampling attempts failed; skip gradient step.
            # Still clear the scan context so source_full is released now
            # rather than persisting until the next call to train().
            self._scan_ctx.clear()
            torch.cuda.empty_cache()
            # Return zeros for ALL component loss keys so that every scan
            # contributes exactly one entry to each key in loss_all.
            # BaseModel.train_return_hook does `_count @ value / _count_sum`
            # which requires len(_count) == len(value) for every key.
            # Returning only {'loss': 0.0} while successful scans also include
            # 'loss_slice' (and any other custom_criterion keys) would cause
            # a "inconsistent tensor size" RuntimeError at epoch end.
            zero = torch.tensor(0.0, device=self.device)
            return {'loss': zero, **{k: zero for k in self._component_loss_keys}}

        loss = sum(losses.values())
        loss.backward()

        # ── Gradient clipping ─────────────────────────────────────────────
        # reco()'s _get_weight contains softmax(w / T=0.001) whose gradient
        # has magnitude ~1/T = 1000×, causing gradient explosion without
        # clipping.  clip_grad_norm_ returns the UNCLIPPED total norm so we
        # can monitor it; it only modifies gradients when the norm exceeds
        # grad_clip_norm (default 1.0).  Set cfg.grad_clip_norm=0 to disable.
        if self.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.backbone.parameters(),
                max_norm=self.grad_clip_norm,
            )
        else:
            # Compute norm for monitoring without modifying gradients
            grad_norm = torch.sqrt(torch.stack([
                p.grad.detach().norm() ** 2
                for p in self.backbone.parameters()
                if p.grad is not None
            ]).sum())

        self.optimizer.step()
        self.scheduler.step(epoch_info['epoch'])

        ret = {
            'loss':      loss.detach(),
            'grad_norm': grad_norm.detach(),
            **{k: v.detach() for k, v in losses.items()},
        }

        # ── Inter-scan GPU memory cleanup ─────────────────────────────────
        # source / source_full were already released in the pre-backward
        # block above (del source, source_full + _scan_ctx.clear()), so
        # _scan_ctx.clear() here is a no-op.  It is kept for safety in case
        # a future code path reaches this point without having cleared it.
        # The remaining gradient-related tensors are explicitly deleted so
        # PyTorch can reclaim their slabs before the next scan allocates its
        # own source_full (~700 MB), preventing the brief double-allocation
        # window that fragments the CUDA cache pool.
        self._scan_ctx.clear()
        del fake_gaps, fake_series, volume, bias, losses, loss
        torch.cuda.empty_cache()

        return ret

    # ------------------------------------------------------------------
    # Test step — full scan, no gradient tracking
    # ------------------------------------------------------------------

    def test(self, epoch_info, sample_dict):
        """Evaluate on one complete scan; returns trajectory error metrics."""
        source    = sample_dict['source'].to(self.device).squeeze(0)   # (N, 1, H, W)
        target    = sample_dict['target'].to(self.device).squeeze(0)   # (N, 15)
        gt_series = target[:, -9:].view(-1, 3, 3)                      # (N, 3, 3)

        self.backbone.eval()

        # use_checkpoint=False: outer torch.no_grad() (set by Main.test)
        # already prevents activation storage
        fake_gaps   = self._run_backbone(source, use_checkpoint=False)  # (N-1, 6)
        fake_series = utils.simulation.dof_to_series(
            gt_series[0:1],
            fake_gaps.unsqueeze(0),
        ).squeeze(0)                                                     # (N, 3, 3)

        return utils.metric.get_metric(gt_series, fake_series)

    def test_return_hook(self, epoch_info, return_all):
        return_info = {
            k: np.sum(v) / epoch_info['batch_per_epoch']
            for k, v in return_all.items()
        }
        if return_info:
            self.logger.info_scalars(
                '{} Epoch: {}\t',
                (epoch_info['log_text'], epoch_info['epoch']),
                return_info,
            )
        return return_all

    # ------------------------------------------------------------------
    # Debug visualisation helper
    # ------------------------------------------------------------------

    def _debug_visualise_volume(self, volume, bias, fake_series, source, epoch_info=None):
        """Interactively visualise the reconstructed 3-D volume with PyVista.

        This method is called from train() inside a clearly-marked debug block.
        Comment out that block (or this call) before running full training.

        The window is **blocking**: training pauses until the PyVista window is
        closed.

        Args:
            volume       (D, H', W')  reconstructed 3-D volume  (CPU or GPU tensor)
            bias         (3,)         world-mm origin offset from reco()
            fake_series  (N, 3, 3)    predicted probe positions (already detached)
            source       (N, 1, H, W) input frames in [0, 1]   (already detached)
            epoch_info   dict | None  passed through for the window title

        Visual elements
        ---------------
        • Volume: bone-coloured, sigmoid opacity transfer function
        • Frame rectangles: a sparse subset of predicted probe positions
          rendered as coloured quads (via utils.plot_functions.add_series_rects)
        """
        import pyvista as pv
        from utils.plot_functions import add_series_rects

        # ── Prepare tensors ────────────────────────────────────────────
        vol_np     = volume.cpu().float().numpy()           # (D, H', W')
        series_cpu = fake_series.cpu()                      # (N, 3, 3)
        # Shift series into volume coordinate space
        series_biased = series_cpu - bias.cpu()             # (N, 3, 3)

        # Source frames for optional texture mapping: (N, H, W) uint8
        frames_np  = (source.cpu().squeeze(1).numpy() * 255).astype(np.uint8)

        N = series_cpu.shape[0]

        # ── Build title ────────────────────────────────────────────────
        title = 'DEBUG: reconstructed volume'
        if epoch_info is not None:
            ep  = epoch_info.get('epoch', '?')
            idx = epoch_info.get('index', '?')
            title = f'DEBUG  epoch={ep}  scan={idx}  N={N}  vol={vol_np.shape}'

        # ── PyVista volume ─────────────────────────────────────────────
        plotter = pv.Plotter(title=title)

        grid = pv.ImageData()
        grid.dimensions = np.array(vol_np.shape)   # (D, H', W') treated as (nx, ny, nz)
        grid.spacing    = (1, 1, 1)                # 1 mm per voxel (after scale)
        grid.point_data['Intensity'] = vol_np.flatten(order='F')
        plotter.add_volume(
            grid,
            scalars='Intensity',
            cmap='bone',
            opacity='sigmoid',
        )

        # ── Frame-position rectangles ──────────────────────────────────
        # Show first, middle, last frames + a sparse sample in between
        step    = max(1, N // 6)
        indices = sorted(set([0] + list(range(0, N, step)) + [N - 1]))
        add_series_rects(
            plotter,
            series_biased,
            indices=indices,
            colors='red',
            opacity=0,
            # frames=frames_np,
        )

        plotter.show_axes()
        plotter.set_background('black')
        plotter.show()   # ← blocks until window is closed