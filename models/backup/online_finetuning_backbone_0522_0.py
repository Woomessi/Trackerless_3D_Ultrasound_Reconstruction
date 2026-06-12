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
down_ratio             float  spatial downscale factor for reco() (default 1.0)

# Loss 1 — slice–real-image similarity
intersect_threshold    float  minimum fraction of a candidate frame's pixels
                              that must fall inside the volume to count as an
                              intersection (default 0.5)
max_reco_attempts      int    how many window re-samples to try when no
                              intersecting frame is found (default 5)
max_dataset_trials     int    random training-set frames to probe as fallback
                              before giving up (default 20)
weight_loss1           float  weight applied to loss_slice (default 1.0)
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


# ──────────────────────────────────────────────────────────────────────────────

class Online_Finetuning_Backbone(models.BaseModel):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)

        # ── Backbone ──────────────────────────────────────────────────
        self.backbone = models.backup.online_baseline_backbone.Backbone(
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

        # ── Scan context (populated by train(), consumed by custom_criterion) ──
        # Holds the full-scan tensors and the current training-window index set.
        self._scan_ctx: dict = {}

    # ------------------------------------------------------------------
    # 3-D reconstruction helper
    # ------------------------------------------------------------------

    def _reconstruct_volume(self, source, series):
        """Reconstruct a 3-D intensity volume from a scan window.

        Runs without gradient tracking (torch.no_grad).  The volume is used
        as a reference/target in the self-supervised loss; gradients flow
        through the volume's bounding-box origin (bias) — see module docstring.

        Args:
            source  (N, 1, H, W)  float32 frames in [0, 1], on self.device
            series  (N, 3, 3)     world-mm positions [center, LL, LR]
                                  — pass fake_series.detach() to avoid
                                    accidentally creating a gradient path here

        Returns:
            volume  (D, H', W')  reconstructed 3-D volume  (no gradient)
            bias    (3,)         world-mm origin offset (detached);
                                 volume-aligned coordinates = series - bias
        """
        with torch.no_grad():
            source_down = F.interpolate(
                source,
                scale_factor=self.down_ratio,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)                   # (N, H·dr, W·dr)

            volume, bias = my_utils.reco(
                source_down, series,
                self.scale_w, self.scale_h,
                self.mat_scale,
            )

            if self.down_ratio != 1.0:
                volume = F.interpolate(
                    volume.unsqueeze(0).unsqueeze(0),
                    scale_factor=1.0 / self.down_ratio,
                ).squeeze(0).squeeze(0)

        return volume, bias

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
        """
        N = source.shape[0]
        chunk_outputs = []

        for start in range(0, N - 1, self.backbone_chunk_size):
            end = min(start + self.backbone_chunk_size, N - 1)

            # Build pair chunk on-the-fly to avoid allocating the full pairs tensor
            chunk = torch.cat(
                [source[start:end], source[start + 1:end + 1]], dim=1
            ).unsqueeze(0)                 # (1, k, 2, H, W)

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
        """Fraction of pixels in a scan plane that fall within volume bounds.

        Converts ``series_pos`` to volume-mm coordinates (``series_pos - bias``),
        builds the pixel mesh for an (H × W) image, and counts the fraction of
        mesh points whose 3-D position lies inside the volume bounding box
        ``[0, D) × [0, H') × [0, W')``.

        Called inside ``torch.no_grad()`` by ``_find_intersecting_frame``.

        Args:
            series_pos  (1, 3, 3)    one scan position in world-mm
                                     [center, lower-left, lower-right]
            volume      (D, H', W')  reconstructed volume (shape used only)
            bias        (3,)         world-mm origin offset from reco()
            H, W        int          image height / width in pixels

        Returns:
            float  fraction of pixels inside the volume, in [0, 1]
        """
        pos_vol = series_pos - bias.detach().view(1, 1, 3)  # (1, 3, 3)
        mesh = my_utils.transform(
            pos_vol, H, W,
            scale_h=self.scale_h, scale_w=self.scale_w,
        )  # (1, H, W, 3)  in volume-mm coords

        in_bounds = (
            (mesh[..., 0] >= 0) & (mesh[..., 0] < volume.shape[0]) &
            (mesh[..., 1] >= 0) & (mesh[..., 1] < volume.shape[1]) &
            (mesh[..., 2] >= 0) & (mesh[..., 2] < volume.shape[2])
        )
        return in_bounds.float().mean().item()

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
            (real_image, gt_pos)  if a qualifying frame is found
                real_image  (1, 1, H, W)  float32 in [0, 1]
                gt_pos      (1, 3, 3)     GT world-mm position
            None  if no qualifying frame is found within the search budget
        """
        threshold     = float(getattr(self.cfg, 'intersect_threshold', 0.5))
        max_ds_trials = int(getattr(self.cfg, 'max_dataset_trials', 20))
        ctx           = self._scan_ctx

        source_full    = ctx['source_full']         # (N_full, 1, H, W)
        gt_series_full = ctx['gt_series_full']      # (N_full, 3, 3)
        window_set     = ctx['window_indices_set']
        N_full         = source_full.shape[0]

        # ── Pool 1: outside-window frames of the current scan ─────────
        with torch.no_grad():
            for idx in range(N_full):
                if idx in window_set:
                    continue
                pos  = gt_series_full[idx:idx + 1]          # (1, 3, 3)
                frac = self._overlap_fraction(pos, volume, bias, H, W)
                if frac >= threshold:
                    return source_full[idx:idx + 1], pos    # (1,1,H,W), (1,3,3)

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
                    return r_src[f_idx:f_idx + 1], pos

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

        real_image, gt_pos = result    # (1, 1, H, W), (1, 3, 3)

        # ── 2. Differentiable bias from fake_series ───────────────────
        # bias returned by _reconstruct_volume is detached (built inside
        # no_grad).  Re-deriving it here from fake_series (with gradient)
        # opens the gradient path through the slice-extraction coordinate.
        _, bias_grad = my_utils.get_reco_size(fake_series, self.mat_scale)
        # bias_grad: (3,)  — gradient via torch.min of trajectory corners

        # ── 3. Extract slice from volume at the GT position ───────────
        # gt_pos - bias_grad: shifts the GT world-mm position into
        # volume-mm coordinates using the differentiable origin.
        gt_pos_vol = gt_pos - bias_grad.view(1, 1, 3)    # (1, 3, 3), grad via bias_grad
        slice_vol  = my_utils.get_slice(
            volume, gt_pos_vol, (H, W),
            scale_h=self.scale_h, scale_w=self.scale_w,
        )                                                  # (1, 1, 1, H, W)
        slice_vol  = slice_vol.squeeze(0)                  # (1, 1, H, W)

        # ── 4. Loss 1: slice – real-image L1 similarity ───────────────
        weight_loss1 = float(getattr(self.cfg, 'weight_loss1', 1.0))
        loss_slice   = F.l1_loss(slice_vol, real_image) * weight_loss1

        return {'loss_slice': loss_slice}

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
            else:
                source    = source_full
                gt_series = gt_series_full
                window_indices_set = set(range(N_full))

            # ── Store full-scan context for custom_criterion ─────────────
            # Populated here so that _find_intersecting_frame can access
            # the outside-window frames of the current scan.
            self._scan_ctx = {
                'source_full':        source_full,
                'gt_series_full':     gt_series_full,
                'window_indices_set': window_indices_set,
            }

            self.optimizer.zero_grad()

            # ── 1. Predict inter-frame gaps (gradient checkpointing) ─────
            fake_gaps = self._run_backbone(source, use_checkpoint=True)    # (N-1, 6)

            # ── 2. Reconstruct predicted probe trajectory ─────────────────
            fake_series = utils.simulation.dof_to_series(
                gt_series[0:1],             # (1, 3, 3)  GT first-frame pose
                fake_gaps.unsqueeze(0),     # (1, N-1, 6)
            ).squeeze(0)                    # (N, 3, 3)  — with gradient

            # ── 3. Reconstruct 3-D volume (no gradient through voxels) ───
            volume, bias = self._reconstruct_volume(
                source.detach(), fake_series.detach()
            )                                                               # (D,H',W'), (3,)

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
                break   # valid window found; exit retry loop

        if losses is None:
            # All window re-sampling attempts failed; skip gradient step.
            return {'loss': torch.tensor(0.0, device=self.device)}

        loss = sum(losses.values())
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(epoch_info['epoch'])

        return {'loss': loss.detach(), **{k: v.detach() for k, v in losses.items()}}

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