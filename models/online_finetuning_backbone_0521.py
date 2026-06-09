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
   gradients.  Gradient flow for the self-supervised loss comes entirely
   through fake_series (and thus fake_gaps → backbone), which represents
   the *position* of slices in the volume — i.e. the differentiable
   rendering coordinate path, not through the voxel intensities.

   For a slice-consistency loss, the gradient chain is:
       backbone weights → fake_gaps → fake_series → slice positions
           → loss (comparing volume slice at predicted position vs. source)
   This is the standard formulation; the volume values need not be
   differentiable.

Config keys (all optional, in res/models/online_finetune_bk.json)
------------------------------------------------------------------
pretrained_weight    str    path to pre-trained backbone checkpoint
max_train_frames     int    maximum frames per training window (default 64)
                            if the scan has more frames a random contiguous
                            window of this length is used each training step
backbone_chunk_size  int    pairs per backbone forward chunk (default 8)
down_ratio           float  spatial downscale factor for reco() (default 1.0)
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

    # ------------------------------------------------------------------
    # 3-D reconstruction helper
    # ------------------------------------------------------------------

    def _reconstruct_volume(self, source, series):
        """Reconstruct a 3-D intensity volume from a scan window.

        Runs without gradient tracking (torch.no_grad).  The volume is used
        as a reference/target in the self-supervised loss; gradients flow
        through fake_series (the *positions* of slices), not the voxel values.

        Args:
            source  (N, 1, H, W)  float32 frames in [0, 1], on self.device
            series  (N, 3, 3)     world-mm positions [center, LL, LR]
                                  — pass fake_series.detach() to avoid
                                    accidentally creating a gradient path here

        Returns:
            volume  (D, H', W')  reconstructed 3-D volume  (no gradient)
            bias    (3,)         world-mm origin offset; subtract from series
                                 to get volume-aligned coordinates
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
    # Self-supervised loss — override in subclasses or replace directly
    # ------------------------------------------------------------------

    def custom_criterion(self, source, fake_gaps, fake_series, volume, bias):
        """Define the self-supervised loss on the reconstructed volume.

        Called once per training step after 3-D reconstruction.

        Args:
            source      (N, 1, H, W)   raw input frames in [0, 1]  (with grad)
            fake_gaps   (N-1, 6)       predicted 6-DoF transforms
                                        [tx, ty, tz, rx, ry, rz] (descaled, with grad)
            fake_series (N, 3, 3)      predicted probe positions in world-mm
                                        [center, lower-left, lower-right]  (with grad)
            volume      (D, H', W')    reconstructed 3-D volume  (NO grad — reference only)
            bias        (3,)           world-mm origin offset from reco();
                                        volume-aligned coordinates = series - bias

        Returns:
            dict[str, Tensor]  {loss_name: scalar_tensor, ...}
                All values are summed to form the total loss for backprop.
                Gradients flow through fake_series / fake_gaps → backbone.

        Typical loss pattern (slice consistency)::

            def custom_criterion(self, source, fake_gaps, fake_series, volume, bias):
                # Slice the volume at predicted positions (gradient flows through positions)
                slices = my_utils.get_slice(
                    volume, fake_series - bias,
                    source.shape[-2], source.shape[-1],
                    scale_h=self.scale_h, scale_w=self.scale_w,
                )  # (N, 1, H, W)
                loss_recon = F.l1_loss(slices, source)
                return {'loss_recon': loss_recon}
        """
        raise NotImplementedError(
            "Implement your self-supervised criterion in "
            "`Online_Finetuning_Backbone.custom_criterion`."
        )

    # ------------------------------------------------------------------
    # Training step — one scan (or window) per call
    # ------------------------------------------------------------------

    def train(self, epoch_info, sample_dict):
        """One training step on a complete scan (or a windowed sub-sequence).

        Expected sample_dict keys (batch_size=1, squeezed inside here):
            source  (1, N, 1, H, W)   float32 ultrasound frames
            target  (1, N, 15)        gaps_padded | series_flat

        Memory strategy
        ~~~~~~~~~~~~~~~
        * If N > max_train_frames, a random contiguous window is used.
        * Backbone runs in eval mode (frozen BN stats) so gradient checkpointing
          does not cause BatchNorm double-update.
        * Volume reconstruction runs under torch.no_grad(); the self-supervised
          gradient signal comes from slice *positions* (fake_series), not voxel values.

        Returns:
            dict[str, Tensor]  {loss_name: detached scalar, ...}
        """
        # ── Unpack (batch_size=1 → squeeze) ───────────────────────────
        source    = sample_dict['source'].to(self.device).squeeze(0)   # (N, 1, H, W)
        target    = sample_dict['target'].to(self.device).squeeze(0)   # (N, 15)
        gt_series = target[:, -9:].view(-1, 3, 3)                      # (N, 3, 3)

        # ── Optional: window long scans ────────────────────────────────
        N = source.shape[0]
        if self.max_train_frames > 0 and N > self.max_train_frames:
            t0        = torch.randint(0, N - self.max_train_frames + 1, (1,)).item()
            source    = source[t0:t0 + self.max_train_frames]
            gt_series = gt_series[t0:t0 + self.max_train_frames]

        # ── Backbone: eval mode freezes BN stats (compatible with checkpointing) ──
        # Parameters still receive gradients; only BN running stats are frozen.
        self.backbone.eval()
        self.optimizer.zero_grad()

        # ── 1. Predict inter-frame gaps (with gradient checkpointing) ──
        fake_gaps = self._run_backbone(source, use_checkpoint=True)    # (N-1, 6)

        # ── 2. Reconstruct predicted probe trajectory ──────────────────
        fake_series = utils.simulation.dof_to_series(
            gt_series[0:1],             # (1, 3, 3)  GT first-frame pose
            fake_gaps.unsqueeze(0),     # (1, N-1, 6)
        ).squeeze(0)                    # (N, 3, 3)  — with gradient

        # ── 3. Reconstruct 3-D volume (no gradient through voxels) ─────
        # Pass detached series so reco() cannot accidentally create a
        # gradient path through the volume intensities.
        volume, bias = self._reconstruct_volume(
            source.detach(), fake_series.detach()
        )                                                               # (D,H',W'), (3,)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  DEBUG: 3-D volume visualisation — comment out for training     ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self._debug_visualise_volume(
            volume, bias,
            fake_series.detach(),
            source.detach(),
            epoch_info,
        )
        # ╚══════════════════════════════════════════════════════════════════╝

        # ── 4. Self-supervised loss ────────────────────────────────────
        # fake_series / fake_gaps still carry gradients → backbone weights
        losses = self.custom_criterion(source, fake_gaps, fake_series, volume, bias)
        loss   = sum(losses.values())

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
