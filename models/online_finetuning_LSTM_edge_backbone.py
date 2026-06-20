"""Online_Finetuning_LSTM_Edge_Backbone — scan-level self-supervised fine-tuning
of the LSTM-edge backbone (ResNet18 + Conv2dLSTM).

The backbone takes 4-channel pairs:

    input[:, t, :, :, :] = [source[t], source[t+1], edge[t], edge[t+1]]

Edge maps are computed on-the-fly inside _run_backbone via utils.image.get_edge
(Canny2D), so no extra dataset keys are required — the standard TUS_complete_scan
dataset (returning source + target) is used unchanged.

Differences from Online_Finetuning_Backbone
-------------------------------------------
* Backbone   : ResNet18 + Conv2dLSTM (4-channel pairs) instead of EfficientNet-B1.
* No chunked gradient checkpointing: the LSTM needs the full sequence to keep
  hidden state.  Memory is controlled via max_train_frames instead.
* Edge features are computed from source frames at the start of _run_backbone.

Config keys
-----------
pretrained_weight   str   path to the LSTM-edge backbone checkpoint
                          (save/online_LSTM_edge_bk-hp_bk-TUS_LSTM_edge_complete/
                           online_LSTM_edge_bk_backbone_<epoch>.pth)
All other keys are identical to Online_Finetuning_Backbone (max_train_frames,
down_ratio, criterion_scale, weight_loss1/rot/jagged/ssim, etc.).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

import configs
import models
import utils
import trial.my_utils.functions as my_utils
from models.online_LSTM_edge_backbone import Backbone as LSTMEdgeBackbone
from utils.plot_functions import read_calib_matrices


# ── Module-level SSIM helpers (used by Loss 4) ────────────────────────────────

def _gaussian_window(size, sigma, device, dtype):
    """2-D separable Gaussian kernel of shape (1, 1, size, size)."""
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-coords.pow(2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, size, 1) * g.view(1, 1, 1, size)


def _batch_ssim(x, y, window_size=11, sigma=1.5, C1=1e-4, C2=9e-4):
    """Mean SSIM between two batches of single-channel images.

    Args:
        x, y        (N, 1, H, W)  float32 in [0, 1]
        window_size int            Gaussian kernel width
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

    k = _gaussian_window(ws, sigma, x.device, x.dtype)

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

class Online_Finetuning_LSTM_Edge_Backbone(models.BaseModel):
    """Self-supervised fine-tuning wrapper for the LSTM-edge backbone."""

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)

        # ── LSTM-Edge Backbone ─────────────────────────────────────────────
        # data_cfg.source.channel must equal 4 (source + edge per-frame pair).
        self.backbone = LSTMEdgeBackbone(
            self.data_cfg.source.channel,       # 4
            self.data_cfg.target.elements - 9,  # 6
        ).to(self.device)

        pretrained_path  = configs.env.getdir(self.cfg.pretrained_weight)
        pretrained_state = torch.load(pretrained_path, map_location=self.device)
        self.backbone.load_state_dict(pretrained_state)

        # ── Optimiser / scheduler ──────────────────────────────────────────
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

        # ── Calibration for 3-D reconstruction ────────────────────────────
        calib_path           = configs.env.getdir(self.data_cfg.paths.calib)
        T_calib_scale, _, _  = read_calib_matrices(calib_path)
        self.scale_w = float(T_calib_scale[0, 0])   # mm / pixel (u → width)
        self.scale_h = float(T_calib_scale[1, 1])   # mm / pixel (v → height)

        self.down_ratio = float(getattr(self.cfg, 'down_ratio', 1.0))
        self.mat_scale  = torch.eye(4, dtype=torch.float32, device=self.device)
        self.mat_scale[0, 0] = self.down_ratio
        self.mat_scale[1, 1] = self.down_ratio
        self.mat_scale[2, 2] = self.down_ratio

        # backbone_chunk_size is stored for API compat but unused (LSTM = full sequence).
        self.backbone_chunk_size = int(getattr(self.cfg, 'backbone_chunk_size', 8))
        self.max_train_frames    = int(getattr(self.cfg, 'max_train_frames', 64) or 0)
        self.grad_clip_norm      = float(getattr(self.cfg, 'grad_clip_norm', 1.0) or 0.0)

        self._scan_ctx: dict = {}
        self._component_loss_keys: list = self._enabled_loss_keys()

    # ------------------------------------------------------------------
    # Loss-key helpers
    # ------------------------------------------------------------------

    def _enabled_loss_keys(self):
        """Return the list of loss keys that will be active given the current config."""
        keys = ['loss_slice']
        if float(getattr(self.cfg, 'weight_loss_jagged',      0.0)) > 0:
            keys.append('loss_jagged')
        if float(getattr(self.cfg, 'weight_loss_ssim',        0.0)) > 0:
            keys.append('loss_ssim')
        if float(getattr(self.cfg, 'weight_loss_consistency', 0.0)) > 0:
            keys.append('loss_consistency')
        return keys

    # ------------------------------------------------------------------
    # 3-D reconstruction helper
    # ------------------------------------------------------------------

    def _reconstruct_volume(self, source, series, enable_grad=False):
        """Reconstruct a 3-D intensity volume from a scan window.

        Args:
            source       (N, 1, H, W)  float32 frames in [0, 1], on self.device
            series       (N, 3, 3)     world-mm positions [center, LL, LR]
            enable_grad  bool          False → torch.no_grad(); True → gradient tracking

        Returns:
            volume  (D, H', W')  reconstructed 3-D volume
            bias    (3,)         world-mm origin offset
        """
        def _body(source_, series_):
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
            ).squeeze(1)

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
            return grad_ckpt(_body, source, series, use_reentrant=False)
        else:
            with torch.no_grad():
                return _body(source, series)

    # ------------------------------------------------------------------
    # Backbone inference — LSTM-edge variant
    # ------------------------------------------------------------------

    def _run_backbone(self, source, use_checkpoint=True):
        """Run the LSTM-edge backbone on N consecutive frames.

        Computes Canny edge maps on-the-fly, forms 4-channel pairs
        [source[t], source[t+1], edge[t], edge[t+1]], and runs the
        ResNet18 + Conv2dLSTM backbone on the full sequence.

        The LSTM processes the whole sequence atomically (no chunking).
        use_checkpoint is accepted for API compatibility but ignored.

        Args:
            source          (N, 1, H, W)  float32 frames in [0, 1]
            use_checkpoint  bool          ignored

        Returns:
            fake_gaps  (N-1, 6)  [tx, ty, tz, rx, ry, rz] (rotation descaled)
        """
        with torch.no_grad():
            edge = utils.image.get_edge(
                source.squeeze(1), device=self.device
            )  # (N, 1, H, W)

        s0 = source[:-1]   # (N-1, 1, H, W)
        s1 = source[1:]    # (N-1, 1, H, W)
        e0 = edge[:-1]     # (N-1, 1, H, W)
        e1 = edge[1:]      # (N-1, 1, H, W)

        bb_scale = float(getattr(self.cfg, 'backbone_input_scale', 1.0))
        if bb_scale != 1.0:
            def _ds(t):
                return F.interpolate(
                    t, scale_factor=bb_scale, mode='bilinear', align_corners=False
                )
            s0, s1, e0, e1 = _ds(s0), _ds(s1), _ds(e0), _ds(e1)

        # (1, N-1, 4, H_s, W_s) — matches the training input layout of
        # Online_LSTM_Edge_Backbone: [source[t], source[t+1], edge[t], edge[t+1]]
        inp = torch.cat([s0, s1, e0, e1], dim=1).unsqueeze(0)

        raw_gaps, _ = self.backbone(inp, return_feature=False)  # (1, N-1, 6)
        raw_gaps = raw_gaps.squeeze(0)                           # (N-1, 6)

        # Descale rotation angles (backbone trained with angles × 100)
        fake_gaps = torch.cat(
            [raw_gaps[:, :3], raw_gaps[:, 3:] / 100.0], dim=-1
        )  # (N-1, 6)
        return fake_gaps

    # ------------------------------------------------------------------
    # Intersection helpers (used by custom_criterion)
    # ------------------------------------------------------------------

    def _overlap_fraction(self, series_pos, volume, bias, H, W):
        """Fraction of a scan plane that falls inside the volume bounding box.

        Uses a coarse 20 × 20 CPU grid to avoid GPU allocation inside the
        frame-search loop.

        Args:
            series_pos  (1, 3, 3)    one scan position in world-mm
            volume      (D, H', W')  reconstructed volume (shape used only)
            bias        (3,)         world-mm origin offset from reco()
            H, W        int          image height / width in pixels

        Returns:
            float  fraction of coarse-grid points inside the volume, in [0, 1]
        """
        GRID = 20

        pos   = series_pos.detach().cpu().float()
        b     = bias.detach().cpu().float()
        pos_vol = pos - b.view(1, 1, 3)

        xs = torch.linspace(-(H - 1) / 2, (H - 1) / 2, GRID, dtype=torch.float32)
        ys = torch.linspace(-(W - 1) / 2, (W - 1) / 2, GRID, dtype=torch.float32)
        gx, gy = torch.meshgrid(xs, ys, indexing='ij')

        local = torch.stack(
            [gy * self.scale_w, -gx * self.scale_h, torch.zeros_like(gx)],
            dim=-1,
        )

        axis   = my_utils.get_axis(pos_vol).permute(0, 2, 1)
        center = pos_vol[:, 0, :]
        mesh   = torch.einsum('ij,HWj->HWi', axis[0], local) + center.view(1, 1, 3)

        vs   = volume.shape
        in_b = (
            (mesh[..., 0] >= 0) & (mesh[..., 0] < vs[0]) &
            (mesh[..., 1] >= 0) & (mesh[..., 1] < vs[1]) &
            (mesh[..., 2] >= 0) & (mesh[..., 2] < vs[2])
        )
        return in_b.float().mean().item()

    def _find_intersecting_frame(self, volume, bias, H, W):
        """Search for a real frame whose GT position intersects the volume.

        Searches two candidate pools in priority order:
        1. Current-scan outside-window frames
        2. Training-dataset fallback

        Args:
            volume  (D, H', W')  reconstructed 3-D volume
            bias    (3,)          world-mm origin offset
            H, W    int           image dimensions (pixels)

        Returns:
            (real_image, gt_pos, overlap_frac, source_tag)  if a frame is found
            None  if no qualifying frame is found within the search budget
        """
        threshold     = float(getattr(self.cfg, 'intersect_threshold', 0.5))
        max_ds_trials = int(getattr(self.cfg, 'max_dataset_trials', 20))
        ctx           = self._scan_ctx

        source_full    = ctx['source_full']
        gt_series_full = ctx['gt_series_full']
        window_set     = ctx['window_indices_set']
        win_start      = ctx['window_start']
        win_size       = ctx['window_size']
        N_full         = source_full.shape[0]

        # ── Pool 1a: non-anchor window frames ─────────────────────────
        win_end  = win_start + win_size
        mid      = win_start + win_size // 2
        pool_1a  = (
            list(range(max(mid, win_start + 1), win_end))
            + list(range(win_start + 1, mid))[::-1]
        )

        with torch.no_grad():
            for idx in pool_1a:
                pos  = gt_series_full[idx:idx + 1]
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
                r_src  = r_data['source'].to(self.device)
                r_tgt  = r_data['target'].to(self.device)
                r_ser  = r_tgt[:, -9:].view(-1, 3, 3)

                f_idx  = torch.randint(r_ser.shape[0], (1,)).item()
                pos    = r_ser[f_idx:f_idx + 1]
                frac   = self._overlap_fraction(pos, volume, bias, H, W)
                if frac >= threshold:
                    return r_src[f_idx:f_idx + 1], pos, frac, 'training_dataset'

        return None

    # ------------------------------------------------------------------
    # Self-supervised loss — Loss 1 + extension hooks
    # ------------------------------------------------------------------

    def custom_criterion(self, source, fake_gaps, fake_series, volume, bias):
        """Self-supervised loss based on slice–real-image similarity (Loss 1).

        Args:
            source      (N, 1, H, W)   raw input frames in [0, 1]
            fake_gaps   (N-1, 6)       predicted 6-DoF transforms
            fake_series (N, 3, 3)      predicted probe positions in world-mm
            volume      (D, H', W')    reconstructed 3-D volume (NO grad)
            bias        (3,)           world-mm origin offset (detached)

        Returns:
            dict[str, Tensor]  or None if no qualifying intersecting frame found
        """
        H, W = source.shape[-2], source.shape[-1]

        result = self._find_intersecting_frame(volume, bias, H, W)
        if result is None:
            return None

        real_image, gt_pos, overlap_frac, _src_tag = result

        dense_grad = bool(getattr(self.cfg, 'dense_grad', False))
        if dense_grad:
            bias_origin = bias
        else:
            _, bias_origin = my_utils.get_reco_size(fake_series, self.mat_scale)

        crit_scale = float(getattr(self.cfg, 'criterion_scale', 1))
        H_c = max(1, int(H * crit_scale))
        W_c = max(1, int(W * crit_scale))

        gt_pos_vol = gt_pos - bias_origin.view(1, 1, 3)
        slice_vol  = my_utils.get_slice(
            volume, gt_pos_vol, (H_c, W_c),
            scale_h=self.scale_h / crit_scale,
            scale_w=self.scale_w / crit_scale,
        ).squeeze(0)

        slice_content_thresh     = float(getattr(self.cfg, 'slice_content_thresh',     0.02))
        slice_min_content_ratio  = float(getattr(self.cfg, 'slice_min_content_ratio',  0.05))
        content_ratio = (slice_vol.detach() > slice_content_thresh).float().mean().item()
        if content_ratio < slice_min_content_ratio:
            return None

        real_small = F.interpolate(
            real_image, size=(H_c, W_c), mode='bilinear', align_corners=False,
        )

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
        #     real_image    = real_small.detach(),
        #     slice_vol     = slice_vol.detach(),
        #     overlap_frac  = overlap_frac,
        #     loss_val      = loss_slice.item() / weight_loss1,
        #     source_tag    = _src_tag,
        #     epoch_info    = self._scan_ctx.get('epoch_info'),
        # )
        # # ╚══════════════════════════════════════════════════════════════════╝

        losses = {'loss_slice': loss_slice}

        weight_loss_jagged = float(getattr(self.cfg, 'weight_loss_jagged', 0.0))
        if weight_loss_jagged > 0:
            losses.update(self._rotation_consistency_loss(
                volume, bias_origin, gt_pos, H, W, H_c, W_c, crit_scale,
            ))

        weight_loss_ssim = float(getattr(self.cfg, 'weight_loss_ssim', 0.0))
        if weight_loss_ssim > 0:
            losses['loss_ssim'] = self._volume_ssim_loss(
                volume, bias_origin, fake_series, H, W, crit_scale, weight_loss_ssim,
            )

        weight_loss_consistency = float(getattr(self.cfg, 'weight_loss_consistency', 0.0))
        if weight_loss_consistency > 0:
            losses['loss_consistency'] = self._slice_consistency_loss(
                volume, bias_origin, gt_pos, slice_vol, H, W, H_c, W_c, crit_scale,
            )

        return losses

    # ------------------------------------------------------------------
    # Volume SSIM loss helper (Loss 4)
    # ------------------------------------------------------------------

    def _volume_ssim_loss(self, volume, bias_origin, fake_series, H, W, crit_scale, weight):
        """Loss 4: 1 − mean SSIM between adjacent trajectory-frame slices.

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

        if 0 < num_pairs < N - 1:
            perm  = torch.randperm(N - 1, device=fake_series.device)[:num_pairs]
            idx_a = perm
        else:
            idx_a = torch.arange(N - 1, device=fake_series.device)
        idx_b = idx_a + 1

        series_a_vol = fake_series[idx_a] - bias_origin.view(1, 1, 3)
        series_b_vol = fake_series[idx_b] - bias_origin.view(1, 1, 3)

        slices_a = my_utils.get_slice(
            volume, series_a_vol, (H_c, W_c),
            scale_h=self.scale_h / crit_scale,
            scale_w=self.scale_w / crit_scale,
        ).squeeze(0)

        slices_b = my_utils.get_slice(
            volume, series_b_vol, (H_c, W_c),
            scale_h=self.scale_h / crit_scale,
            scale_w=self.scale_w / crit_scale,
        ).squeeze(0)

        ssim_val = _batch_ssim(slices_a, slices_b, window_size=window_size)
        return (1.0 - ssim_val) * weight

    # ------------------------------------------------------------------
    # Rotation losses helper (Loss 2 + Loss 3)
    # ------------------------------------------------------------------

    def _rotation_consistency_loss(
        self, volume, bias_origin, gt_pos, H, W, H_c, W_c, crit_scale,
    ):
        """Jaggedness loss on a randomly rotated slice (Loss 3).

        Args:
            volume      (D, H', W')  reconstructed 3-D volume
            bias_origin (3,)          differentiable vol-mm origin
            gt_pos      (1, 3, 3)    GT world-mm position (detached)
            H, W        int           full-resolution image dims
            H_c, W_c    int           criterion-scale dims
            crit_scale  float         criterion downscale factor

        Returns:
            dict[str, Tensor]  subset of {loss_jagged}
        """
        weight_loss_jagged = float(getattr(self.cfg, 'weight_loss_jagged', 0.0))
        max_angle          = float(getattr(self.cfg, 'rotation_max_angle', 0.02))

        axis = my_utils.get_axis(gt_pos.float())
        ax_x = axis[0, 0]
        ax_y = axis[0, 1]
        ax_z = axis[0, 2]

        theta = (torch.rand(1, device=gt_pos.device).item() * 2.0 - 1.0) * max_angle
        dev, dt = gt_pos.device, gt_pos.dtype
        cos_t    = torch.tensor(theta, device=dev, dtype=dt).cos()
        sin_t    = torch.tensor(theta, device=dev, dtype=dt).sin()
        ax_y_rot = cos_t * ax_y + sin_t * ax_z

        half_w = (W - 1) / 2.0 * self.scale_w
        half_h = (H - 1) / 2.0 * self.scale_h
        center = gt_pos[0, 0]
        ll_rot = center - ax_x * half_w - ax_y_rot * half_h
        lr_rot = center + ax_x * half_w - ax_y_rot * half_h
        gt_pos_rot = torch.stack([center, ll_rot, lr_rot]).unsqueeze(0)

        gt_pos_rot_vol = gt_pos_rot - bias_origin.view(1, 1, 3)

        slice_rot = my_utils.get_slice(
            volume, gt_pos_rot_vol, (H_c, W_c),
            scale_h=self.scale_h / crit_scale,
            scale_w=self.scale_w / crit_scale,
        ).squeeze(0)

        losses = {}

        if weight_loss_jagged > 0:
            drow = (slice_rot[:, :, 1:, :] - slice_rot[:, :, :-1, :]).abs()
            dcol = (slice_rot[:, :, :, 1:] - slice_rot[:, :, :, :-1]).abs()
            jagged_mask_thresh = float(getattr(self.cfg, 'jagged_mask_thresh', 0.0))
            if jagged_mask_thresh > 0:
                # 仅在相邻像素均为有效内容（非越界黑区）处计算梯度
                content   = (slice_rot.detach() > jagged_mask_thresh)
                drow_mask = content[:, :, 1:, :] & content[:, :, :-1, :]
                dcol_mask = content[:, :, :, 1:] & content[:, :, :, :-1]
                if drow_mask.any() and dcol_mask.any():
                    losses['loss_jagged'] = (
                        F.relu(drow[drow_mask].mean() - dcol[dcol_mask].mean().detach())
                        * weight_loss_jagged
                    )
            else:
                losses['loss_jagged'] = (
                    F.relu(drow.mean() - dcol.mean().detach()) * weight_loss_jagged
                )

        # # ╔══════════════════════════════════════════════════════════════════╗
        # # ║  DEBUG: rotation-consistency visualisation                      ║
        # # ╚══════════════════════════════════════════════════════════════════╝
        # self._debug_visualise_rotation_consistency(
        #     volume      = volume,
        #     bias        = bias_origin.detach(),
        #     gt_pos      = gt_pos,
        #     gt_pos_rot  = gt_pos_rot,
        #     ax_x        = ax_x,
        #     slice_rot   = slice_rot.detach(),
        #     theta       = theta,
        #     loss_jagged = losses.get('loss_jagged'),
        # )
        # # ╚══════════════════════════════════════════════════════════════════╝

        return losses

    # ------------------------------------------------------------------
    # Slice Consistency Loss helper (Loss 5)
    # ------------------------------------------------------------------

    def _slice_consistency_loss(
        self, volume, bias_origin, gt_pos, slice_orig, H, W, H_c, W_c, crit_scale,
    ):
        """Loss 5: spatial consistency via slightly rotated probe-position slices.

        Extracts a volume slice at a small random rotation from gt_pos and
        compares it to the original slice at gt_pos. If the reconstructed volume
        is spatially consistent, nearby probe orientations should yield similar
        images. Trajectory errors that cause staircase artifacts in the volume
        will produce large differences between the two slices.

        Algorithm
        ~~~~~~~~~
        1. Derive local frame axes (ax_x, ax_y, ax_z) from gt_pos.
        2. Sample a random tilt θ ∈ [-consistency_max_angle, +consistency_max_angle]
           around ax_x (the horizontal in-plane axis of the probe).
        3. Build a rotated probe position: same center and physical size,
           but ax_y replaced by cos(θ)·ax_y + sin(θ)·ax_z.
        4. Extract a volume slice at the rotated position → slice_rot.
        5. loss_consistency = L1(slice_orig, slice_rot)   or  (1 − SSIM)

        Gradient path (sparse, same as loss_slice)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        bias_origin → gt_pos_rot_vol → get_slice / F.grid_sample ∂grid
            → loss_consistency
        Gradient also flows back through slice_orig via bias_origin
        (both slices share the same differentiable origin).

        Config keys
        ~~~~~~~~~~~
        weight_loss_consistency  float  loss weight (default 0.0; set > 0 to enable)
        consistency_max_angle    float  max tilt angle in radians
                                        (default: same as rotation_max_angle, 0.02)
        consistency_loss_fn      str    'l1' (default) or 'ssim'

        Args:
            volume       (D, H', W')       reconstructed 3-D volume
            bias_origin  (3,)              differentiable vol-mm origin
            gt_pos       (1, 3, 3)         GT world-mm position (detached)
            slice_orig   (1, 1, H_c, W_c) volume slice at gt_pos (already computed)
            H, W         int               full-resolution image dims
            H_c, W_c     int               criterion-scale dims
            crit_scale   float             criterion downscale factor

        Returns:
            scalar Tensor  weighted consistency loss
        """
        weight = float(getattr(self.cfg, 'weight_loss_consistency', 0.0))

        axis = my_utils.get_axis(gt_pos.float())
        ax_x = axis[0, 0]
        ax_y = axis[0, 1]
        ax_z = axis[0, 2]

        # 固定角度幅值，随机正负方向，使每步 loss 量级一致且覆盖两侧扰动。
        angle = float(getattr(self.cfg, 'consistency_angle', 0.01))
        if torch.rand(1).item() < 0.5:
            angle = -angle

        dev, dt = gt_pos.device, gt_pos.dtype
        cos_t    = torch.tensor(angle, device=dev, dtype=dt).cos()
        sin_t    = torch.tensor(angle, device=dev, dtype=dt).sin()
        ax_y_rot = cos_t * ax_y + sin_t * ax_z

        half_w = (W - 1) / 2.0 * self.scale_w
        half_h = (H - 1) / 2.0 * self.scale_h
        center = gt_pos[0, 0]
        ll_rot = center - ax_x * half_w - ax_y_rot * half_h
        lr_rot = center + ax_x * half_w - ax_y_rot * half_h
        gt_pos_rot = torch.stack([center, ll_rot, lr_rot]).unsqueeze(0)  # (1, 3, 3)

        gt_pos_rot_vol = gt_pos_rot - bias_origin.view(1, 1, 3)

        slice_rot = my_utils.get_slice(
            volume, gt_pos_rot_vol, (H_c, W_c),
            scale_h=self.scale_h / crit_scale,
            scale_w=self.scale_w / crit_scale,
        ).squeeze(0)  # (1, 1, H_c, W_c)

        use_ssim = str(getattr(self.cfg, 'consistency_loss_fn', 'l1')).lower() == 'ssim'
        if use_ssim:
            ssim_val = _batch_ssim(slice_orig, slice_rot)
            loss_consistency = (1.0 - ssim_val) * weight
        else:
            loss_consistency = F.l1_loss(slice_orig, slice_rot) * weight

        # # ╔══════════════════════════════════════════════════════════════════╗
        # # ║  DEBUG: slice-consistency visualisation — comment out for train ║
        # # ╚══════════════════════════════════════════════════════════════════╝
        # self._debug_visualise_slice_consistency(
        #     volume         = volume,
        #     bias           = bias_origin.detach(),
        #     gt_pos         = gt_pos,
        #     gt_pos_rot     = gt_pos_rot,
        #     ax_x           = ax_x,
        #     slice_orig     = slice_orig.detach(),
        #     slice_rot      = slice_rot.detach(),
        #     angle          = angle,
        #     loss_consistency = loss_consistency,
        # )
        # # ╚══════════════════════════════════════════════════════════════════╝

        return loss_consistency

    # ------------------------------------------------------------------
    # Debug visualisation for slice consistency loss
    # ------------------------------------------------------------------

    def _debug_visualise_slice_consistency(
        self,
        volume,
        bias,
        gt_pos,
        gt_pos_rot,
        ax_x,
        slice_orig,
        slice_rot,
        angle,
        loss_consistency=None,
    ):
        """Visualise Loss 5 (slice consistency): original vs rotated probe slice.

        Shows two blocking windows (close each to continue):

        **Window 1 — Matplotlib** (3 panels):
        ┌──────────────┬──────────────┬──────────────┐
        │ slice_orig   │  slice_rot   │ |orig - rot| │
        │ (gt_pos)     │ (rotated θ)  │  diff map    │
        └──────────────┴──────────────┴──────────────┘

        **Window 2 — PyVista** (3-D):
        • Reconstructed volume (bone colormap, sigmoid opacity)
        • Original GT frame — green quad with slice_orig texture
        • Rotated frame     — cyan quad with slice_rot texture
        • Rotation axis     — red line through center along ax_x
        • Shared center     — yellow sphere
        """
        import matplotlib.pyplot as plt
        import pyvista as pv
        from utils.plot_functions import add_series_rects

        orig_np  = slice_orig.squeeze().cpu().float().numpy()   # (H_c, W_c)
        rot_np   = slice_rot.squeeze().cpu().float().numpy()    # (H_c, W_c)
        diff_np  = np.abs(orig_np - rot_np)

        angle_deg   = float(np.degrees(angle))
        cons_str    = (
            f'loss_consistency={loss_consistency.item():.4f}'
            if loss_consistency is not None else 'loss_consistency=disabled'
        )

        # ══════════════════════════════════════════════════════════════
        # Window 1 — Matplotlib: 2-D slice comparison
        # ══════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(orig_np, cmap='gray', vmin=0, vmax=1, aspect='auto')
        axes[0].set_title('slice_orig\n(volume at GT pose)', fontsize=10)
        axes[0].axis('off')

        axes[1].imshow(rot_np, cmap='gray', vmin=0, vmax=1, aspect='auto')
        axes[1].set_title(f'slice_rot\n(rotated θ={angle_deg:+.2f}°)', fontsize=10)
        axes[1].axis('off')

        vmax_diff = max(float(diff_np.max()), 1e-6)
        im_d = axes[2].imshow(diff_np, cmap='hot', vmin=0, vmax=vmax_diff, aspect='auto')
        axes[2].set_title(f'|orig − rot|\nmax={vmax_diff:.3f}   {cons_str}', fontsize=10)
        axes[2].axis('off')
        plt.colorbar(im_d, ax=axes[2], fraction=0.046, pad=0.02)

        fig.suptitle(
            f'DEBUG slice consistency loss   θ={angle_deg:+.2f}°   {cons_str}',
            fontsize=12,
        )
        plt.tight_layout()
        plt.show()

        # ══════════════════════════════════════════════════════════════
        # Window 2 — PyVista: 3-D volume + frame positions
        # ══════════════════════════════════════════════════════════════
        vol_np       = volume.detach().cpu().float().numpy()
        bias_cpu     = bias.cpu()
        ax_x_np      = ax_x.detach().cpu().float().numpy()

        gt_pos_vol     = (gt_pos.cpu()     - bias_cpu).float().numpy()   # (1, 3, 3)
        gt_pos_rot_vol = (gt_pos_rot.cpu() - bias_cpu).float().numpy()   # (1, 3, 3)
        center_vol     = gt_pos_vol[0, 0]

        ll_vol = gt_pos_vol[0, 1]; lr_vol = gt_pos_vol[0, 2]
        half_w = float(np.linalg.norm(lr_vol - ll_vol)) / 2.0
        line_a = center_vol - ax_x_np * half_w
        line_b = center_vol + ax_x_np * half_w

        orig_u8 = (np.clip(orig_np, 0, 1) * 255).astype(np.uint8)
        rot_u8  = (np.clip(rot_np,  0, 1) * 255).astype(np.uint8)

        pv_title = f'DEBUG slice consistency   θ={angle_deg:+.2f}°   {cons_str}'
        plotter  = pv.Plotter(title=pv_title)

        grid = pv.ImageData()
        grid.dimensions = np.array(vol_np.shape)
        grid.spacing    = (1.0, 1.0, 1.0)
        grid.point_data['Intensity'] = vol_np.flatten(order='F')
        plotter.add_volume(grid, scalars='Intensity', cmap='bone', opacity='sigmoid')

        add_series_rects(
            plotter, gt_pos_vol, indices=[0],
            colors='green', opacity=0.15, edge_width=3,
            frames=orig_u8[np.newaxis],
        )
        add_series_rects(
            plotter, gt_pos_rot_vol, indices=[0],
            colors='cyan', opacity=0.15, edge_width=3,
            frames=rot_u8[np.newaxis],
        )

        line_pts  = np.array([line_a, line_b], dtype=np.float32)
        line_mesh = pv.Spline(line_pts, n_points=2)
        plotter.add_mesh(line_mesh, color='red', line_width=4)

        plotter.add_points(
            center_vol.reshape(1, 3).astype(np.float32),
            color='yellow', point_size=12, render_points_as_spheres=True,
        )
        plotter.add_text(
            f'green  = original GT frame (slice_orig texture)\n'
            f'cyan   = rotated frame θ={angle_deg:+.2f}° (slice_rot texture)\n'
            f'red    = rotation axis (ax_x)\n'
            f'yellow = shared center point\n'
            f'{cons_str}',
            position='upper_left', font_size=9, color='white',
        )
        plotter.show_axes()
        plotter.set_background('black')
        plotter.show()

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
        slice_rot,
        theta,
        loss_jagged=None,
    ):
        """Visualise Loss 3 (jaggedness) on a randomly rotated slice."""
        import matplotlib.pyplot as plt
        import pyvista as pv
        from utils.plot_functions import add_series_rects

        rot_np   = slice_rot.squeeze().cpu().float().numpy()
        H_c, W_c = rot_np.shape

        drow_np = np.abs(np.diff(rot_np, axis=0))
        dcol_np = np.abs(np.diff(rot_np, axis=1))
        jagged_np = np.pad(drow_np, ((0, 1), (0, 0)), mode='edge') \
                  + np.pad(dcol_np, ((0, 0), (0, 1)), mode='edge')

        theta_deg  = float(np.degrees(theta))
        jagged_str = f'loss_jagged={loss_jagged.item():.4f}' if loss_jagged is not None else 'loss_jagged=disabled'

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(rot_np, cmap='gray', vmin=0, vmax=1, aspect='auto')
        axes[0].set_title(f'Rotated slice  θ={theta_deg:+.1f}°', fontsize=10)
        axes[0].axis('off')

        jmax = max(jagged_np.max(), 1e-6)
        im_j = axes[1].imshow(jagged_np, cmap='hot', vmin=0, vmax=jmax, aspect='auto')
        axes[1].set_title(f'Jaggedness  |Δrow|+|Δcol|\n{jagged_str}', fontsize=10)
        axes[1].axis('off')
        plt.colorbar(im_j, ax=axes[1], fraction=0.046, pad=0.02)

        fig.suptitle(f'DEBUG jaggedness loss   θ={theta_deg:+.1f}°   {jagged_str}', fontsize=12)
        plt.tight_layout()
        plt.show()

        vol_np       = volume.cpu().float().numpy()
        bias_cpu     = bias.cpu()
        ax_x_np      = ax_x.detach().cpu().float().numpy()

        gt_pos_vol     = (gt_pos.cpu()     - bias_cpu).float().numpy()
        gt_pos_rot_vol = (gt_pos_rot.cpu() - bias_cpu).float().numpy()
        center_vol     = gt_pos_vol[0, 0]

        ll_vol = gt_pos_vol[0, 1]; lr_vol = gt_pos_vol[0, 2]
        half_w = float(np.linalg.norm(lr_vol - ll_vol)) / 2.0
        line_a = center_vol - ax_x_np * half_w
        line_b = center_vol + ax_x_np * half_w

        pv_title = f'DEBUG jaggedness loss   θ={theta_deg:+.1f}°   {jagged_str}'
        plotter = pv.Plotter(title=pv_title)

        grid = pv.ImageData()
        grid.dimensions = np.array(vol_np.shape)
        grid.spacing    = (1.0, 1.0, 1.0)
        grid.point_data['Intensity'] = vol_np.flatten(order='F')
        plotter.add_volume(grid, scalars='Intensity', cmap='bone', opacity='sigmoid')

        add_series_rects(plotter, gt_pos_vol,     indices=[0], colors='green', opacity=0,    edge_width=3)
        add_series_rects(plotter, gt_pos_rot_vol, indices=[0], colors='cyan',  opacity=0,    edge_width=3)

        line_pts  = np.array([line_a, line_b], dtype=np.float32)
        line_mesh = pv.Spline(line_pts, n_points=2)
        plotter.add_mesh(line_mesh, color='red', line_width=4)

        plotter.add_points(center_vol.reshape(1, 3).astype(np.float32), color='yellow', point_size=12, render_points_as_spheres=True)
        plotter.add_text(
            f'green  = original GT frame\n'
            f'cyan   = rotated frame (θ={theta_deg:+.1f}° around ax_x)\n'
            f'red    = rotation axis (ax_x)\n'
            f'yellow = shared center point\n'
            f'{jagged_str}',
            position='upper_left', font_size=9, color='white',
        )
        plotter.show_axes()
        plotter.set_background('black')
        plotter.show()

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
        """Visualise the criterion pipeline for one training step."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import pyvista as pv
        from utils.plot_functions import add_series_rects

        vol_np    = volume.cpu().float().numpy()
        ser_cpu   = fake_series.cpu()
        bias_cpu  = bias.cpu()
        ser_vol   = ser_cpu - bias_cpu
        gt_vol    = gt_pos.cpu() - bias_cpu

        slice_np  = slice_vol.squeeze().cpu().float().numpy()
        real_np   = real_image.squeeze().cpu().float().numpy()
        diff_np   = np.abs(slice_np - real_np)
        real_u8   = (np.clip(real_np, 0, 1) * 255).astype(np.uint8)

        N          = ser_cpu.shape[0]
        mid_frame  = source[N // 2].squeeze().cpu().float().numpy()

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

        fig = plt.figure(figsize=(20, 5.5))
        gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.08)

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[3])

        ax0.imshow(slice_np, cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax0.set_title('Volume slice\n(at GT position)', fontsize=11)
        ax0.axis('off')

        ax1.imshow(real_np, cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax1.set_title(f'Real US image\n(at same position, source: {source_tag})', fontsize=11)
        ax1.axis('off')

        vmax_diff = max(float(diff_np.max()), 1e-6)
        im = ax2.imshow(diff_np, cmap='hot', vmin=0, vmax=vmax_diff, aspect='auto')
        ax2.set_title(f'|Slice − Real|\nmax={vmax_diff:.3f}   L1={loss_val:.4f}', fontsize=11)
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)

        ax3.imshow(mid_frame, cmap='gray', vmin=0, vmax=1, aspect='auto')
        ax3.set_title(f'Training window frame [N/2]\n(context, N={N})', fontsize=11)
        ax3.axis('off')

        fig.suptitle(f'DEBUG custom_criterion   {ep_str}{idx_str}\n{metric_str}', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.show()

        pv_title = (
            f'DEBUG criterion   {ep_str}{idx_str}\n'
            f'window={N} frames (red)   found GT frame (green)   {metric_str}'
        )
        plotter = pv.Plotter(title=pv_title)

        grid = pv.ImageData()
        grid.dimensions = np.array(vol_np.shape)
        grid.spacing    = (1.0, 1.0, 1.0)
        grid.point_data['Intensity'] = vol_np.flatten(order='F')
        plotter.add_volume(grid, scalars='Intensity', cmap='bone', opacity='sigmoid')

        step       = max(1, N // 8)
        win_idx    = sorted(set([0] + list(range(0, N, step)) + [N - 1]))
        add_series_rects(plotter, ser_vol, indices=win_idx, colors='red', opacity=0, edge_width=2)

        gt_vol_np = gt_vol.float().numpy()
        add_series_rects(plotter, gt_vol_np, indices=[0], colors='green', opacity=0.15, edge_width=4, frames=real_u8[np.newaxis])

        plotter.add_text(
            f'{metric_str}\nred = training window ({N} frames)\ngreen = found GT frame',
            position='upper_left', font_size=9, color='white',
        )
        plotter.show_axes()
        plotter.set_background('black')
        plotter.show()

    # ------------------------------------------------------------------
    # Training step — one scan (or window) per call
    # ------------------------------------------------------------------

    def train(self, epoch_info, sample_dict):
        """One training step on a complete scan (or a windowed sub-sequence).

        Returns:
            dict[str, Tensor]  {loss_name: detached scalar, ...}
        """
        source_full    = sample_dict['source'].to(self.device).squeeze(0)
        target_full    = sample_dict['target'].to(self.device).squeeze(0)
        gt_series_full = target_full[:, -9:].view(-1, 3, 3)
        N_full         = source_full.shape[0]

        self.backbone.eval()

        max_reco_attempts = int(getattr(self.cfg, 'max_reco_attempts', 5))
        losses = None

        for _attempt in range(max_reco_attempts):

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

            self._scan_ctx = {
                'source_full':        source_full,
                'gt_series_full':     gt_series_full,
                'window_indices_set': window_indices_set,
                'window_start':       win_start,
                'window_size':        source.shape[0],
                'epoch_info':         epoch_info,
            }

            self.optimizer.zero_grad()

            fake_gaps = self._run_backbone(source, use_checkpoint=True)

            fake_series = utils.simulation.dof_to_series(
                gt_series[0:1],
                fake_gaps.unsqueeze(0),
            ).squeeze(0)

            dense_grad = bool(getattr(self.cfg, 'dense_grad', False))
            try:
                volume, bias = self._reconstruct_volume(
                    source.detach().clone(),
                    fake_series if dense_grad else fake_series.detach(),
                    enable_grad=dense_grad,
                )
            except (ValueError, RuntimeError) as _reco_err:
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

            losses = self.custom_criterion(source, fake_gaps, fake_series, volume, bias)
            if losses is not None:
                self._component_loss_keys = self._enabled_loss_keys()

                del source, source_full
                self._scan_ctx.clear()
                torch.cuda.empty_cache()
                break

            del fake_gaps, fake_series, volume, bias
            torch.cuda.empty_cache()

        if losses is None:
            self._scan_ctx.clear()
            torch.cuda.empty_cache()
            zero = torch.tensor(0.0, device=self.device)
            return {'loss': zero, **{k: zero for k in self._component_loss_keys}}

        loss = sum(losses.values())
        loss.backward()

        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.backbone.parameters(),
                max_norm=self.grad_clip_norm,
            )

        self.optimizer.step()
        self.scheduler.step(epoch_info['epoch'])

        ret = {
            'loss': loss.detach(),
            **{k: v.detach() for k, v in losses.items()},
        }

        self._scan_ctx.clear()
        del fake_gaps, fake_series, volume, bias, losses, loss
        torch.cuda.empty_cache()

        return ret

    # ------------------------------------------------------------------
    # Eval-loss step — same forward as train() but no backward/optimizer
    # ------------------------------------------------------------------

    def eval_loss(self, epoch_info, sample_dict):
        """Forward pass + criterion without updating weights.

        Returns:
            dict[str, Tensor]  {loss_name: detached scalar, ...}
        """
        source_full    = sample_dict['source'].to(self.device).squeeze(0)
        target_full    = sample_dict['target'].to(self.device).squeeze(0)
        gt_series_full = target_full[:, -9:].view(-1, 3, 3)
        N_full         = source_full.shape[0]

        self.backbone.eval()
        max_reco_attempts = int(getattr(self.cfg, 'max_reco_attempts', 5))

        with torch.no_grad():
            for _attempt in range(max_reco_attempts):
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

                self._scan_ctx = {
                    'source_full':        source_full,
                    'gt_series_full':     gt_series_full,
                    'window_indices_set': window_indices_set,
                    'window_start':       win_start,
                    'window_size':        source.shape[0],
                    'epoch_info':         epoch_info,
                }

                fake_gaps = self._run_backbone(source, use_checkpoint=False)
                fake_series = utils.simulation.dof_to_series(
                    gt_series[0:1],
                    fake_gaps.unsqueeze(0),
                ).squeeze(0)

                try:
                    volume, bias = self._reconstruct_volume(
                        source.detach().clone(),
                        fake_series.detach(),
                        enable_grad=False,
                    )
                except (ValueError, RuntimeError):
                    del fake_gaps, fake_series
                    torch.cuda.empty_cache()
                    continue

                losses = self.custom_criterion(source, fake_gaps, fake_series, volume, bias)
                self._scan_ctx.clear()

                if losses is not None:
                    loss = sum(losses.values())
                    ret  = {'loss': loss.detach(),
                            **{k: v.detach() for k, v in losses.items()}}
                    del fake_gaps, fake_series, volume, bias, losses, loss
                    torch.cuda.empty_cache()
                    return ret

                del fake_gaps, fake_series, volume, bias
                torch.cuda.empty_cache()

        self._scan_ctx.clear()
        torch.cuda.empty_cache()
        zero = torch.tensor(0.0, device=self.device)
        return {'loss': zero, **{k: zero for k in self._component_loss_keys}}

    # ------------------------------------------------------------------
    # Test step — full scan, no gradient tracking
    # ------------------------------------------------------------------

    def test(self, epoch_info, sample_dict):
        """Evaluate on one complete scan; returns trajectory error metrics."""
        source    = sample_dict['source'].to(self.device).squeeze(0)
        target    = sample_dict['target'].to(self.device).squeeze(0)
        gt_series = target[:, -9:].view(-1, 3, 3)

        self.backbone.eval()

        fake_gaps   = self._run_backbone(source, use_checkpoint=False)
        fake_series = utils.simulation.dof_to_series(
            gt_series[0:1],
            fake_gaps.unsqueeze(0),
        ).squeeze(0)

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
        """Interactively visualise the reconstructed 3-D volume with PyVista."""
        import pyvista as pv
        from utils.plot_functions import add_series_rects

        vol_np     = volume.cpu().float().numpy()
        series_cpu = fake_series.cpu()
        series_biased = series_cpu - bias.cpu()

        frames_np  = (source.cpu().squeeze(1).numpy() * 255).astype(np.uint8)

        N = series_cpu.shape[0]

        title = 'DEBUG: reconstructed volume'
        if epoch_info is not None:
            ep  = epoch_info.get('epoch', '?')
            idx = epoch_info.get('index', '?')
            title = f'DEBUG  epoch={ep}  scan={idx}  N={N}  vol={vol_np.shape}'

        plotter = pv.Plotter(title=title)

        grid = pv.ImageData()
        grid.dimensions = np.array(vol_np.shape)
        grid.spacing    = (1, 1, 1)
        grid.point_data['Intensity'] = vol_np.flatten(order='F')
        plotter.add_volume(grid, scalars='Intensity', cmap='bone', opacity='sigmoid')

        step    = max(1, N // 6)
        indices = sorted(set([0] + list(range(0, N, step)) + [N - 1]))
        add_series_rects(plotter, series_biased, indices=indices, colors='red', opacity=0)

        plotter.show_axes()
        plotter.set_background('black')
        plotter.show()