import numpy as np
import torch
import torch.nn.functional as F

import configs
import models
import utils
import trial.my_utils.functions as my_utils
from utils.plot_functions import read_calib_matrices


class Online_Finetuning_Backbone(models.BaseModel):
    """
    Self-supervised fine-tuning framework built on the pre-trained baseline backbone.

    Usage:
        1. Override `custom_criterion` with your self-supervised loss.
        2. Adjust `res/models/online_finetune_bk.json` and `res/run/hp_finetune_bk.json`
           as needed, then run `trial/train/main_complete_finetuning_train.py`.
    """

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)
        self.backbone = models.backup.online_baseline_backbone.Backbone(
            self.data_cfg.source.channel, self.data_cfg.target.elements - 9
        ).to(self.device)

        pretrained_path = configs.env.getdir(self.cfg.pretrained_weight)
        pretrained_state = torch.load(pretrained_path, map_location=self.device)
        self.backbone.load_state_dict(pretrained_state)

        self.optimizer = torch.optim.Adam(
            self.backbone.parameters(), lr=self.run.lr, betas=self.run.betas
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.run.step_size, gamma=self.run.gamma
        )

        # ── Calibration for 3-D reconstruction ────────────────────────────────
        calib_path = configs.env.getdir(self.data_cfg.paths.calib)
        T_calib_scale, _, _ = read_calib_matrices(calib_path)
        self.scale_w = float(T_calib_scale[0, 0])   # u → W  (mm / pixel)
        self.scale_h = float(T_calib_scale[1, 1])   # v → H  (mm / pixel)

        self.down_ratio = float(getattr(self.cfg, 'down_ratio', 1.0))
        self.mat_scale = torch.eye(4, dtype=torch.float32, device=self.device)
        self.mat_scale[0, 0] = self.down_ratio
        self.mat_scale[1, 1] = self.down_ratio
        self.mat_scale[2, 2] = self.down_ratio

    # ------------------------------------------------------------------

    def _reconstruct_volume(self, source, series):
        """Reconstruct a 3-D volume from one sequence of frames and positions.

        Mirrors the pipeline used in
        ``trial/inference/infer_baseline_3d_slice_random.py``:
        downsample → ``my_utils.reco`` → upsample.

        Args:
            source  (N, 1, H, W)  float32 frames in [0, 1], on ``self.device``
            series  (N, 3, 3)     world-mm probe positions [center, LL, LR]

        Returns:
            volume  (D, H', W')  reconstructed 3-D intensity volume
            bias    (3,)         origin offset applied inside ``reco``
                                 (subtract from series to get volume-aligned coords)
        """
        # Spatially downsample frames: (N, 1, H, W) → (N, H·dr, W·dr)
        source_down = F.interpolate(
            source, scale_factor=self.down_ratio,
            mode='bilinear', align_corners=False,
        ).squeeze(1)  # (N, H·dr, W·dr)

        volume, bias = my_utils.reco(
            source_down, series,
            self.scale_w, self.scale_h,
            self.mat_scale,
        )  # volume: (D, H', W'),  bias: (3,)

        # Upsample back to original resolution
        if self.down_ratio != 1.0:
            volume = F.interpolate(
                volume.unsqueeze(0).unsqueeze(0),
                scale_factor=1.0 / self.down_ratio,
            ).squeeze(0).squeeze(0)

        return volume, bias

    # ------------------------------------------------------------------
    # Override this method to implement your self-supervised loss.
    # ------------------------------------------------------------------
    def custom_criterion(self, source, fake_gaps, fake_series, volumes):
        """
        Define your custom self-supervised loss here.

        Args:
            source      (B, N, 1, H, W)  raw input ultrasound frames in [0, 1]
            fake_gaps   (B, N-1, 6)      predicted 6-DoF inter-frame transforms
                                          [tx, ty, tz, rx, ry, rz] (descaled)
            fake_series (B, N, 3, 3)     predicted probe positions in world-mm space,
                                          each row = [center, lower-left, lower-right]
            volumes     list of B tuples (volume, bias):
                          volume (D, H', W') – reconstructed 3-D intensity volume
                          bias   (3,)        – world-mm origin offset; subtract from
                                              series before indexing into the volume

        Returns:
            dict: {loss_name: scalar_tensor, ...}
                  All values are summed to form the total loss.
        """
        raise NotImplementedError(
            "Implement your custom self-supervised criterion in "
            "`Online_Finetuning_Backbone.custom_criterion`."
        )

    # ------------------------------------------------------------------

    def train(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)   # (B, N, 1, H, W)
        real_target = sample_dict['target'].to(self.device)   # (B, N, 15)

        self.backbone.train()
        self.optimizer.zero_grad()

        inp = torch.cat([real_source[:, :-1], real_source[:, 1:]], dim=2)  # (B, N-1, 2, H, W)
        fake_target, _ = self.backbone(inp, return_feature=False)           # (B, N-1, 6)

        fake_gaps = torch.cat(
            [fake_target[:, :, :3], fake_target[:, :, 3:] / 100], dim=-1
        )  # (B, N-1, 6)

        B = real_target.shape[0]
        real_series = real_target[:, :, -9:].view(B, -1, 3, 3)             # (B, N, 3, 3)
        fake_series = utils.simulation.dof_to_series(
            real_series[:, 0], fake_gaps
        )                                                                    # (B, N, 3, 3)

        # ── Reconstruct 3-D volumes from predicted series ────────────────────
        # Each volume is built from the per-sample source frames (already in
        # [0, 1]) and the corresponding predicted world-mm probe positions,
        # mirroring the pipeline in
        # ``trial/inference/infer_baseline_3d_slice_random.py``.
        volumes = []
        for b in range(B):
            vol, bias = self._reconstruct_volume(
                real_source[b],    # (N, 1, H, W)  float32 in [0, 1]
                fake_series[b],    # (N, 3, 3)     world-mm positions
            )
            volumes.append((vol, bias))
        # ────────────────────────────────────────────────────────────────────

        losses = self.custom_criterion(real_source, fake_gaps, fake_series, volumes)
        loss = sum(losses.values())
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(epoch_info['epoch'])

        return {'loss': loss, **losses}

    def test(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device).squeeze(0)     # (N, 15)

        real_series = real_target[:, -9:].view(-1, 3, 3)                   # (N, 3, 3)

        self.backbone.eval()

        inp = torch.cat([real_source[:, :-1], real_source[:, 1:]], dim=2)
        fake_gaps, _ = self.backbone(inp)
        fake_gaps = fake_gaps[0, :, :]
        fake_gaps[:, 3:] /= 100

        fake_series = utils.simulation.dof_to_series(
            real_series[0:1], fake_gaps.unsqueeze(0)
        ).squeeze(0)
        return utils.metric.get_metric(real_series, fake_series)

    def test_return_hook(self, epoch_info, return_all):
        return_info = {
            k: np.sum(v) / epoch_info['batch_per_epoch']
            for k, v in return_all.items()
        }
        if return_info:
            self.logger.info_scalars(
                '{} Epoch: {}\t', (epoch_info['log_text'], epoch_info['epoch']), return_info
            )
        return return_all