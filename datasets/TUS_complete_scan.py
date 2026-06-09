"""TUS_complete_scan — dataset that returns one *complete* ultrasound scan per item.

Unlike TUS_complete (which samples random 2-frame pairs), every __getitem__ call
loads **all N frames** of a single scan.  This enables full 3-D reconstruction
inside the training loop.

Returned sample_dict
--------------------
source  (N, 1, H, W)   float32 frames in [0, 1]
series  (N, 3, 3)      normalised GT probe positions [center, lower-left, lower-right]
                        in world-mm (offset so the bounding box starts near the origin)
target  (N, 15)        gaps_padded (N, 6) | series_flat (N, 9)
                        – same layout as the baseline dataset, useful for eval metrics

Notes
-----
* series_per_data is forced to [1, 1, 1]: each scan appears exactly once per epoch.
* Frames are loaded lazily (only paths are stored at load() time).
* num_workers=0, pin_memory=False (frames already moved to cfg.device in __getitem__).
"""

import glob
import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import configs
import utils
from datasets.BaseDataset import BaseDataset
from utils.plot_functions import data_pairs_adjacent, transform_t2t, read_calib_matrices

__all__ = ['TUS_complete_scan']


class TUS_complete_scan(BaseDataset):

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def more(cfg):
        cfg.source.elements = cfg.source.width * cfg.source.height * cfg.source.channel
        cfg.paths.h5   = configs.env.getdir(cfg.paths.h5)
        cfg.paths.calib = configs.env.getdir(cfg.paths.calib)
        if not hasattr(cfg, 'train_val_test_split'):
            cfg.train_val_test_split = [0.6, 0.2, 0.2]
        # One complete scan maps to exactly one dataset index
        cfg.series_per_data = [1, 1, 1]
        cfg.num_workers  = 0
        cfg.pin_memory   = False
        cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return cfg

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _build_series(self, tforms_np, H_img, W_img):
        """Compute series (N, 3, 3) = [center, lower-left, lower-right] in world mm."""
        tforms = torch.from_numpy(tforms_np)                          # (N, 4, 4)
        data_pairs   = data_pairs_adjacent(tforms.shape[0])
        tforms_f2f0  = transform_t2t(tforms, torch.linalg.inv(tforms), data_pairs)

        _, tform_calib_R_T, tform_calib = read_calib_matrices(self.cfg.paths.calib)
        T_combined = torch.matmul(
            torch.linalg.inv(tform_calib_R_T).unsqueeze(0),
            torch.matmul(tforms_f2f0, tform_calib.unsqueeze(0)),
        )  # (N, 4, 4)

        pixel_pts = torch.tensor(
            [[W_img / 2.0,  H_img / 2.0, 0.0, 1.0],
             [1.0,          float(H_img), 0.0, 1.0],
             [float(W_img), float(H_img), 0.0, 1.0]],
            dtype=T_combined.dtype,
        ).T  # (4, 3)

        N = T_combined.shape[0]
        world_pts = torch.bmm(T_combined, pixel_pts.unsqueeze(0).expand(N, 4, 3))
        return world_pts[:, :3, :].permute(0, 2, 1)   # (N, 3, 3)

    def _normalise_series(self, series):
        """Shift series so its bounding box starts at the origin.

        Returns
        -------
        series_norm  (N, 3, 3)   normalised series
        target_dof   (N-1, 6)    6-DoF inter-frame transforms
        target_point (N, 9)      series_norm flattened to (N, 9)
        """
        # Include all four corners to compute the tight bounding box
        pall = torch.cat([
            series,
            2 * series[:, 0:1, :] - series[:, 1:2, :],
            2 * series[:, 0:1, :] - series[:, 2:3, :],
        ], dim=1)                                          # (N, 5, 3)
        min_loca    = torch.min(pall.reshape(-1, 3), dim=0)[0]  # (3,)
        series_norm = series - min_loca.unsqueeze(0).unsqueeze(0)  # (N, 3, 3)

        series_norm  = series_norm.contiguous()
        target_dof   = utils.simulation.series_to_dof(series_norm)  # (N-1, 6)
        target_point = series_norm.reshape(-1, 9)                    # (N, 9)
        return series_norm, target_dof, target_point

    # ------------------------------------------------------------------
    # Dataset loading (metadata only – frames are lazy-loaded)
    # ------------------------------------------------------------------

    def load(self):
        root   = self.cfg.paths.h5
        H_out, W_out = self.cfg.source.height, self.cfg.source.width

        subjects = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        if not subjects:
            raise FileNotFoundError(f'No subject directories found under {root}')

        # Subject-level split (no data leakage between train / val / test)
        n_subs   = len(subjects)
        ratio    = self.cfg.train_val_test_split
        ratio    = [r / sum(ratio) for r in ratio]
        set_sizes = [int(n_subs * r) for r in ratio]
        for ii in range(n_subs - sum(set_sizes)):
            set_sizes[ii] += 1
        boundaries    = [sum(set_sizes[:i]) for i in range(len(set_sizes) + 1)]
        split_subjects = [
            subjects[boundaries[i]:boundaries[i + 1]]
            for i in range(len(set_sizes))
        ]
        split_files = [
            [f for sub in subs
               for f in sorted(glob.glob(os.path.join(root, sub, '*.h5')))]
            for subs in split_subjects
        ]

        scan_list = []   # ordered: all train scans, then val, then test

        for files in split_files:
            for h5_path in files:
                with h5py.File(h5_path, 'r') as f:
                    tforms_np = f['tforms'][()]    # (N, 4, 4) float32
                    n_frames  = f['frames'].shape[0]

                series = self._build_series(tforms_np, H_out, W_out)  # (N, 3, 3)
                series_norm, target_dof, target_point = self._normalise_series(series)

                scan_list.append({
                    'path':         h5_path,
                    'n_frames':     n_frames,
                    'series_norm':  series_norm,    # (N, 3, 3)  normalised GT
                    'target_dof':   target_dof,     # (N-1, 6)
                    'target_point': target_point,   # (N, 9)
                })

        # Update train_test_range to actual scan counts
        self.cfg.train_test_range = [
            len(split_files[0]), len(split_files[1]), len(split_files[2])
        ]
        data_count = sum(
            self.cfg.train_test_range[i] * self.cfg.series_per_data[i]
            for i in range(3)
        )
        return {'scans': scan_list}, data_count

    # ------------------------------------------------------------------
    # Item retrieval — returns the full scan
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        idx  = self.get_idx(index)
        scan = self.data['scans'][idx]

        H_out, W_out = self.cfg.source.height, self.cfg.source.width

        # ── Load all frames from disk ──────────────────────────────────
        with h5py.File(scan['path'], 'r') as f:
            frames_np = f['frames'][()]          # (N, H_orig, W_orig) uint8

        source = torch.from_numpy(frames_np.astype(np.float32) / 255.0)  # (N, H, W)
        H_orig, W_orig = source.shape[1], source.shape[2]
        if H_orig != H_out or W_orig != W_out:
            source = F.interpolate(
                source.unsqueeze(1), size=(H_out, W_out),
                mode='bilinear', align_corners=False,
            ).squeeze(1)
        source = source.unsqueeze(1)             # (N, 1, H, W)

        # ── GT series and target ───────────────────────────────────────
        series_norm  = scan['series_norm']       # (N, 3, 3)
        target_dof   = scan['target_dof']        # (N-1, 6)
        target_point = scan['target_point']      # (N, 9)

        # target layout: gaps_padded (N, 6) | series_flat (N, 9)  → (N, 15)
        # (last gap row is zero-padded to keep a consistent shape)
        gaps_padded = F.pad(target_dof, (0, 0, 0, 1))       # (N, 6)
        target      = torch.cat([gaps_padded, target_point], dim=-1)  # (N, 15)

        sample_dict = {
            'source': source,       # (N, 1, H, W)
            'series': series_norm,  # (N, 3, 3)   normalised GT positions
            'target': target,       # (N, 15)
        }

        utils.common.set_seed(int(time.time() * 1000) % (1 << 32) + index)
        return sample_dict, index
