import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import configs
import datasets
import utils
from utils.plot_functions import data_pairs_adjacent, transform_t2t, read_calib_matrices

__all__ = ['LHPerL']


class LHPerL(datasets.BaseDataset):

    @staticmethod
    def more(cfg):
        cfg.source.elements = cfg.source.width * cfg.source.height * cfg.source.channel
        cfg.paths.h5 = configs.env.getdir(cfg.paths.h5)
        cfg.paths.calib = configs.env.getdir(cfg.paths.calib)
        cfg.num_workers = 0
        cfg.pin_memory = False
        cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return cfg

    def _build_series(self, tforms_np, H_img, W_img):
        """Compute series (N, 3, 3) = [center, lower-left, lower-right] in world mm.

        Pixel anchor points match trial/recon.py:
            center      (W/2, H/2)
            lower-left  (1,   H  )
            lower-right (W,   H  )
        """
        tforms = torch.from_numpy(tforms_np)
        data_pairs = data_pairs_adjacent(tforms.shape[0])
        tforms_f2f0 = transform_t2t(tforms, torch.linalg.inv(tforms), data_pairs)

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
        return world_pts[:, :3, :].permute(0, 2, 1)  # (N, 3, 3)

    def load(self):
        with h5py.File(self.cfg.paths.h5, 'r') as f:
            frames_np = f['frames'][()]   # (N, H, W) uint8
            tforms_np = f['tforms'][()]   # (N, 4, 4) float32

        H_orig, W_orig = frames_np.shape[1], frames_np.shape[2]
        H_out, W_out = self.cfg.source.height, self.cfg.source.width

        # Normalize frames and resize if cfg dimensions differ from raw
        source_all = torch.from_numpy(frames_np.astype(np.float32) / 255.0)  # (N, H, W)
        if H_orig != H_out or W_orig != W_out:
            source_all = F.interpolate(
                source_all.unsqueeze(1), size=(H_out, W_out),
                mode='bilinear', align_corners=False,
            ).squeeze(1)

        # Recompute series with output image dimensions so anchor pixels
        # (W/2, H/2), (1, H), (W, H) correspond to the resized frames
        series = self._build_series(tforms_np, H_out, W_out)  # (N, 3, 3)

        # Split into train / val / test segments along the time axis
        N = source_all.shape[0]
        ratio = self.cfg.train_val_test_split   # e.g. [0.70, 0.15, 0.15]
        n_train = int(N * ratio[0])
        n_val   = int(N * ratio[1])

        segments = [
            (source_all[:n_train],               series[:n_train]),
            (source_all[n_train:n_train + n_val], series[n_train:n_train + n_val]),
            (source_all[n_train + n_val:],        series[n_train + n_val:]),
        ]

        source_list, target_point_list, target_dof_list = [], [], []
        for src, ser in segments:
            # tp, td = self.preprocessing(ser.view(-1, 9))
            ###############################################
            tp, td = self.preprocessing(ser.reshape(-1, 9))
            ###############################################
            source_list.append(src)
            target_point_list.append(tp)
            target_dof_list.append(td)

        trainset_length = int(self.cfg.series_per_data[0] * self.cfg.train_test_range[0])
        valset_length   = int(self.cfg.series_per_data[1] * self.cfg.train_test_range[1])
        testset_length  = int(self.cfg.series_per_data[2] * self.cfg.train_test_range[2])
        data_count = trainset_length + valset_length + testset_length

        return {
            'source': source_list,
            'target_point': target_point_list,
            'target_dof': target_dof_list,
        }, data_count

    def preprocessing(self, tp):
        tp = tp.view(-1, 3, 3)
        pall = torch.cat([
            tp,
            2 * tp[:, 0:1, :] - tp[:, 1:2, :],
            2 * tp[:, 0:1, :] - tp[:, 2:3, :],
        ], dim=1)
        min_loca = torch.min(pall.reshape(-1, 3), dim=0)[0]
        tp = tp - min_loca.unsqueeze(0).unsqueeze(0)
        td = utils.simulation.series_to_dof(tp)
        tp = tp.view(-1, 9)
        return tp, td

    def __getitem__(self, index):
        idx = self.get_idx(index)

        source = self.data['source'][idx].to(self.cfg.device)
        target_point = self.data['target_point'][idx]

        frame_rate = torch.randint(self.cfg.frame_rate[0], self.cfg.frame_rate[1] + 1, (1,))
        source = source[::frame_rate]
        target_point = target_point[::frame_rate]
        target_point, target_dof = self.preprocessing(target_point.view(-1, 3, 3))

        optical_flow = utils.image.get_optical_flow(source, device=self.cfg.device)
        edge = utils.image.get_edge(source, device=self.cfg.device)

        source_out = source.unsqueeze(1)
        target_out = torch.cat([F.pad(target_dof, (0, 0, 0, 1)), target_point.view(-1, 9)], dim=-1)

        sample_dict = {
            'source': source_out,
            'target': target_out,
            'optical_flow': optical_flow,
            'edge': edge,
            'frame_rate': frame_rate,
            'info': torch.tensor(len(source_out)),
        }

        utils.common.set_seed(int(time.time() * 1000) % (1 << 32) + index)
        return sample_dict, index