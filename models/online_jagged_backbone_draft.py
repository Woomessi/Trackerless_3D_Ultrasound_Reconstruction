import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from fontTools.misc.py23 import tounicode
from torchvision.models import efficientnet_b1

import models
import utils
import pyvista as pv
import matplotlib.pyplot as plt
from utils.plot_functions import add_series_rects


class Backbone(nn.Module):

    def __init__(self, in_planes, num_classes):
        super().__init__()
        # self.resnet = timm.create_model('resnet18', pretrained=True, in_chans=in_planes, num_classes=0, global_pool='')
        # self.lstm = models.layers.convolutional_rnn.Conv2dLSTM(512, 512, kernel_size=3, batch_first=True)
        # self.avg = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512, num_classes)

        # self.efficientnet_b1 = efficientnet_b1(weights=None)
        # self.efficientnet_b1 = timm.create_model('efficientnet_b1', pretrained=True, in_chans=in_planes, num_classes=num_classes, global_pool='')
        self.efficientnet_b1 = timm.create_model('efficientnet_b1', pretrained=True, in_chans=in_planes, num_classes=num_classes)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.epoch = 1

    def forward(self, x, return_feature=False):
        b, t, c, h, w = x.shape
        x = (x - torch.mean(x, dim=[3, 4], keepdim=True)) / (torch.std(x, dim=[3, 4], keepdim=True) + 1e-6)
        x = x.view(b * t, c, h, w)

        # x = self.resnet(x)
        x = self.efficientnet_b1(x)

        x = x.view(b, t, *x.shape[1:])

        if return_feature:
            f = self.avg(x)
            f = f.view(f.size(0), f.size(1), -1)
        else:
            f = None

        # x = self.lstm(x)[0]
        # x = self.avg(x)
        # x = x.view(x.size(0), x.size(1), -1)
        # x = self.fc(x)

        return x, f


class Online_Jagged_Backbone(models.BaseModel):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)
        self.backbone = Backbone(self.data_cfg.source.channel, self.data_cfg.target.elements - 9).to(self.device)
        self.mat_scale = torch.eye(4, dtype=torch.float32, device=self.device)
        self.mat_scale[0, 0] = self.cfg.down_ratio
        self.mat_scale[1, 1] = self.cfg.down_ratio
        self.mat_scale[2, 2] = self.cfg.down_ratio
        self.optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.run.lr, betas=self.run.betas)
        # self.optimizer_g = torch.optim.Adam(self.backbone.parameters(), lr=self.run.lr, betas=self.run.betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.run.step_size, gamma=self.run.gamma)
        self.flag_motion = True

    def criterion(self, real_target, fake_target, feature=None):
        real_dist, real_angle = real_target.split([3, self.data_cfg.target.elements - 12], dim=-1)
        fake_dist, fake_angle = fake_target.split([3, self.data_cfg.target.elements - 12], dim=-1)

        loss_dist = F.l1_loss(real_dist, fake_dist) * 3
        loss_angle = F.l1_loss(real_angle, fake_angle) * 3
        loss_corr = utils.metric.correlation_loss(real_target, fake_target)

        loss_dict = {'loss_dist': loss_dist, 'loss_angle': loss_angle, 'loss_corr': loss_corr}

        if self.flag_motion:
            fake_motion = torch.norm(fake_dist, p=2, dim=-1) + 1e-6
            feature = torch.norm(feature, p=2, dim=-1)
            loss_motion = torch.mean(feature / fake_motion) * self.cfg.weight_motion
            loss_dict['loss_motion'] = loss_motion

        return loss_dict

    def train(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)
        #####################################
        real_source_0_1 = real_source[:,:2,:,:,:]
        #####################################

        real_target = sample_dict['target'].to(self.device)
        real_gaps = real_target[:, :-1, :-9]

        edge = sample_dict['edge'].to(self.device)
        optical_flow = sample_dict['optical_flow'].to(self.device)

        ####################################################
        B = real_target.shape[0]
        N = real_target.shape[1]
        real_series = real_target[:, :, -9:].view(B, N, 3, 3)
        ####################################################

        ###############################################
        real_gaps[:, :, 3:] = real_gaps[:, :, 3:] * 100
        ###############################################

        ############################
        # self.backbone.train()
        # self.optimizer.zero_grad()
        with torch.no_grad():
            self.backbone.eval()
        ###########################

            input_0_1 = torch.cat([real_source_0_1[:, :-1, ...], real_source_0_1[:, 1:, ...]], dim=2)
            fake_gaps_0_1, feature = self.backbone(input_0_1, return_feature=self.flag_motion)

            ###########################
            input_reco = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...]], dim=2)
            fake_gaps_reco, _ = self.backbone(input_reco, return_feature=False)

            down_source = F.interpolate(real_source.squeeze(-3), scale_factor=self.cfg.down_ratio).unsqueeze(-3)

            fake_gaps_reco = torch.cat([fake_gaps_reco[:, :, :3], fake_gaps_reco[:, :, 3:] / 100], dim=-1)

            # fake_series = utils.simulation.dof_to_series(real_series[0:1, :, :], fake_gaps_reco.unsqueeze(0)).squeeze(0)
            fake_series = utils.simulation.dof_to_series(real_series[:, 0, :, :], fake_gaps_reco)

            for b in range(B):
                reco, min_point = utils.reconstruction.reco(down_source[b].squeeze(1),
                                                        fake_series[b], mat_scale=self.mat_scale)
                r_rec = F.interpolate(reco.unsqueeze(0).unsqueeze(0), scale_factor=1 / self.cfg.down_ratio).squeeze(0).squeeze(0)
                start_point = fake_series[b][N//2].unsqueeze(0)  # 注意 real_series 和 fake_series 的选择，以及是否需要 unsqueeze

                dof = torch.tensor([
                    [
                        [0., 0., 0., torch.pi / 2, 0., 0.]
                    ]
                ], device=self.device)

                slicer = utils.simulation.dof_to_series(start_point, dof).squeeze(0)[1].unsqueeze(0)  # (1, 3, 3)
                slicer_biased = slicer - min_point.unsqueeze(0).unsqueeze(0)

                # slicer = utils.simulation.dof_to_series(start_point, dof)
                # slicer = slicer.squeeze(0)
                # slicer_biased = slicer[1].unsqueeze(0) - min_point.unsqueeze(0).unsqueeze(0)

                slice = utils.reconstruction.get_slice(r_rec, slicer_biased, real_source.shape[-2:])
                slice = slice.squeeze(0, 1, 2)

                # ── Visualise slice ──────────────────────────────────────
                fig, axes = plt.subplots(1, 1, figsize=(3, 3), squeeze=False)
                fig.suptitle("slice from r_rec")

                # orig = real_source[b, 0, 0].detach().cpu()
                # axes[0, 0].imshow(orig, cmap="gray", vmin=0, vmax=1)
                # axes[0, 0].axis("off")

                axes[0, 0].imshow(slice.detach().cpu(), cmap="gray", vmin=0, vmax=1)
                axes[0, 0].axis("off")

                plt.tight_layout()
                plt.show()

                # ── 3D volume rendering with PyVista ─────────────────────
                r_rec_cpu = r_rec.detach().cpu()
                series_biased = (fake_series[b] - min_point).detach().cpu()
                slicer_biased_cpu = slicer_biased.detach().cpu()
                source_frames = real_source[b, :, 0].detach().cpu()  # (N, H, W)

                plotter = pv.Plotter(title='r_rec volume')
                vol_np = r_rec_cpu.numpy()
                grid = pv.ImageData()
                grid.dimensions = np.array(vol_np.shape)
                grid.spacing = (1, 1, 1)
                grid.point_data["Intensity"] = vol_np.flatten(order="F")
                plotter.add_volume(grid, scalars="Intensity", cmap="bone", opacity="sigmoid")

                add_series_rects(plotter, series_biased,
                                 indices=[0, N // 2, N - 1], colors='red',
                                 opacity=1, frames=source_frames)

                # add_series_rects(plotter, series_biased,
                #                  indices=[0, N - 1], colors='red',
                #                  opacity=1)

                add_series_rects(plotter, slicer_biased_cpu, indices=[0], colors='blue', opacity=1)

                plotter.show_axes()
                plotter.set_background('black')
                plotter.show()

            ###########################

            losses = self.criterion(real_gaps, fake_gaps_0_1, feature)
            loss = sum(losses.values())
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(epoch_info['epoch'])

        # #################
        # ### 2 channel ###
        # #################
        # input = torch.cat([real_source_0_1[:, :-1, ...], real_source_0_1[:, 1:, ...]], dim=2)
        # fake_gaps_0_1, feature = self.backbone(input, return_feature=self.flag_motion)
        #
        # input_reco = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...]], dim=2)
        # fake_gaps, _ = self.backbone(input_reco, return_feature=False)
        #
        # down_source = F.interpolate(real_source.squeeze(-3), scale_factor=self.cfg.down_ratio).unsqueeze(-3)
        #
        # fake_gaps = torch.cat([fake_gaps[0, :, :3], fake_gaps[0, :, 3:] / 100], dim=-1)
        # fake_series = utils.simulation.dof_to_series(real_series[0:1, :, :], fake_gaps.unsqueeze(0)).squeeze(0)
        #
        # reco, min_point = utils.reconstruction.reco(down_source,
        #                                             fake_series, mat_scale=self.mat_scale)
        # r_rec = F.interpolate(reco.unsqueeze(0).unsqueeze(0), scale_factor=1 / self.cfg.down_ratio).squeeze(0).squeeze(
        #     0)
        # start_point = real_series[0].unsqueeze(0) # 注意 real_series 和 fake_series 的选择，以及是否需要 unsqueeze
        # dof = torch.tensor([
        #     [
        #         [0., 0., 0., torch.pi/2, 0., 0.]
        #     ]
        # ])
        # slicer = utils.simulation.dof_to_series(start_point, dof)
        # slicer = slicer.squeeze(0)
        # slicer_biased = slicer[1].unsqueeze(0) - min_point.unsqueeze(0).unsqueeze(0)
        # slices = utils.reconstruction.get_slice(r_rec, slicer_biased, real_source.shape[-2:])
        # slice = slice.squeeze(0, 1, 2)
        #
        # losses = self.criterion(real_gaps, fake_gaps_0_1, feature)
        # loss = sum(losses.values())
        # loss.backward()
        # self.optimizer.step()
        # self.scheduler.step(epoch_info['epoch'])

        return {'loss': loss, **losses}

    def test(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device).squeeze(0)
        edge = sample_dict['edge'].to(self.device)
        optical_flow = sample_dict['optical_flow'].to(self.device)

        real_series = real_target[:, -9:].view(-1, 3, 3)

        self.backbone.eval()

        #################
        ### 6 channel ###
        #################
        # input = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...], edge[:, :-1, ...], edge[:, 1:, ...], optical_flow], dim=2)

        #################
        ### 2 channel ###
        #################
        input = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...]], dim=2)

        fake_gaps, _ = self.backbone(input)
        fake_gaps = fake_gaps[0, :, :]
        fake_gaps[:, 3:] /= 100

        fake_series = utils.simulation.dof_to_series(real_series[0:1, :, :], fake_gaps.unsqueeze(0)).squeeze(0)
        losses = utils.metric.get_metric(real_series, fake_series)

        return losses

    def test_return_hook(self, epoch_info, return_all):
        return_info = {}
        for key, value in return_all.items():
            return_info[key] = np.sum(value) / epoch_info['batch_per_epoch']
        if return_info:
            self.logger.info_scalars('{} Epoch: {}\t', (epoch_info['log_text'], epoch_info['epoch']), return_info)
        return return_all
