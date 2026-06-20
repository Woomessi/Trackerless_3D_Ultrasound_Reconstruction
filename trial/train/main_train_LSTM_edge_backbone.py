import argparse
import glob
import os
import re
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import configs
import datasets
import models
import utils

# from utils.loader import Dataset


class Main(object):

    def __init__(self):
        self.model_cfg = configs.BaseConfig('/home/wu/Documents/projects/my_projects/Trackerless_3D_Ultrasound_Reconstruction/res/models/online_LSTM_edge_bk.json')

        self.run_cfg = configs.Run('/home/wu/Documents/projects/my_projects/Trackerless_3D_Ultrasound_Reconstruction/res/run/hp_bk.json',
                                   gpus='0')
        # self.run_cfg = configs.Run('/home/wu/Documents/projects/cloned_repositories/RecON/res/run/hp_bk.json')

        self.dataset_cfg = datasets.functional.common.more(configs.BaseConfig('/home/wu/Documents/projects/my_projects/Trackerless_3D_Ultrasound_Reconstruction/res/datasets/TUS_LSTM_edge_complete.json'))

        self._init()
        self._get_component()
        self.show_cfgs()

    def _init(self):
        utils.common.set_seed(0)
        self.msg = {}

    def _get_component(self):
        self.path = utils.common.get_path(self.model_cfg, self.dataset_cfg, self.run_cfg)
        self.logger = utils.Logger(self.path, utils.common.get_filename(self.model_cfg._path))

        self.dataset = datasets.functional.common.find(self.dataset_cfg.name)(self.dataset_cfg, logger=self.logger)
        self.model = models.functional.common.find(self.model_cfg.name)(
            self.model_cfg, self.dataset.cfg, self.run_cfg, dataset=self.dataset, logger=self.logger, main_msg=self.msg)
        self.start_epoch = self.model.load(None)

    def show_cfgs(self):
        self.logger.info(self.model.cfg)
        self.logger.info(self.run_cfg)
        self.logger.info(self.dataset.cfg)

    def split(self):
        self.trainset, self.valset, self.testset = self.dataset.split()

        self.train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.run_cfg.batch_size,
            shuffle=True,
            collate_fn=getattr(self.trainset.dataset, 'collate_fn', None),
            num_workers=self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
            sampler=None
        )

        self.run_cfg.test_batch_size = getattr(self.run_cfg, 'test_batch_size', self.run_cfg.batch_size)
        self.val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.run_cfg.test_batch_size,
            shuffle=False,
            collate_fn=getattr(self.valset.dataset, 'collate_fn', None),
            num_workers=self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
            sampler=None
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.run_cfg.test_batch_size,
            shuffle=False,
            collate_fn=getattr(self.testset.dataset, 'collate_fn', None),
            num_workers=self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
            sampler=None
        )

    def train(self, epoch):
        utils.common.set_seed(int(time.time()) + epoch)
        torch.cuda.empty_cache()
        count, loss_all = 0, {}
        batch_per_epoch, count_data = len(self.train_loader), len(self.train_loader.dataset)
        log_step = 1
        epoch_info = {'epoch': epoch, 'batch_per_epoch': batch_per_epoch, 'count_data': count_data}
        for batch_idx, (sample_dict, index) in enumerate(self.train_loader):
            _count = len(list(sample_dict.values())[0])
            epoch_info['batch_idx'] = batch_idx
            epoch_info['index'] = index
            epoch_info['batch_count'] = _count
            loss_dict = self.model.train(epoch_info, sample_dict)
            loss_dict['_count'] = _count
            utils.common.merge_dict(loss_all, loss_dict)
            count += _count
            if batch_idx % log_step == 0:
                self.logger.info_scalars('Train Epoch: {} [{}/{} ({:.0f}%)]\t', (epoch, count, count_data, 100. * count / count_data), loss_dict)
        if epoch % self.run_cfg.save_step == 0:
            loss_file = os.path.join(self.path, self.model.name + '_' + str(epoch) + configs.env.paths.loss_file)
            self.logger.save_npy(loss_file, {k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for k, v in loss_all.items()})
        loss_all = self.model.train_return_hook(epoch_info, loss_all)
        self.logger.info_scalars('Train Epoch: {}\t', (epoch,), loss_all)
        if epoch % self.run_cfg.save_step == 0:
            self.model.save(epoch)
            self.plot_loss_curve()

    def test(self, epoch, data_loader=None, log_text=None):
        utils.common.set_seed(int(time.time()) + epoch)
        torch.cuda.empty_cache()
        predict, count = {}, 0
        data_loader = data_loader or self.test_loader
        log_text = log_text or 'Test'
        with torch.no_grad():
            batch_per_epoch, count_data = len(data_loader), len(data_loader.dataset)
            log_step = max(int(np.power(10, np.floor(np.log10(batch_per_epoch / 10)))), 1) if batch_per_epoch > 0 else 1
            epoch_info = {'epoch': epoch, 'batch_per_epoch': batch_per_epoch, 'count_data': count_data, 'log_text': log_text}
            for batch_idx, (sample_dict, index) in enumerate(data_loader):
                _count = len(list(sample_dict.values())[0])
                epoch_info['batch_idx'] = batch_idx
                epoch_info['index'] = index
                epoch_info['batch_count'] = _count
                output_dict = self.model.test(epoch_info, sample_dict)
                count += _count
                if batch_idx % log_step == 0:
                    self.logger.info('{} Epoch: {} [{}/{} ({:.0f}%)]'.format(log_text, epoch, count, count_data, 100. * count / count_data))
                for name, value in output_dict.items():
                    v = value.float() if value.shape else value.unsqueeze(0)
                    v = v.cpu().numpy()
                    predict[name] = np.concatenate([predict[name], v]) if name in predict.keys() else v
        predict = self.model.test_return_hook(epoch_info, predict)
        predict_file = os.path.join(self.path, self.model.name + '_' + str(epoch) + configs.env.paths.predict_file)
        self.logger.save_npy(predict_file, predict)

    def val_test(self, epoch):
        self.test(epoch, data_loader=self.val_loader, log_text='Val')
        # self.test(epoch, data_loader=self.test_loader, log_text='Test')
        self.test_loss(epoch, data_loader=self.val_loader, prefix='val')
        # self.test_loss(epoch, data_loader=self.test_loader, prefix='test')

    def test_loss(self, epoch, data_loader=None, prefix='test'):
        """Compute loss on a split without gradient updates and save to *_{prefix}_loss.npy."""
        torch.cuda.empty_cache()
        data_loader = data_loader or self.test_loader
        loss_all = {}
        batch_per_epoch = len(data_loader)

        epoch_info = {'epoch': epoch, 'batch_per_epoch': batch_per_epoch, 'count_data': batch_per_epoch}

        for batch_idx, (sample_dict, index) in enumerate(data_loader):
            _count = len(list(sample_dict.values())[0])
            epoch_info['batch_idx'] = batch_idx
            epoch_info['index'] = index
            epoch_info['batch_count'] = _count

            loss_dict = self.model.eval_loss(epoch_info, sample_dict)
            loss_dict['_count'] = _count
            utils.common.merge_dict(loss_all, loss_dict)

        loss_file = os.path.join(
            self.path,
            self.model.name + '_' + str(epoch) + '_' + prefix + '_loss.npy',
        )
        self.logger.save_npy(
            loss_file,
            {k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
             for k, v in loss_all.items()},
        )
        self.logger.info('{} loss saved to {}'.format(prefix.capitalize(), loss_file))
        self.plot_loss_curve()

    def _load_loss_files(self, suffix):
        """Load per-epoch loss .npy files matching *<suffix> and return (epochs, loss_data)."""
        regex   = re.compile(r'_(\d+)' + re.escape(suffix) + r'$')
        pattern = os.path.join(self.path, self.model.name + '_*' + suffix)
        files   = sorted(
            [p for p in glob.glob(pattern) if regex.search(p)],
            key=lambda p: int(regex.search(p).group(1)),
        )
        epochs, loss_data = [], {}
        for fpath in files:
            m = regex.search(fpath)
            if m is None:
                continue
            ep   = int(m.group(1))
            data = np.load(fpath, allow_pickle=True).item()
            count     = np.array(data.pop('_count', None) or [1.0], dtype=np.float32)
            count_sum = count.sum()
            epochs.append(ep)
            for key, val in data.items():
                val    = np.array(val, dtype=np.float32).flatten()
                scalar = float(np.dot(count[:len(val)], val[:len(count)]) / count_sum)
                loss_data.setdefault(key, []).append(scalar)
        return epochs, loss_data

    def plot_loss_curve(self):
        train_epochs, train_data = self._load_loss_files('_loss.npy')
        val_epochs,   val_data   = self._load_loss_files('_val_loss.npy')
        if not train_epochs and not val_epochs:
            return

        all_keys = ['loss'] + sorted({
            k for d in (train_data, val_data) for k in d if k != 'loss'
        })
        n     = len(all_keys)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes_flat  = axes.flatten()

        for i, key in enumerate(all_keys):
            ax = axes_flat[i]
            plotted = False
            if key in train_data and train_epochs:
                ax.plot(train_epochs, train_data[key], marker='o', markersize=3,
                        color='steelblue', label='train')
                plotted = True
            if key in val_data and val_epochs:
                ax.plot(val_epochs, val_data[key], marker='^', markersize=3,
                        color='seagreen', label='val')
                plotted = True
            ax.set_title(key)
            ax.set_xlabel('Epoch')
            ax.grid(True)
            if plotted:
                ax.legend(fontsize=8)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle('Training Loss Curves', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.path, 'loss_curve.png')
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        self.logger.info('Loss curve saved to {}'.format(save_path))


def run():

    main = Main()
    main.split()

    # if main.start_epoch == 0:
    #     main.val_test(main.start_epoch)
    for epoch in range(main.start_epoch + 1, main.run_cfg.epochs + 1):
        main.train(epoch)
        if epoch % main.run_cfg.save_step == 0:
            main.val_test(epoch)

if __name__ == '__main__':
    run()
