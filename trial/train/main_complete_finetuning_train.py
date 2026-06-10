"""Scan-level self-supervised fine-tuning training script.

Each training step processes one *complete* ultrasound scan:
    1. Backbone infers inter-frame gaps for all N-1 adjacent pairs.
    2. dof_to_series reconstructs the probe trajectory (N frames).
    3. my_utils.reco builds the 3-D intensity volume.
    4. Online_Finetuning_Backbone.custom_criterion() applies the
       self-supervised loss on the volume.

The same reconstructed volume can be reused for multiple self-supervised
signals by returning several loss terms from custom_criterion().

Dataset  : TUS_complete_scan  (series_per_data=[1,1,1] → one scan per epoch step)
Model    : Online_Finetuning_Backbone
Configs  : res/models/online_finetune_bk.json
           res/run/hp_finetune_bk.json
           res/datasets/TUS_complete_scan.json
"""

import glob
import os
import re
import sys
import time

# Ensure the project root (two levels up from this script) is on sys.path so
# that `import configs`, `import datasets`, etc. work when the script is run
# directly (e.g. `python trial/train/main_complete_finetuning_train.py`).
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# Must be set before any CUDA memory is allocated (i.e. before `import torch`
# initialises the CUDA context).  expandable_segments lets the allocator grow
# and shrink memory segments on demand, eliminating the "reserved but
# unallocated" fragmentation that causes OOM during backbone recomputation.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import matplotlib.pyplot as plt
import numpy as np
import torch

import configs
import datasets
import models
import utils


class Main(object):

    def __init__(self):
        self.model_cfg = configs.BaseConfig(
            os.path.join(_PROJ_ROOT, 'res/models/online_finetune_bk_cpu.json')
        )
        self.run_cfg = configs.Run(
            os.path.join(_PROJ_ROOT, 'res/run/hp_finetune_bk.json'),
            gpus='0'
        )
        # Complete-scan dataset: each __getitem__ returns all N frames of one scan
        self.dataset_cfg = datasets.functional.common.more(
            configs.BaseConfig(
                os.path.join(_PROJ_ROOT, 'res/datasets/TUS_complete_scan.json')
            )
        )

        self._init()
        self._get_component()
        self.show_cfgs()

    def _init(self):
        utils.common.set_seed(0)
        self.msg = {}

    def _get_component(self):
        self.path   = utils.common.get_path(self.model_cfg, self.dataset_cfg, self.run_cfg)
        self.logger = utils.Logger(self.path, utils.common.get_filename(self.model_cfg._path))

        self.dataset = datasets.functional.common.find(self.dataset_cfg.name)(
            self.dataset_cfg, logger=self.logger
        )
        self.model = models.functional.common.find(self.model_cfg.name)(
            self.model_cfg, self.dataset.cfg, self.run_cfg,
            dataset=self.dataset, logger=self.logger, main_msg=self.msg
        )
        self.start_epoch = self.model.load(None)

    def show_cfgs(self):
        self.logger.info(self.model.cfg)
        self.logger.info(self.run_cfg)
        self.logger.info(self.dataset.cfg)

    # ------------------------------------------------------------------
    # DataLoader construction
    # ------------------------------------------------------------------

    def split(self):
        self.trainset, self.valset, self.testset = self.dataset.split()

        # batch_size=1: each batch is one complete scan.
        # Scans have variable length (N frames), so larger batch sizes require
        # a custom collate with padding — not needed for batch_size=1.
        self.train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=True,
            collate_fn=getattr(self.trainset.dataset, 'collate_fn', None),
            num_workers=self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
            sampler=None,
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=1,
            shuffle=False,
            collate_fn=getattr(self.valset.dataset, 'collate_fn', None),
            num_workers=self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
            sampler=None,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=1,
            shuffle=False,
            collate_fn=getattr(self.testset.dataset, 'collate_fn', None),
            num_workers=self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
            sampler=None,
        )

    # ------------------------------------------------------------------
    # Training loop — one epoch = one pass over all training scans
    # ------------------------------------------------------------------

    def train(self, epoch):
        utils.common.set_seed(int(time.time()) + epoch)
        torch.cuda.empty_cache()

        count, loss_all = 0, {}
        n_scans   = len(self.train_loader)       # total number of training scans
        log_step  = 1                            # log every scan (scans are slow)

        epoch_info = {
            'epoch':           epoch,
            'batch_per_epoch': n_scans,
            'count_data':      n_scans,
        }

        for scan_idx, (sample_dict, index) in enumerate(self.train_loader):
            # sample_dict tensors have a leading batch dim of 1 (e.g. source: (1,N,1,H,W))
            # The model's train() squeezes that dim internally.
            n_frames = sample_dict['source'].shape[1]   # N for this scan

            epoch_info['batch_idx']   = scan_idx
            epoch_info['index']       = index
            epoch_info['batch_count'] = 1               # always 1 scan per step

            loss_dict = self.model.train(epoch_info, sample_dict)
            loss_dict['_count'] = 1

            utils.common.merge_dict(loss_all, loss_dict)
            count += 1

            if scan_idx % log_step == 0:
                self.logger.info_scalars(
                    'Train Epoch: {} [scan {}/{} ({:.0f}%), N={}]\t',
                    (epoch, count, n_scans, 100.0 * count / n_scans, n_frames),
                    loss_dict,
                )

        if epoch % self.run_cfg.save_step == 0:
            loss_file = os.path.join(
                self.path,
                self.model.name + '_' + str(epoch) + configs.env.paths.loss_file,
            )
            self.logger.save_npy(
                loss_file,
                {k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
                 for k, v in loss_all.items()},
            )

        loss_all = self.model.train_return_hook(epoch_info, loss_all)
        self.logger.info_scalars('Train Epoch: {}\t', (epoch,), loss_all)

        if epoch % self.run_cfg.save_step == 0:
            self.model.save(epoch)
            self.plot_loss_curve()

    # ------------------------------------------------------------------
    # Test / validation loop
    # ------------------------------------------------------------------

    def test(self, epoch, data_loader=None, log_text=None):
        utils.common.set_seed(int(time.time()) + epoch)
        torch.cuda.empty_cache()

        predict    = {}
        count      = 0
        data_loader = data_loader or self.test_loader
        log_text    = log_text    or 'Test'

        n_scans  = len(data_loader)
        log_step = max(int(np.power(10, np.floor(np.log10(max(n_scans / 10, 1))))), 1)

        epoch_info = {
            'epoch':           epoch,
            'batch_per_epoch': n_scans,
            'count_data':      n_scans,
            'log_text':        log_text,
        }

        with torch.no_grad():
            for scan_idx, (sample_dict, index) in enumerate(data_loader):
                epoch_info['batch_idx']   = scan_idx
                epoch_info['index']       = index
                epoch_info['batch_count'] = 1

                output_dict = self.model.test(epoch_info, sample_dict)
                count += 1

                if scan_idx % log_step == 0:
                    self.logger.info(
                        '{} Epoch: {} [{}/{} ({:.0f}%)]'.format(
                            log_text, epoch, count, n_scans,
                            100.0 * count / n_scans,
                        )
                    )

                for name, value in output_dict.items():
                    v = value.float() if value.shape else value.unsqueeze(0)
                    v = v.cpu().numpy()
                    predict[name] = (
                        np.concatenate([predict[name], v]) if name in predict else v
                    )

        predict = self.model.test_return_hook(epoch_info, predict)
        predict_file = os.path.join(
            self.path,
            self.model.name + '_' + str(epoch) + configs.env.paths.predict_file,
        )
        self.logger.save_npy(predict_file, predict)

    def val_test(self, epoch):
        self.test(epoch, data_loader=self.val_loader,  log_text='Val')
        self.test(epoch, data_loader=self.test_loader, log_text='Test')

    # ------------------------------------------------------------------
    # Loss curve plotting
    # ------------------------------------------------------------------

    def plot_loss_curve(self):
        pattern = os.path.join(self.path, self.model.name + '_*' + configs.env.paths.loss_file)
        files   = sorted(
            glob.glob(pattern),
            key=lambda p: int(re.search(r'_(\d+)_loss\.npy$', p).group(1)),
        )
        if not files:
            return

        epochs, loss_data = [], {}
        for fpath in files:
            m = re.search(r'_(\d+)_loss\.npy$', fpath)
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

        if not epochs:
            return

        keys  = [k for k in loss_data if k != 'loss']
        n     = len(keys) + 1
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes_flat  = axes.flatten()

        for i, key in enumerate(['loss'] + keys):
            ax = axes_flat[i]
            ax.plot(epochs, loss_data[key], marker='o', markersize=3)
            ax.set_title(key)
            ax.set_xlabel('Epoch')
            ax.grid(True)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle('Fine-tuning Loss Curves (scan-level)', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.path, 'loss_curve.png')
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        self.logger.info('Loss curve saved to {}'.format(save_path))


# ── Entry point ───────────────────────────────────────────────────────────────

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
