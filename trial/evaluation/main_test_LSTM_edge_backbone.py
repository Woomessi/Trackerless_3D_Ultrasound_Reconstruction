import argparse
import os

import numpy as np
import torch

import configs
import datasets
import models
import utils

# ── Checkpoint settings ───────────────────────────────────────────────────────
CKPT_DIR   = "/media/wu/Extreme SSD/online_LSTM_edge_bk-hp_bk-TUS_LSTM_edge_complete"
CKPT_EPOCH = 21   # set to None to use the latest checkpoint
# ─────────────────────────────────────────────────────────────────────────────


class Evaluator(object):

    def __init__(self, ckpt_dir=CKPT_DIR, epoch=CKPT_EPOCH):
        self.model_cfg = configs.BaseConfig(
            '/home/wu/Documents/projects/my_projects/Trackerless_3D_Ultrasound_Reconstruction'
            '/res/models/online_LSTM_edge_bk.json'
        )
        self.run_cfg = configs.Run(
            '/home/wu/Documents/projects/my_projects/Trackerless_3D_Ultrasound_Reconstruction'
            '/res/run/hp_bk.json',
            gpus='0',
        )
        self.dataset_cfg = datasets.functional.common.more(
            configs.BaseConfig(
                '/home/wu/Documents/projects/my_projects/Trackerless_3D_Ultrasound_Reconstruction'
                '/res/datasets/TUS_LSTM_edge_complete.json'
            )
        )
        self._ckpt_dir = ckpt_dir
        self._requested_epoch = epoch
        self._init()
        self._get_component()
        self.show_cfgs()

    def _init(self):
        utils.common.set_seed(0)
        self.msg = {}

    def _get_component(self):
        self.path = utils.common.get_path(self.model_cfg, self.dataset_cfg, self.run_cfg)
        self.logger = utils.Logger(self.path, utils.common.get_filename(self.model_cfg._path))

        self.dataset = datasets.functional.common.find(self.dataset_cfg.name)(
            self.dataset_cfg, logger=self.logger
        )
        self.model = models.functional.common.find(self.model_cfg.name)(
            self.model_cfg, self.dataset.cfg, self.run_cfg,
            dataset=self.dataset, logger=self.logger, main_msg=self.msg,
        )
        self.epoch = self.model.load(self._requested_epoch, path=self._ckpt_dir)
        self.logger.info('Loaded checkpoint from {} at epoch {}'.format(self._ckpt_dir, self.epoch))

    def show_cfgs(self):
        self.logger.info(self.model.cfg)
        self.logger.info(self.run_cfg)
        self.logger.info(self.dataset.cfg)

    def split(self):
        _, self.valset, self.testset = self.dataset.split()

        self.run_cfg.test_batch_size = getattr(self.run_cfg, 'test_batch_size', self.run_cfg.batch_size)

        self.val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.run_cfg.test_batch_size,
            shuffle=False,
            collate_fn=getattr(self.valset.dataset, 'collate_fn', None),
            num_workers=self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.run_cfg.test_batch_size,
            shuffle=False,
            collate_fn=getattr(self.testset.dataset, 'collate_fn', None),
            num_workers=self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
        )

    def _run_split(self, data_loader, split_name):
        """Run model.test() over data_loader and return per-sequence metric arrays."""
        torch.cuda.empty_cache()
        all_metrics = {}
        count = 0
        batch_per_epoch = len(data_loader)
        count_data = len(data_loader.dataset)
        log_step = max(int(np.power(10, np.floor(np.log10(max(batch_per_epoch / 10, 1))))), 1)

        epoch_info = {
            'epoch': self.epoch,
            'batch_per_epoch': batch_per_epoch,
            'count_data': count_data,
            'log_text': split_name,
        }

        with torch.no_grad():
            for batch_idx, (sample_dict, index) in enumerate(data_loader):
                _count = len(list(sample_dict.values())[0])
                epoch_info['batch_idx'] = batch_idx
                epoch_info['index'] = index
                epoch_info['batch_count'] = _count
                metric_dict = self.model.test(epoch_info, sample_dict)
                count += _count
                if batch_idx % log_step == 0:
                    self.logger.info('{} [{}/{} ({:.0f}%)]'.format(
                        split_name, count, count_data, 100.0 * count / count_data,
                    ))
                for name, value in metric_dict.items():
                    v = value.float() if value.shape else value.unsqueeze(0)
                    v = v.cpu().numpy()
                    all_metrics[name] = (
                        np.concatenate([all_metrics[name], v])
                        if name in all_metrics else v
                    )

        return all_metrics

    def _report(self, metrics, split_name):
        """Print mean ± std for each metric and save results."""
        header = '\n{} Results (epoch {})'.format(split_name, self.epoch)
        self.logger.info(header)
        print(header)
        summary = {}
        for key in sorted(metrics.keys()):
            vals = np.array(metrics[key], dtype=np.float64)
            mean, std = float(np.mean(vals)), float(np.std(vals))
            summary[key] = {'mean': mean, 'std': std, 'values': vals}
            line = '  {:6s}: {:.6f} ± {:.6f}'.format(key, mean, std)
            self.logger.info(line)
            print(line)

        save_path = os.path.join(
            self.path,
            '{}_metrics_epoch{}.npy'.format(split_name.lower(), self.epoch),
        )
        np.save(save_path, summary)
        self.logger.info('Saved metrics to {}'.format(save_path))
        print('Saved metrics to {}'.format(save_path))
        return summary

    def evaluate(self, splits=('val', 'test')):
        results = {}
        if 'val' in splits:
            self.logger.info('Evaluating on validation set ...')
            val_metrics = self._run_split(self.val_loader, 'Val')
            results['val'] = self._report(val_metrics, 'Val')
        if 'test' in splits:
            self.logger.info('Evaluating on test set ...')
            test_metrics = self._run_split(self.test_loader, 'Test')
            results['test'] = self._report(test_metrics, 'Test')
        return results


def run():
    parser = argparse.ArgumentParser(description='Evaluate Online_LSTM_Edge_Backbone')
    parser.add_argument('--ckpt-dir', type=str, default=CKPT_DIR,
                        help='Directory containing model checkpoints (default: CKPT_DIR)')
    parser.add_argument('--epoch', type=int, default=CKPT_EPOCH,
                        help='Checkpoint epoch to load (default: CKPT_EPOCH, None = latest)')
    parser.add_argument('--split', choices=['val', 'test', 'both'], default='both',
                        help='Which split to evaluate (default: both)')
    args = parser.parse_args()

    splits = ('val', 'test') if args.split == 'both' else (args.split,)

    evaluator = Evaluator(ckpt_dir=args.ckpt_dir, epoch=args.epoch)
    evaluator.split()
    evaluator.evaluate(splits=splits)


if __name__ == '__main__':
    run()
