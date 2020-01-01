# -*- coding: utf-8 -*-
import io
from torch.utils.data import Dataset
import numpy as np

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiShardsCSVDataset(Dataset):
    def __init__(self, shard_paths, feat_idxes, target_idx):
        self.shard_paths = shard_paths
        self.feat_idxes = feat_idxes
        self.target_idx = target_idx
        lens = [0] * len(self.shard_paths)
        for i, path in enumerate(shard_paths):
            for _ in io.open(path):
                lens[i] += 1
        self.lens = lens
        self.length = sum(self.lens)
        self.cur_shard_idx = -1
        self.cur_dataset = None

    def __len__(self):
        return self.length

    def _get_cur_dataset(self, shard_idx):
        if shard_idx != self.cur_shard_idx:
            res = []
            logger.info('loading shard {}'.format(self.shard_paths[shard_idx]))
            with io.open(self.shard_paths[shard_idx], encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    arr = line.split(',')
                    arr = [int(float(v)) for v in arr]
                    feat = np.asarray([arr[i] for i in self.feat_idxes])
                    res.append((feat, arr[self.target_idx]))
            self.cur_shard_idx = shard_idx
            self.cur_dataset = res
        return self.cur_dataset

    def __getitem__(self, index):
        shard_idx = 0
        for i, len in enumerate(self.lens):
            if index - len > 0:
                index = index - len
            else:
                shard_idx = i
                break

        cur_dataset = self._get_cur_dataset(shard_idx)
        return cur_dataset[index]

class IterableDataset(Dataset):
    def __iter__(self):
        raise NotImplementedError


class MultiShardsCSVIterableDataset(IterableDataset):
    def __init__(self, shard_paths, target_idx):
        self.shard_paths = shard_paths
        self.target_idx = target_idx
        len = 0
        for path in shard_paths:
            for _ in io.open(path):
                len += 1
        self.length = len

    def __len__(self):
        return self.length

    def _iter_samples(self, path):
        with io.open(path, encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                arr = line.split(',')
                arr = [int(float(v)) for v in arr]
                yield arr, arr[self.target_idx]

    def _iter_shards(self):
        for path in self.shard_paths:
            yield self._iter_samples(path)

    def __iter__(self):
        return self._iter_shards()

