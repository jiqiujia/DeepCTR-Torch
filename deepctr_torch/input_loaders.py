# -*- coding: utf-8 -*-
import io
from torch.utils.data import Dataset
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
import numpy as np
import pandas as pd
import dask.dataframe
import csv

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
            if index - len >= 0:
                index = index - len
            else:
                shard_idx = i
                break

        cur_dataset = self._get_cur_dataset(shard_idx)
        return cur_dataset[index]

# support varlen feature; don't support multiprocessing
class MultiShardsCSVDatasetV2(Dataset):
    def __init__(self, shard_paths, header, feat_cols, target_col, phase='train', oov=-1):
        self.shard_paths = shard_paths
        self.header = header
        self.feat_cols = feat_cols
        self.target_col = target_col
        self.phase = phase
        self.oov = oov
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
            logger.info('loading shard {}'.format(self.shard_paths[shard_idx]))
            # df = dask.dataframe.read_csv(self.shard_paths[shard_idx], header=None, names=self.header,
            #                  encoding='utf-8')
            df = csv.DictReader(io.open(self.shard_paths[shard_idx], encoding='utf-8'), fieldnames=self.header)
            dataset = []
            # for _, row in df.iterrows():
            for row in df:
                feat = []
                skip = False
                for col in self.feat_cols:
                    if isinstance(col, SparseFeat):
                        tmp_val = int(float(row[col.name]))
                        if tmp_val >= col.dimension:
                            if self.oov > 0:
                                tmp_val = col.dimension - 1
                            elif self.oov < 0:
                                skip = True
                                tmp_val = 0
                            else:
                                tmp_val = 0
                        feat += [tmp_val]
                    elif isinstance(col, DenseFeat):
                        feat += [float(row[col.name])]
                    elif isinstance(col, VarLenSparseFeat):
                        idxes = [int(v) for v in row[col.name].split(" ")[:col.maxlen]]
                        idxes += [0] * (col.maxlen - len(idxes))
                        idxes = [val if val < col.dimension else col.dimension - 1 for val in idxes]
                        feat += idxes
                    else:
                        assert False

                if self.target_col is not None:
                    if skip:
                        dataset.append((np.asarray(feat), self.oov))
                    else:
                        dataset.append((np.asarray(feat), 1.0 if float(row[self.target_col]) > 0 else 0.0))
                else:
                    dataset.append(np.asarray(feat))
            self.cur_shard_idx = shard_idx
            self.cur_dataset = dataset
        return self.cur_dataset

    def __getitem__(self, index):
        shard_idx = 0
        for i, len in enumerate(self.lens):
            if index - len >= 0:
                index = index - len
            else:
                shard_idx = i
                break

        cur_dataset = self._get_cur_dataset(shard_idx)
        return cur_dataset[index]
