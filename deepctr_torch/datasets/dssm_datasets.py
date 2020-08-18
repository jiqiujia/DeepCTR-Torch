# -*- coding: utf-8 -*-
import io
from torch.utils.data import Dataset
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
import numpy as np
import random
import csv

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class TextMatchingDataset(Dataset):
    def __init__(self, query_path, doc_path, pair_path,
                  query_maxlen, doc_maxlen, phase='train'):
        self.id2query = {}
        with io.open(query_path, encoding='utf-8') as fin:
            for line in fin:
                arr = line.split('\t')
                self.id2query[int(arr[1])] = [int(v) for v in arr[-1].split(' ')]
        self.id2doc = {}
        with io.open(doc_path, encoding='utf-8') as fin:
            for line in fin:
                arr = line.split('\t')
                self.id2doc[int(arr[1])] = [int(v) for v in arr[-1].split(' ')]
        # doc, query
        self.pairs = []
        with io.open(pair_path, encoding='utf-8') as fin:
            for line in fin:
                arr = line.split('\t')
                self.pairs.append((int(arr[0]), int(arr[1])))

        self.phase = phase
        self.length = len(self.pairs)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rint = random.randint(0, 1)
        pair = self.pairs[index]
        qfeat = self.id2query[pair[1]]
        if rint == 1:
            dfeat = self.id2doc[pair[0]]
        else:
            negpair = random.choice(self.pairs)
            dfeat = self.id2doc[negpair[0]]
        qfeat += [0] * (self.query_maxlen - len(qfeat))
        dfeat += [0] * (self.doc_maxlen - len(dfeat))
        feat = qfeat + dfeat
        return np.asarray(feat), rint*1.0
