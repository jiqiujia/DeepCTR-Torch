# -*- coding: utf-8 -*-

import io
import sys
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    shard_paths = glob.glob(sys.argv[1])
    lines = []
    for path in shard_paths:
        with io.open(path, encoding='utf-8') as fin:
            for line in fin:
                lines.append(line)

    val_prop = float(sys.argv[2])
    test_prop = float(sys.argv[3])
    out_path = sys.argv[4]

    shard_num = len(shard_paths)

    train, val = train_test_split(lines, test_size=val_prop)
    if test_prop > 0:
        train, test = train_test_split(train, test_size=test_prop)

    segs = np.linspace(0, len(train), shard_num + 1, dtype=np.int32)
    for i in range(shard_num):
        with io.open(os.path.join(out_path, 'train_%d' % i), 'w+', encoding='utf-8') as fout:
            for line in train[segs[i]:segs[i+1]]:
                fout.write(line)
    with io.open(os.path.join(out_path, 'val'), 'w+', encoding='utf-8') as fout:
        for line in val:
            fout.write(line)
    if test_prop > 0:
        with io.open(os.path.join(out_path + 'test'), 'w+', encoding='utf-8') as fout:
            for line in test:
                fout.write(line)