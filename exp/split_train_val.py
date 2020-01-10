# -*- coding: utf-8 -*-

import io
import sys
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

def write_out(lines, out_path, phase, shard_num):
    segs = np.linspace(0, len(lines), shard_num+1, dtype=np.int32)
    for i in range(shard_num):
        with io.open(os.path.join(out_path, '%s_%d' % (phase, i)), 'w+', encoding='utf-8') as fout:
            for line in lines[segs[i]:segs[i+1]]:
                fout.write(line)

if __name__ == '__main__':
    shard_paths = glob.glob(sys.argv[1])
    print(shard_paths)
    lines = []
    for path in shard_paths:
        with io.open(path, encoding='utf-8') as fin:
            for line in fin:
                lines.append(line)

    val_prop = float(sys.argv[2])
    test_prop = float(sys.argv[3])
    train_shard_num = int(sys.argv[4])
    test_shard_num = int(sys.argv[5])
    out_path = sys.argv[6]


    train, val = train_test_split(lines, test_size=val_prop)
    if test_prop > 0:
        train, test = train_test_split(train, test_size=test_prop)

    write_out(train, out_path, "train", train_shard_num)
    write_out(val, out_path, "val", test_shard_num)
    if test_prop > 0:
        write_out(test, out_path, "train", test_shard_num)
