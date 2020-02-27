# -*- coding: utf-8 -*-
import os
import sys
import io
import glob
import pandas as pd

if __name__ == '__main__':
    root = sys.argv[1]
    header_file = sys.argv[2]
    var_len_cols = sys.argv[3].split(",")
    out_path = sys.argv[4]

    with io.open(header_file, encoding='utf-8') as fin:
        header = fin.readlines()[0].strip().split(',')
    data_paths = glob.glob(root)
    print(data_paths)

    unique_vals = [0 for _ in range(len(header))]
    dtypes = {}
    for col in var_len_cols:
        dtypes[col] = str
    for path in data_paths:
        print(path)
        df = pd.read_csv(path, header=None, names=header, dtype=dtypes)
        for col in var_len_cols:
            df[col] = df[col].apply(lambda x: max([int(float(v)) for v in x.split(' ')]))
        maxs = df.max()
        for i, val in enumerate(maxs):
            unique_vals[i] = max(unique_vals[i], int(float(val)))
        # with io.open(path, encoding='utf-8') as fin:
        #     for line in fin:
        #         arr = line.strip().split(',')
        #         for i, val in enumerate(arr):
        #             unique_vals[i] = max(unique_vals[i], int(float(val)))


    header_dim_dict = {}
    for col, val in zip(header, unique_vals):
        header_dim_dict[col] = val + 2

    with io.open(out_path, 'w+', encoding='utf-8') as fout:
        for key, val in header_dim_dict.items():
            fout.write(key + '\t' + str(val) + '\n')
