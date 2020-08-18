# -*- coding: utf-8 -*-

import io
import sys
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

query_path = sys.argv[1]
doc_path = sys.argv[2]
pair_path = sys.argv[3]

val_prop = float(sys.argv[4])
test_prop = float(sys.argv[5])
out_path = sys.argv[6]

print('loading all queries...')
all_queries = {}
with io.open(query_path, encoding='utf-8') as fin:
    for line in fin.readlines():
        arr = line.split('\t')
        all_queries[int(arr[1])] = line
print('loading all docs...')
all_docs = {}
with io.open(doc_path, encoding='utf-8') as fin:
    for line in fin.readlines():
        arr = line.split('\t')
        all_docs[int(arr[1])] = line


def write_out(lines, out_path, phase):
    with io.open(os.path.join(out_path, '%s_pair.txt' % phase), 'w+', encoding='utf-8') as fout:
        docs = set()
        queries = set()
        for line in lines:
            arr = line.split('\t')
            docs.add(int(arr[0]))
            queries.add(int(arr[1]))
            fout.write(line)
    with io.open(os.path.join(out_path, '%s_%s' % (phase, os.path.basename(query_path))), 'w+', encoding='utf-8') as fout:
        for query in queries:
            fout.write(all_queries[query])
    with io.open(os.path.join(out_path, '%s_%s' % (phase, os.path.basename(doc_path))), 'w+', encoding='utf-8') as fout:
        for doc in docs:
            fout.write(all_docs[doc])


if __name__ == '__main__':
    pairs = []
    with io.open(pair_path, encoding='utf-8') as fin:
        pairs = fin.readlines()

    print('split train val')
    train, val = train_test_split(pairs, test_size=val_prop)
    if test_prop > 0:
        print('split train test')
        train, test = train_test_split(train, test_size=test_prop)
        write_out(test, out_path, "test")

    print('writing out...')
    write_out(train, out_path, "train")
    write_out(val, out_path, "val")
