# -*- coding: utf-8 -*-
import argparse
import io
import os
import glob
import logging
from deepctr_torch import dict_helper as utils
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, default="")
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--min_freq", type=int, default=0)
    args = parser.parse_args()

    shard_paths = glob.glob(args.src_file)
    print(shard_paths)

    srcDict = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=True)
    tgtDict = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=True)
    for path in shard_paths:
        with io.open(path, encoding='utf-8') as fin:
            for line in fin:
                arr = line.split('_!_')
                for qw in arr[0].split(' '):
                    srcDict.add(qw)
                for dw in arr[1].split(' '):
                    tgtDict.add(dw)

    srcDict = srcDict.prune(srcDict.size(), args.min_freq)
    srcDict.writeFile(os.path.join(args.out_path, 'src.vocab'))

    tgtDict = srcDict.prune(tgtDict.size(), args.min_freq)
    tgtDict.writeFile(os.path.join(args.out_path, 'tgt.vocab'))

    for path in shard_paths:
        with io.open(path, encoding='utf-8') as fin, \
            io.open(os.path.join(args.out_path, os.path.split(path)[-1]), 'w+', encoding='utf-8') as fout:
                for line in fin:
                    arr = line.split('_!_')
                    src_ids = srcDict.convertToIdx(arr[0].split(' '))
                    tgt_ids = tgtDict.convertToIdx(arr[1].split(' '))
                    fout.write(' '.join([str(id) for id in src_ids]) +','+ ' '.join([str(id) for id in tgt_ids]) + '\n')

    with io.open(os.path.join(args.out_path, 'col_dim.txt'), 'w+', encoding='utf-8') as fout:
        fout.write('query\t' + str(srcDict.size()) + '\n')
        fout.write('doc\t' + str(tgtDict.size()) + '\n')