# -*- coding: utf-8 -*-
import argparse
import io
import os
import glob
import logging
import concurrent.futures
from functools import partial
from deepctr_torch import dict_helper as utils
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_files(path):
    srcSet = set()
    tgtSet = set()
    srcDict = utils.Dict(lower=True)
    tgtDict = utils.Dict(lower=True)
    logging.info(path)
    with io.open(path, encoding='utf-8') as fin:
        for line in fin:
            arr = line.strip().split('_!_')
            srcSet.add(arr[0])
            tgtSet.add(arr[1])
    for src in srcSet:
        for qw in src.split(' '):
            srcDict.add(qw)
    for tgt in tgtSet:
        for dw in tgt.split(' '):
            tgtDict.add(dw)
    return srcDict, tgtDict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, default="")
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--min_freq", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    shard_paths = glob.glob(args.src_file)
    logging.info(shard_paths)

    srcDict = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=True)
    tgtDict = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for isrcDict, itgtDict in executor.map(load_files, shard_paths):
            for key, idx in isrcDict.labelToIdx.items():
                srcDict.add(key, freq=isrcDict.frequencies[idx])
            for key, idx in itgtDict.labelToIdx.items():
                tgtDict.add(key, freq=itgtDict.frequencies[idx])
    print(srcDict.size(), tgtDict.size())
    srcDict = srcDict.prune(srcDict.size(), args.min_freq)
    srcDict.writeFile(os.path.join(args.out_path, 'src.vocab'))

    tgtDict = tgtDict.prune(tgtDict.size(), args.min_freq)
    tgtDict.writeFile(os.path.join(args.out_path, 'tgt.vocab'))
    print(srcDict.size(), tgtDict.size())

    with io.open(os.path.join(args.out_path, 'col_dim.txt'), 'w+', encoding='utf-8') as fout:
        fout.write('query\t' + str(srcDict.size()) + '\n')
        fout.write('doc\t' + str(tgtDict.size()) + '\n')
