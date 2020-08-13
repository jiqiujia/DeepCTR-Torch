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


def write_files(path, out_path=None, srcDict=None, tgtDict=None):
    out_path = os.path.join(out_path, os.path.split(path)[-1])
    logging.info(path + "\t" + out_path)
    with io.open(path, encoding='utf-8') as fin, \
            io.open(out_path, 'w+', encoding='utf-8') as fout:
        for line in fin:
            arr = line.split('_!_')
            src_ids = srcDict.convertToIdx(arr[0].split(' '))
            tgt_ids = tgtDict.convertToIdx(arr[1].split(' '))
            fout.write(' '.join([str(id) for id in src_ids]) + ',' + ' '.join([str(id) for id in tgt_ids]) + '\n')
    return out_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, default="")
    parser.add_argument("--src_vocab", type=str)
    parser.add_argument("--tgt_vocab", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    shard_paths = glob.glob(args.src_file)
    logging.info(shard_paths)

    srcDict = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=True)
    tgtDict = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=True)
    srcDict.loadFile(args.src_vocab)
    tgtDict.loadFile(args.tgt_vocab)

    partial_fun = partial(write_files, out_path=args.out_path, srcDict=srcDict, tgtDict=tgtDict)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for i in executor.map(partial_fun, shard_paths):
            print(i)
