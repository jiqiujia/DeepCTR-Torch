# -*- coding: utf-8 -*-
import sys
import io
import glob
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.input_loaders import MultiShardsCSVDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import argparse
import collections
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--label_col", type=str, default="clk")
    parser.add_argument("--header_file", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch_num", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, required=True)
    parser.add_argument("--use_cuda", action='store_true')
    parser.add_argument("--col_dim_file", type=str, required=True)
    args = parser.parse_args()

    logger.info("loading data...")
    with io.open(args.header_file, encoding='utf-8') as fin:
        columns = fin.readlines()[0].strip().split(',')

    skip_columns = [args.label_col, 'adid']
    feat_columns = list(columns)
    for col in skip_columns:
        feat_columns.remove(col)

    logger.info('feat_columns: {}'.format(feat_columns))
    target = [args.label_col]

    col_dim_dict = {}
    with io.open(args.col_dim_file, encoding='utf-8') as fin:
        for line in fin.readlines():
            line = line.strip()
            arr = line.split('\t')
            col_dim_dict[arr[0]] = int(arr[1])

    query_dnn_feature_columns = [SparseFeat(feat, col_dim_dict[feat])
                                 for feat in feat_columns if feat.startswith('kad')]
    match_dnn_feature_columns = [SparseFeat(feat, col_dim_dict[feat])
                                 for feat in feat_columns if not feat.startswith('kad')]

    query_feature_names = get_feature_names(query_dnn_feature_columns)
    match_feature_names = get_feature_names(match_dnn_feature_columns)
    feature_names = query_feature_names + match_feature_names

    feat_idx = [columns.index(name) for name in feature_names]
    target_idx = columns.index(args.label_col)
    # 3.generate input data for model

    shard_paths = glob.glob(args.data_file)
    train_dataset = MultiShardsCSVDataset(shard_paths, feat_idx, target_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1,
                                  collate_fn=default_collate)

    val_dataset = MultiShardsCSVDataset([args.val_file], feat_idx, target_idx)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1,
                            collate_fn=default_collate)

    test_dataset = MultiShardsCSVDataset([args.test_file], feat_idx, target_idx)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1,
                             collate_fn=default_collate)
    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        logger.info('cuda ready...')
        device = 'cuda:0'

    model = DSSM(query_dnn_feature_columns, match_dnn_feature_columns, task='binary',
                 embedding_size=args.embed_dim, dnn_dropout=0.1,
                 l2_reg_embedding=0, dnn_use_bn=True, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], lr=0.01)
    model.fit_loader(train_dataloader, val_loader, sample_num=train_dataset.__len__(),
                     batch_size=args.batch_size, epochs=args.epoch_num, verbose=1)

    eval_result = model.evaluate_loader(test_loader, 256)
    eval_str = ""
    for name, result in eval_result.items():
        eval_str += " test_" + name + \
                    ": {0: .4f}".format(result)
    logger.info(eval_str)
