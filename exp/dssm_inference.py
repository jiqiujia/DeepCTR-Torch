# -*- coding: utf-8 -*-
import sys
import io
import glob
from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.input_loaders import MultiShardsCSVDatasetV2
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
    parser.add_argument("--key_col", type=str, default="clk")
    parser.add_argument("--header_file", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_cuda", action='store_true')
    parser.add_argument("--embed_dim", type=int, required=True)
    parser.add_argument("--col_dim_file", type=str, required=True)
    parser.add_argument("--query_varlen_feats", type=str, default=None,
                        help="allow multiple varlen feats, separated by comma")
    parser.add_argument("--match_varlen_feats", type=str, default=None,
                        help="allow multiple varlen feats, separated by comma")
    parser.add_argument("--varlen_feat_maxlen", type=int, default=20)
    args = parser.parse_args()

    logger.info("loading data...")
    with io.open(args.header_file, encoding='utf-8') as fin:
        columns = fin.readlines()[0].strip().split(',')

    query_varlen_feat_cols = args.query_varlen_feats.split(",") if args.query_varlen_feats is not None else []
    match_varlen_feat_cols = args.match_varlen_feats.split(",") if args.match_varlen_feats is not None else []

    skip_columns = [args.key_col, 'adid', 'kadsidefeats_creativeidIndex', 'creativeid', 'xad_adinfo_tid',
                    'id', 'kadsidefeats_adinfo_gdtadinfo_advertiseridIndex',
                    'kadsidefeats_adinfo_gdtadinfo_bidpriceIndex'] + \
                   query_varlen_feat_cols + match_varlen_feat_cols
    feat_columns = list(columns)
    for col in skip_columns:
        if col in feat_columns:
            feat_columns.remove(col)

    logger.info('feat_columns: {}'.format(feat_columns))
    target = [args.key_col]

    col_dim_dict = {}
    with io.open(args.col_dim_file, encoding='utf-8') as fin:
        for line in fin.readlines():
            line = line.strip()
            arr = line.split('\t')
            col_dim_dict[arr[0]] = int(arr[1])

    query_fixlen_feature_columns = [SparseFeat(feat, col_dim_dict[feat])
                                 for feat in feat_columns if feat.startswith('kad') or feat.startswith("xad")]
    query_varlen_feature_columns = [VarLenSparseFeat(feat, col_dim_dict[feat], args.varlen_feat_maxlen, 'mean')
                                    for feat in query_varlen_feat_cols]
    query_dnn_feature_columns = query_fixlen_feature_columns + query_varlen_feature_columns

    query_feature_names = get_feature_names(query_dnn_feature_columns)

    # 3.generate input data for model

    shard_paths = glob.glob(args.data_file)
    train_dataset = MultiShardsCSVDatasetV2(shard_paths, columns, query_dnn_feature_columns, args.key_col, 'test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1,
                                  collate_fn=default_collate)

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        logger.info('cuda ready...')
        device = 'cuda:0'

    model = DSSM(query_dnn_feature_columns, [], task='binary',
                 embedding_size=args.embed_dim, dnn_dropout=0.1,
                 l2_reg_embedding=0, dnn_use_bn=True, device=device)

    feats, ids = model.inference(train_dataloader)
    print(feats.shape, ids.shape)