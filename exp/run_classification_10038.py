# -*- coding: utf-8 -*-
import sys
import io
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import torch
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--label_col", type=str, default="clk")
    parser.add_argument("--header_file", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch_num", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, required=True)
    parser.add_argument("--use_cuda", action='store_true')
    args = parser.parse_args()

    with io.open(args.header_file, encoding='utf-8') as fin:
        header = fin.readlines()[0].strip().split(',')
    data = pd.read_csv(args.data_file, header=None, names=header)

    skip_columns = [args.label_col, 'adid']
    feat_columns = list(data.columns)
    for col in skip_columns:
        feat_columns.remove(col)

    print('feat_columns: ', feat_columns)
    data[feat_columns] = data[feat_columns].fillna(-1)
    target = [args.label_col]

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # for feat in feat_columns:
    #     print(feat)
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])

    # 2.count #unique features for each sparse field,and record dense feature field name

    query_dnn_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                 for feat in feat_columns if feat.startswith('kad')]
    match_dnn_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                 for feat in feat_columns if not feat.startswith('kad')]

    query_feature_names = get_feature_names(query_dnn_feature_columns)
    match_feature_names = get_feature_names(match_dnn_feature_columns)
    feature_names = query_feature_names + match_feature_names

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DSSM(query_dnn_feature_columns, match_dnn_feature_columns, task='binary',
                 embedding_size=args.embed_dim,
                 l2_reg_embedding=1e-5, dnn_use_bn=True, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], lr=0.01)
    model.fit(train_model_input, train[target].values,
              batch_size=args.batch_size, epochs=args.epoch_num,
              validation_split=0.01, verbose=1)

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
