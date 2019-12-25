# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import torch

if __name__ == "__main__":
    data = pd.read_csv('./pctr_10038_sample.txt')
    skip_columns = set()
    null_columns = data.isnull().sum(axis=0) / len(data)
    null_columns = null_columns[null_columns > 0.5]
    for col, _ in null_columns.items():
        skip_columns.add(col)

    feat_columns = list(data.columns)
    for col in skip_columns:
        feat_columns.remove(col)
    feat_columns = list(filter(lambda col: col.startswith('k'), feat_columns))#'interest' in col

    data[feat_columns] = data[feat_columns].fillna(-1)
    target = ['clk']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in feat_columns:
        print(feat)
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2.count #unique features for each sparse field,and record dense feature field name

    query_dnn_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                 for feat in feat_columns if feat.startswith('kad')]
    match_dnn_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                 for feat in feat_columns if not feat.startswith('kad')]

    query_feature_names = get_feature_names(query_dnn_feature_columns)
    match_feature_names = get_feature_names(match_dnn_feature_columns)
    feature_names = query_feature_names + match_feature_names

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DSSM(query_dnn_feature_columns, match_dnn_feature_columns, task='binary',
                 l2_reg_embedding=1e-5, dnn_use_bn=True, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    model.fit(train_model_input, train[target].values,
              batch_size=32, epochs=10, validation_split=0.1, verbose=2)

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
