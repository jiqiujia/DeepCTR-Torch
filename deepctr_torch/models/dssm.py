# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com
Reference:
    [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.(https://arxiv.org/pdf/1606.07792.pdf)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN


class DSSM(BaseModel):
    """Instantiates the Wide&Deep Learning architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param embedding_size: positive integer,sparse feature embedding_size
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.

    """

    def __init__(self, query_dnn_feature_columns, match_dnn_feature_columns, embedding_size=8,
                 dnn_hidden_units=(300, 300, 128),
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation=F.relu,
                 dnn_use_bn=False, task='binary', device='cpu'):
        super(DSSM, self).__init__([], query_dnn_feature_columns + match_dnn_feature_columns,
                                   embedding_size=embedding_size,
                                   dnn_hidden_units=dnn_hidden_units,
                                   l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, init_std=init_std,
                                   seed=seed,
                                   dnn_dropout=dnn_dropout, dnn_activation=dnn_activation,
                                   task=task, device=device)

        self.query_dnn_feature_columns = query_dnn_feature_columns
        self.match_dnn_feature_columns = match_dnn_feature_columns
        self.query_dnn = DNN(self.compute_input_dim(self.query_dnn_feature_columns, embedding_size), dnn_hidden_units,
                             activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                             init_std=init_std, device=device)
        self.match_dnn = DNN(self.compute_input_dim(self.match_dnn_feature_columns, embedding_size), dnn_hidden_units,
                             activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                             init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.add_regularization_loss(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.query_dnn.named_parameters()), l2_reg_dnn)
        self.add_regularization_loss(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.match_dnn.named_parameters()), l2_reg_dnn)
        self.add_regularization_loss(self.dnn_linear.weight, l2_reg_dnn)

        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.query_dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        query_dnn_output = self.query_dnn(dnn_input)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.match_dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        match_dnn_output = self.match_dnn(dnn_input)
        cosine_logit = F.cosine_similarity(query_dnn_output, match_dnn_output)

        y_pred = self.out(cosine_logit)

        return y_pred