import time
import argparse
import os

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import joblib
import pickle

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

from utils import build_optimizer, objectview, get_known_mask, mask_edge
from uci import get_data

from gnn_model import GNNStack
from prediction_model import MLPNet
from plot_utils import plot_result

from baseline import baseline_inpute
from sklearn.linear_model import LinearRegression
from utils import construct_missing_X


def train(data, args):
    x = data.x.clone().detach()
    train_edge_mask = data.train_edge_mask.numpy()
    train_edge_index = data.train_edge_index.clone().detach()
    train_edge_attr = data.train_edge_attr.clone().detach()
    test_edge_index = data.test_edge_index.clone().detach()
    test_edge_attr = data.test_edge_attr.clone().detach()

    y = data.y.detach().numpy()
    train_y_mask = data.train_y_mask.clone().detach()
    # print(torch.sum(train_y_mask))
    test_y_mask = data.test_y_mask.clone().detach()
    y_train = y[train_y_mask]
    y_test = y[test_y_mask]

    if args.method == 'gnn':
        load_path = './Data/uci/' + args.uci_data + '/' + args.load_dir + '/'
        result = joblib.load(load_path + 'result.pkl')
        result = objectview(result)
        model_args = result.args
        from gnn_model import GNNStack
        model = GNNStack(data.num_node_features, model_args.node_dim,
                         model_args.edge_dim, model_args.edge_mode,
                         model_args.model_types, model_args.dropout)
        model.load_state_dict(torch.load(load_path + 'model.pt'))
        model.eval()
        impute_model = MLPNet([model_args.node_dim, model_args.node_dim], 1,
                              hidden_layer_sizes=model_args.impute_hiddens,
                              dropout=model_args.dropout)
        impute_model.load_state_dict(torch.load(load_path + 'impute_model.pt'))
        impute_model.eval()

        x_embd = model(x, train_edge_attr, train_edge_index)
        # X = x_embd.detach().numpy()[:y.shape[0],:]
        x_pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
        x_pred = x_pred[:int(test_edge_attr.shape[0] / 2)]
        X_true, X_incomplete = construct_missing_X(train_edge_mask, data.df_X)
        X = X_incomplete
        for i in range(int(test_edge_attr.shape[0] / 2)):
            assert X_true[test_edge_index[0, i], test_edge_index[1, i] - y.shape[0]] == test_edge_attr[i]
            X[test_edge_index[0, i], test_edge_index[1, i] - y.shape[0]] = x_pred[i]
    else:
        X = baseline_inpute(data, args.method)

    reg = LinearRegression().fit(X[train_y_mask, :], y_train)
    # print(reg.score(X[test_y_mask,:], y_test))
    y_pred_test = reg.predict(X[test_y_mask, :])
    print(np.mean((y_pred_test - y_test) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uci_data', type=str, default='housing')  # 'pks', 'cancer', 'housing', 'wine'
    parser.add_argument('--method', type=str, default='mean')
    parser.add_argument('--train_edge', type=float, default=0.7)
    parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--load_dir', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    df_X = pd.read_csv('./Data/uci/' + args.uci_data + "/" + args.uci_data + '.csv')
    df_y = pd.read_csv('./Data/uci/' + args.uci_data + "/" + args.uci_data + '_target.csv', header=None)
    data = get_data(df_X, df_y, args.train_edge, args.train_y, args.seed)

    train(data, args)


if __name__ == '__main__':
    main()