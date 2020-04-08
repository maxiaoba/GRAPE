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

from utils import build_optimizer, objectview, get_known_mask, mask_edge, auto_select_gpu
from uci import get_data

from gnn_model import GNNStack
from prediction_model import MLPNet
from plot_utils import plot_result

from baseline import baseline_inpute
from sklearn.linear_model import LinearRegression
from utils import construct_missing_X, get_impute_mae

def train(data, method):
    train_edge_mask = data.train_edge_mask.numpy()
    X, X_incomplete = construct_missing_X(train_edge_mask, data.df_X)

    X_filled = baseline_inpute(data, method)
    mae = get_impute_mae(X,X_filled, train_edge_mask.reshape(X.shape))

    return mae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uci_data', type=str, default='housing') # 'pks', 'cancer', 'housing', 'wine'
    parser.add_argument('--train_edge', type=float, default=0.7)
    parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--comment', type=str, default='v1')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    # device is cpu by default

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df_np = pd.read_csv('./Data/RNAseq/rnaseq_normalized_0408.csv', header=None, sep=" ")
    df_X = pd.DataFrame(df_np) #extract a subset for experiment
    df_y = pd.DataFrame([0] * df_X.shape[0])

    train_fraction = 0.7
    train_index = np.random.uniform(0,1,df_X.shape[0]) < train_fraction
    df_X_train = df_X[train_index]
    df_X_test = df_X[~train_index]
    df_y_train = df_y[train_index]
    df_y_test = df_y[~train_index]

    data = get_data(df_X_test, df_y_test, args.train_edge, args.train_y, args.seed, normalize=False)

    for method in ['mean','svd','knn']: #'mice'
        log_path = './Data/mdi_results/{}_{}/rnaseq/{}/'.format(method, args.comment, args.seed)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        mae = train(data, method)

        with open("{}results.txt".format(log_path), "w") as text_file:
            text_file.write('{}'.format(mae))

if __name__ == '__main__':
    main()

