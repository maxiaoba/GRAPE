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
    mae = get_impute_mae(X,X_filled,train_edge_mask.reshape(X.shape))

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

    ## new
    for dataset in ['concrete', 'energy', 'housing', 'kin8nm',
                    'naval', 'power', 'protein', 'wine', 'yacht']:
        df_np = np.loadtxt('./Data/uci_all/{}/data/data.txt'.format(dataset))
        df_y = pd.DataFrame(df_np[:, -1:])
        df_X = pd.DataFrame(df_np[:, :-1])
        for seed in [0,1,2,3,4]:
            data = get_data(df_X, df_y, args.train_edge, args.train_y, seed)
            for method in ['mean', 'knn', 'svd', 'mice']:
                log_path = './Data/mdi_results/{}_{}/{}/{}/'.format(method, args.comment, dataset, seed)
                if not os.path.isdir(log_path):
                    os.makedirs(log_path)
                mae = train(data, method)

                with open("{}results.txt".format(log_path), "w") as text_file:
                    text_file.write('{}'.format(mae))

if __name__ == '__main__':
    main()

