import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from training.baseline import baseline_mdi
from uci.uci_data import load_data
from utils.utils import construct_missing_X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='uci')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--train_edge', type=float, default=0.7)
    parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--node_mode', type=int, default=0)  # 0: feature node onehot, sample node all 1; 1: all onehot

    parser.add_argument('--method', type=str, default='mean')
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--comment', type=str, default='v1')
    args = parser.parse_args()
    # device is cpu by default

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## new
    for args.data in ['concrete', 'energy', 'housing', 'kin8nm',
                     'naval', 'power', 'protein', 'wine', 'yacht']:
        data = load_data(args)
        log_path = './uci/data_with_missing/{}/{}/'.format(args.data, args.seed)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        train_edge_mask = data.train_edge_mask.numpy()
        X, X_incomplete = construct_missing_X(train_edge_mask, data.df_X)

        y = data.y.detach().numpy()
        train_y_mask = data.train_y_mask.numpy()
        test_y_mask = data.test_y_mask.numpy()

        train_X = X[train_y_mask, :]
        test_X = X[test_y_mask, :]
        
        train_X_missing = pd.DataFrame(X_incomplete[train_y_mask, :])
        test_X_missing = pd.DataFrame(X_incomplete[test_y_mask, :])

        train_y = pd.DataFrame(y[train_y_mask])
        test_y = pd.DataFrame(y[test_y_mask])
        
        train_X_missing.to_csv('./uci/data_with_missing/{}/{}/train_X_missing.txt'.format(args.data, args.seed),\
            index=False)
        test_X_missing.to_csv('./uci/data_with_missing/{}/{}/test_X_missing.txt'.format(args.data, args.seed),\
            index=False)

        train_y.to_csv('./uci/data_with_missing/{}/{}/train_y.txt'.format(args.data, args.seed),\
            index=False)
        test_y.to_csv('./uci/data_with_missing/{}/{}/test_y.txt'.format(args.data, args.seed),\
            index=False)

        print(args.data)
        print(args.seed)


if __name__ == '__main__':
    main()

