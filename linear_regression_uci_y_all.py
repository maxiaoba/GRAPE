import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from training.linear_regression import linear_regression
from uci.uci_data import load_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='uci')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--train_edge', type=float, default=0.7)
    parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--comment', type=str, default='v1')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    # device is cpu by default

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## new
    for args.data in ['concrete', 'energy', 'housing', 'kin8nm',
                    'naval', 'power', 'protein', 'wine', 'yacht']:
        # for args.method in ['gnn_mdi', 'mean', 'knn', 'svd', 'mice']:
        for args.method in ['mean', 'knn', 'svd', 'mice']:
            data = load_data(args)

            log_path = './uci/y_results/{}_{}/{}/{}/'.format(args.method, args.comment, args.data, args.seed)
            if not os.path.isdir(log_path):
                os.makedirs(log_path)
            load_path = './uci/mdi_results/gnn_mdi_{}/{}/{}/'.format(args.comment, args.data, args.seed)
            linear_regression(data, args, log_path, load_path)

if __name__ == '__main__':
    main()

