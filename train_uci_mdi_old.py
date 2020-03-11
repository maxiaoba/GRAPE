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


def train(data, args, log_path):
    # build model
    model = GNNStack(data.num_node_features, args.node_dim,
                     args.edge_dim, args.edge_mode,
                     args.model_types, args.dropout)
    impute_model = MLPNet([args.node_dim, args.node_dim], 1,
                          hidden_layer_sizes=args.impute_hiddens,
                          dropout=args.dropout)

    # build optimizer
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    Train_loss = []
    Valid_mse = []
    Valid_l1 = []

    best_test_l1 = np.inf
    best_epoch = None

    x = data.x.clone().detach()
    train_edge_index = data.train_edge_index.clone().detach()
    train_edge_attr = data.train_edge_attr.clone().detach()
    test_edge_index = data.test_edge_index.clone().detach()
    test_edge_attr = data.test_edge_attr.clone().detach()
    for epoch in range(args.epochs):

        if scheduler is not None:
            scheduler.step(epoch)
        # for param_group in opt.param_groups:
        #     print('lr',param_group['lr'])

        model.train()
        impute_model.train()

        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2))
        # now concat all masks by it self
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
        pred_train = pred[:int(train_edge_attr.shape[0] / 2)]
        label_train = train_edge_attr[:int(train_edge_attr.shape[0] / 2)]

        loss = F.mse_loss(pred_train, label_train)
        loss.backward()
        opt.step()
        train_loss = loss.item()

        model.eval()
        impute_model.eval()

        x_embd = model(x, train_edge_attr, train_edge_index)
        pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
        pred_test = pred[:int(test_edge_attr.shape[0] / 2)]
        label_test = test_edge_attr[:int(test_edge_attr.shape[0] / 2)]
        mse = F.mse_loss(pred_test, label_test, 'mse')
        test_mse = mse.item()
        l1 = F.l1_loss(pred_test, label_test, 'l1')
        test_l1 = l1.item()

        Train_loss.append(train_loss)
        Valid_mse.append(test_mse)
        Valid_l1.append(test_l1)
        print('epoch: ', epoch)
        print('loss: ', train_loss)
        print('test mse: ', test_mse)
        print('test l1: ', test_l1)

    pred_train = pred_train.detach().numpy()
    label_train = label_train.detach().numpy()
    pred_test = pred_test.detach().numpy()
    label_test = label_test.detach().numpy()

    obj = dict()
    obj['args'] = args
    obj['train_loss'] = Train_loss
    obj['test_mse'] = Valid_mse
    obj['test_l1'] = Valid_l1
    obj['pred_train'] = pred_train
    obj['label_train'] = label_train
    obj['pred_test'] = pred_test
    obj['label_test'] = label_test
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    torch.save(model.state_dict(), log_path + 'model.pt')
    torch.save(impute_model.state_dict(), log_path + 'impute_model.pt')

    obj = objectview(obj)
    plot_result(obj, log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uci_data', type=str, default='housing')  # 'pks', 'cancer', 'housing', 'wine'
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight 1: as input to mlp
    parser.add_argument('--impute_hiddens', type=str, default='64')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_edge', type=float, default=0.7)
    parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--known', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='0')
    args = parser.parse_args()
    args.model_types = args.model_types.split('_')
    if args.impute_hiddens == '':
        args.impute_hiddens = []
    else:
        args.impute_hiddens = list(map(int, args.impute_hiddens.split('_')))

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    df_X = pd.read_csv('./Data/uci/' + args.uci_data + "/" + args.uci_data + '.csv')
    df_y = pd.read_csv('./Data/uci/' + args.uci_data + "/" + args.uci_data + '_target.csv', header=None)
    data = get_data(df_X, df_y, args.train_edge, args.train_y, args.seed)

    log_path = './Data/uci/' + args.uci_data + '/' + args.log_dir + '/'
    os.mkdir(log_path)

    train(data, args, log_path)


if __name__ == '__main__':
    main()