import time
import argparse
import os
import sys, copy, math, time, pdb, warnings, traceback
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

import sys
sys.path.append("..")
from utils import build_optimizer, objectview, get_known_mask, mask_edge

from gnn_model import GNNStack
from prediction_model import MLPNet
from mc_plot_utils import plot_result

from preprocessing import *
from mc_data import get_data, split_edge

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
    Valid_rmse = []

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

        train_loss = []
        train_edge_attrs, train_edge_indexes = \
            split_edge(train_edge_attr,train_edge_index,args.batch_size)
        for (train_edge_attr_i, train_edge_index_i) in \
                zip(train_edge_attrs, train_edge_indexes): 
            known_mask = get_known_mask(args.known,args.batch_size)
            double_known_mask = torch.cat((known_mask, known_mask),dim=0)
            known_edge_index, known_edge_attr = mask_edge(train_edge_index_i,train_edge_attr_i,double_known_mask,True)

            opt.zero_grad()
            x_embd = model(x, known_edge_attr, known_edge_index)
            pred_train = impute_model([x_embd[train_edge_index_i[0,:args.batch_size]],x_embd[train_edge_index_i[1,:args.batch_size]]])
            label_train = train_edge_attr_i[:args.batch_size]

            loss = F.mse_loss(pred_train, label_train)
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)

        model.eval()
        impute_model.eval()

        x_embd = model(x, train_edge_attr, train_edge_index)
        pred = impute_model([x_embd[test_edge_index[0],:],x_embd[test_edge_index[1],:]])
        pred_test = pred[:int(test_edge_attr.shape[0]/2)]
        label_test = test_edge_attr[:int(test_edge_attr.shape[0]/2)]
        mse = F.mse_loss(pred_test, label_test, 'mse')
        test_mse = mse.item()
        test_rmse = np.sqrt(test_mse)

        Train_loss.append(train_loss)
        Valid_mse.append(test_mse)
        Valid_rmse.append(test_rmse)
        print('epoch: ',epoch)
        print('loss: ',train_loss)
        print('test mse: ',test_mse)
        print('test rmse: ',test_rmse)

    pred_train = pred_train.detach().numpy()
    label_train = label_train.detach().numpy()
    pred_test = pred_test.detach().numpy()
    label_test = label_test.detach().numpy()

    obj = dict()
    obj['args'] = args
    obj['train_loss'] = Train_loss
    obj['test_mse'] = Valid_mse
    obj['test_rmse'] = Valid_rmse
    obj['pred_train'] = pred_train
    obj['label_train'] = label_train
    obj['pred_test'] = pred_test
    obj['label_test'] = label_test
    pickle.dump(obj, open(log_path+'result.pkl', "wb" ))

    torch.save(model.state_dict(), log_path+'model.pt')
    torch.save(impute_model.state_dict(), log_path+'impute_model.pt')

    obj = objectview(obj)
    plot_result(obj, log_path)

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Inductive Graph-based Matrix Completion')
    # general settings
    parser.add_argument('--testing', action='store_true', default=True,
                        help='if set, use testing mode which splits all ratings into train/test;\
                        otherwise, use validation model which splits all ratings into \
                        train/val/test and evaluate on val only')
    parser.add_argument('--data-name', default='douban', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--use-features', action='store_true', default=False,
                        help='whether to use node features (side information)')
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--edge_mode', type=int, default=1) # 0: use it as weight 1: as input to mlp
    parser.add_argument('--impute_hiddens', type=str, default='64')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--log_dir', type=str, default='0')
    '''
        Set seeds, prepare for transfer learning (if --transfer)
    '''
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    rating_map, post_rating_map = None, None

    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = load_data_monti(args.data_name, args.seed, args.testing, rating_map, post_rating_map)

    print('All ratings are:')
    print(class_values)
    '''
    Explanations of the above preprocessing:
        class_values are all the original continuous ratings, e.g. 0.5, 2...
        They are transformed to rating labels 0, 1, 2... acsendingly.
        Thus, to get the original rating from a rating label, apply: class_values[label]
        Note that train_labels etc. are all rating labels.
        But the numbers in adj_train are rating labels + 1, why? Because to accomodate neutral ratings 0! Thus, to get any edge label from adj_train, remember to substract 1.
        If testing=True, adj_train will include both train and val ratings, and all train data will be the combination of train and val.
    '''

    if args.use_features:
        u_features, v_features = u_features.toarray(), v_features.toarray()
        n_features = u_features.shape[1] + v_features.shape[1]
        print('Number of user features {}, item features {}, total features {}'.format(u_features.shape[1], v_features.shape[1], n_features))
    else:
        u_features, v_features = None, None
        n_features = 0

    print('#train: %d, #val: %d, #test: %d' % (len(train_u_indices), len(val_u_indices), len(test_u_indices)))

    '''
        Transfer to torch geometric Data
    '''
    data = get_data(u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values)

    '''
        Train and apply the GNN model
    '''
    args.model_types = args.model_types.split('_')
    if args.impute_hiddens == '':
        args.impute_hiddens = []
    else:
        args.impute_hiddens = list(map(int,args.impute_hiddens.split('_')))

    log_path = './Data/'+args.data_name+'/'+args.log_dir+'/'
    os.mkdir(log_path)

    train(data, args, log_path) 

if __name__ == '__main__':
    main()




