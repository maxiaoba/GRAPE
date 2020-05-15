import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing

import torch
import random
import numpy as np
import pdb

from utils.utils import get_known_mask, mask_edge

def create_node(df, mode):
    if mode == 0: # onehot feature node, all 1 sample node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol))
        feature_node[np.arange(ncol), feature_ind] = 1
        sample_node = [[1]*ncol for i in range(nrow)]
        node = sample_node + feature_node.tolist()
    elif mode == 1: # onehot sample and feature node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol+1))
        feature_node[np.arange(ncol), feature_ind+1] = 1
        sample_node = np.zeros((nrow,ncol+1))
        sample_node[:,0] = 1
        node = sample_node.tolist() + feature_node.tolist()
    return node

def create_edge(df):
    n_row, n_col = df.shape
    edge_start = []
    edge_end = []
    for x in range(n_row):
        edge_start = edge_start + [x] * n_col # obj
        edge_end = edge_end + list(n_row+np.arange(n_col)) # att    
    edge_start_new = edge_start + edge_end
    edge_end_new = edge_end + edge_start
    return (edge_start_new, edge_end_new)

def create_edge_attr(df):
    nrow, ncol = df.shape
    edge_attr = []
    for i in range(nrow):
        for j in range(ncol):
            edge_attr.append([float(df.iloc[i,j])])
    edge_attr = edge_attr + edge_attr
    return edge_attr

def get_data(df_X, df_y, node_mode, train_edge_prob, split_sample_ratio, split_by, train_y_prob, seed=0, normalize=True):
    if len(df_y.shape)==1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape)==2:
        df_y = df_y[0].to_numpy()

    if normalize:
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)
    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    node_init = create_node(df_X, node_mode) 
    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)
    
    #set seed to fix known/unknwon edges
    torch.manual_seed(seed)
    #keep train_edge_prob of all edges
    train_edge_mask = get_known_mask(train_edge_prob, int(edge_attr.shape[0]/2))
    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)

    #mask edges based on the generated train_edge_mask
    #train_edge_index is known, test_edge_index in unknwon, i.e. missing
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr,
                                                double_train_edge_mask, True)
    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2),0]
    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~double_train_edge_mask, True)
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2),0]
    #mask the y-values during training, i.e. how we split the training and test sets
    train_y_mask = get_known_mask(train_y_prob, y.shape[0])
    test_y_mask = ~train_y_mask

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
            train_y_mask=train_y_mask, test_y_mask=test_y_mask,
            train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,
            train_edge_mask=train_edge_mask,train_labels=train_labels,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr,
            test_edge_mask=~train_edge_mask,test_labels=test_labels, 
            df_X=df_X,df_y=df_y,
            edge_attr_dim=train_edge_attr.shape[-1],
            user_num=df_X.shape[0]
            )

    if split_sample_ratio > 0.:
        if split_by == 'y':
            sorted_y, sorted_y_index = torch.sort(torch.reshape(y,(-1,)))
        elif split_by == 'random':
            sorted_y_index = torch.randperm(y.shape[0])
        lower_y_index = sorted_y_index[:int(np.floor(y.shape[0]*split_sample_ratio))]
        higher_y_index = sorted_y_index[int(np.floor(y.shape[0]*split_sample_ratio)):]
        # here we don't split x, only split edge
        # train
        half_train_edge_index = train_edge_index[:,:int(train_edge_index.shape[1]/2)];
        lower_train_edge_mask = []
        for node_index in half_train_edge_index[0]:
            if node_index in lower_y_index:
                lower_train_edge_mask.append(True)
            else:
                lower_train_edge_mask.append(False)
        lower_train_edge_mask = torch.tensor(lower_train_edge_mask)
        double_lower_train_edge_mask = torch.cat((lower_train_edge_mask, lower_train_edge_mask), dim=0)
        lower_train_edge_index, lower_train_edge_attr = mask_edge(train_edge_index, train_edge_attr,
                                                double_lower_train_edge_mask, True)
        lower_train_labels = lower_train_edge_attr[:int(lower_train_edge_attr.shape[0]/2),0]
        higher_train_edge_index, higher_train_edge_attr = mask_edge(train_edge_index, train_edge_attr,
                                                ~double_lower_train_edge_mask, True)
        higher_train_labels = higher_train_edge_attr[:int(higher_train_edge_attr.shape[0]/2),0]
        # test
        half_test_edge_index = test_edge_index[:,:int(test_edge_index.shape[1]/2)];
        lower_test_edge_mask = []
        for node_index in half_test_edge_index[0]:
            if node_index in lower_y_index:
                lower_test_edge_mask.append(True)
            else:
                lower_test_edge_mask.append(False)
        lower_test_edge_mask = torch.tensor(lower_test_edge_mask)
        double_lower_test_edge_mask = torch.cat((lower_test_edge_mask, lower_test_edge_mask), dim=0)
        lower_test_edge_index, lower_test_edge_attr = mask_edge(test_edge_index, test_edge_attr,
                                                double_lower_test_edge_mask, True)
        lower_test_labels = lower_test_edge_attr[:int(lower_test_edge_attr.shape[0]/2),0]
        higher_test_edge_index, higher_test_edge_attr = mask_edge(test_edge_index, test_edge_attr,
                                                ~double_lower_test_edge_mask, True)
        higher_test_labels = higher_test_edge_attr[:int(higher_test_edge_attr.shape[0]/2),0]


        data.lower_y_index = lower_y_index
        data.higher_y_index = higher_y_index
        data.lower_train_edge_index = lower_train_edge_index
        data.lower_train_edge_attr = lower_train_edge_attr
        data.lower_train_labels = lower_train_labels
        data.higher_train_edge_index = higher_train_edge_index
        data.higher_train_edge_attr = higher_train_edge_attr
        data.higher_train_labels = higher_train_labels
        data.lower_test_edge_index = lower_test_edge_index
        data.lower_test_edge_attr = lower_test_edge_attr
        data.lower_test_labels = lower_train_labels
        data.higher_test_edge_index = higher_test_edge_index
        data.higher_test_edge_attr = higher_test_edge_attr
        data.higher_test_labels = higher_test_labels
        
    return data

def load_data(args):
    uci_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    df_np = np.loadtxt(uci_path+'/raw_data/{}/data/data.txt'.format(args.data))
    df_y = pd.DataFrame(df_np[:, -1:])
    df_X = pd.DataFrame(df_np[:, :-1])
    if not hasattr(args,'split_sample'):
        args.split_sample = 0
    data = get_data(df_X, df_y, args.node_mode, args.train_edge, args.split_sample, args.split_by, args.train_y, args.seed)
    return data


