import pandas as pd
import os.path as osp
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import undirected
from sklearn import preprocessing

import torch
import random
import numpy as np

from utils import get_known_mask, mask_edge

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

def create_node(df):
    nrow, ncol = df.shape
    feature_ind = np.array(range(ncol))
    feature_node = np.zeros((ncol,ncol))
    feature_node[np.arange(ncol), feature_ind] = 1
    sample_node = [[1]*ncol for i in range(nrow)]
    node = sample_node + feature_node.tolist()
    return node

def get_dataset(df_X, df_y, train_edge, train_y, seed=0):
    #df = pd.read_csv('./Data/uci/'+ uci_data+"/"+uci_data+'.csv')
    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    node_init = create_node(df_X) 
    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y[0].to_numpy(), dtype=torch.float)
    # data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    torch.manual_seed(seed)
    train_edge_mask = get_known_mask(train_edge,int(edge_attr.shape[0]/2))
    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask),dim=0)
    train_edge_index, train_edge_attr = mask_edge(edge_index,edge_attr,
                                                double_train_edge_mask,True)
    test_edge_index, test_edge_attr = mask_edge(edge_index,edge_attr,
                                                ~double_train_edge_mask,True)

    train_y_mask = get_known_mask(train_y, y.shape[0])
    test_y_mask = ~train_y_mask

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
            train_y_mask=train_y_mask,train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,
            test_y_mask=test_y_mask,test_edge_index=test_edge_index,test_edge_attr=test_edge_attr)
    return [data]


