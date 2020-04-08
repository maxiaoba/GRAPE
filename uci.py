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
import pdb

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

def create_node_random(df, initial_node_dim=128):
    nrow, ncol = df.shape
    #feature_ind = np.array(range(ncol))
    #feature_node = np.zeros((ncol,ncol))
    #feature_node[np.arange(ncol), feature_ind] = 1
    feature_node = np.random.uniform(0,1,ncol*initial_node_dim)
    feature_node = feature_node.reshape(ncol, initial_node_dim)
    sample_node = [[1]*initial_node_dim for i in range(nrow)]
    node = sample_node + feature_node.tolist()
    return node

def get_data(df_X, df_y, train_edge_prob, train_y_prob, seed=0, normalize=True):
    if len(df_y.shape)==1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape)==2:
        df_y = df_y[0].to_numpy()

    if normalize:
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)
    
    print("Creating edges...")
    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    print("Creating edge attributes...")
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    print("Creating nodes...")
    #node_init = create_node(df_X)
    node_init = create_node_random(df_X)
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
    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~double_train_edge_mask, True)

    #mask the y-values during training, i.e. how we split the training and test sets
    train_y_mask = get_known_mask(train_y_prob, y.shape[0])
    test_y_mask = ~train_y_mask

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
            train_y_mask=train_y_mask, test_y_mask=test_y_mask,
            train_edge_mask = train_edge_mask,
            train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr,
            df_X=df_X,df_y=df_y)
    
    return data

def get_data_separate(df_X, df_y, train_edge_prob, train_y_prob, seed=0, normalize=True):
    if len(df_y.shape)==1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape)==2:
        df_y = df_y[0].to_numpy()

    if normalize:
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)
    
    print("Creating edges...")
    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    print("Creating edge attributes...")
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    print("Creating nodes...")
    node_init = create_node(df_X) 
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
    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~double_train_edge_mask, True)

    #mask the y-values during training, i.e. how we split the training and test sets
    #train_y_mask = get_known_mask(train_y_prob, y.shape[0])
    #test_y_mask = ~train_y_mask

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
            #train_y_mask=train_y_mask, test_y_mask=test_y_mask,
            train_edge_mask = train_edge_mask,
            train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr,
            df_X=df_X,df_y=df_y)
    
    return data

