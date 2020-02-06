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

def get_dataset(df):
    #df = pd.read_csv('./Data/uci/'+ uci_data+"/"+uci_data+'.csv')
    edge_start, edge_end = create_edge(df)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_attr = torch.tensor(create_edge_attr(df), dtype=torch.float)
    node_init = create_node(df) 
    x = torch.tensor(node_init, dtype=torch.float)
    #generate random response for now
    y = torch.tensor(np.random.normal(0,1,len(df)), dtype=torch.float)
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    return [data]


