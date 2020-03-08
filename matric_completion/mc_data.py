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

def create_node(df):
    nrow, ncol = df.shape
    feature_ind = np.array(range(ncol))
    feature_node = np.zeros((ncol,ncol))
    feature_node[np.arange(ncol), feature_ind] = 1
    sample_node = [[1]*ncol for i in range(nrow)]
    node = sample_node + feature_node.tolist()
    return node

def get_data(u_features, v_features, adj_train,
    train_labels, train_u_indices, train_v_indices,
    val_labels, val_u_indices, val_v_indices, 
    test_labels, test_u_indices, test_v_indices, 
    class_values):

    n_row, n_col = adj_train.shape
    if (u_features is None) or (v_features is None):
        x = torch.tensor(create_node(adj_train), dtype=torch.float)
    else:
        print("using given feature")
        # x = torch.cat((torch.tensor(u_features,dtype=torch.float),
        #                 torch.tensor(v_features,dtype=torch.float)),
        #                 dim=0)
        x = torch.zeros((u_features.shape[0]+v_features.shape[0],
                        np.maximum(u_features.shape[1],v_features.shape[1]))
                        ,dtype=torch.float)
        x[:u_features.shape[0],:u_features.shape[1]] = u_features
        x[u_features.shape[0]:,:v_features.shape[1]] = v_features

    train_v_indices = train_v_indices + n_row
    train_edge_index = torch.tensor([np.append(train_u_indices,train_v_indices),
                        np.append(train_v_indices,train_u_indices)],
                        dtype=int)
    train_edge_attr = torch.tensor(np.append(train_labels,train_labels)[:,None],
                        dtype=torch.float)

    val_v_indices = val_v_indices + n_row
    val_edge_index = torch.tensor([np.append(val_u_indices,val_v_indices),
                        np.append(val_v_indices,val_u_indices)],
                        dtype=int)
    val_edge_attr = torch.tensor(np.append(val_labels,val_labels)[:,None],
                        dtype=torch.float)

    test_v_indices = test_v_indices + n_row
    test_edge_index = torch.tensor([np.append(test_u_indices,test_v_indices),
                        np.append(test_v_indices,test_u_indices)],
                        dtype=int)
    test_edge_attr = torch.tensor(np.append(test_labels,test_labels)[:,None],
                        dtype=torch.float)

    data = Data(x=x,
            train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,
            val_edge_index=val_edge_index,val_edge_attr=val_edge_attr,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr,
            )
    return data

def split_edge(edge_attr,edge_index,batch_size):
    edge_num = int(edge_attr.shape[0]/2)
    perm = np.random.permutation(edge_num)
    edge_attrs, edge_indexes = [],[]
    index = 0
    while index + batch_size < edge_num:
        edge_attr_i = torch.cat((edge_attr[index:index+batch_size,:],
                                edge_attr[index:index+batch_size,:]),dim=0)
        edge_start_i = torch.cat([edge_index[0,index:index+batch_size],
                                    edge_index[1,index:index+batch_size]])
        edge_end_i = torch.cat([edge_index[1,index:index+batch_size],
                                    edge_index[0,index:index+batch_size]])
        edge_index_i = torch.tensor([edge_start_i.detach().numpy(),
                                    edge_end_i.detach().numpy()],
                                    dtype=int)
        index += batch_size
        edge_attrs.append(edge_attr_i)
        edge_indexes.append(edge_index_i)
    return edge_attrs, edge_indexes


