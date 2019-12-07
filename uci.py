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

'''
1) ID number
2) Outcome (R = recur, N = nonrecur)
3) Time (recurrence time if field 2 = R, disease-free time if 
    field 2 = N)
4-33) Ten real-valued features are computed for each cell nucleus:

    a) radius (mean of distances from center to points on the perimeter)
    b) texture (standard deviation of gray-scale values)
    c) perimeter
    d) area
    e) smoothness (local variation in radius lengths)
    f) compactness (perimeter^2 / area - 1.0)
    g) concavity (severity of concave portions of the contour)
    h) concave points (number of concave portions of the contour)
    i) symmetry 
    j) fractal dimension ("coastline approximation" - 1)
'''


#print(df.head())

def process(df):
    df = df.drop(df.columns[[0, 1, 2]], axis=1)
    nrow, ncol = df.shape
    for i in range(nrow):
        for j in range(ncol):
            if df.iloc[i,j] == "?":
                df.iloc[i, j] = 0
                
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled)
    return df_normalized

def remove_edges(df, nrem):
    nrow, ncol = df.shape
    remove_rows = random.sample(range(nrow), nrem)
    remove_cols = random.sample(range(ncol), nrem)
    rem_edges = [[remove_rows[i],remove_cols[i]] 
                 for i in range(len(remove_rows))]
    return rem_edges

def create_edge(df, remove_rate=0.3):
    n_row, n_col = df.shape
    edge_start = []
    edge_end = []
    for x in range(n_row):
        edge_start = edge_start + [x] * n_col # obj
        edge_end = edge_end + list(n_row+np.arange(n_col)) # att

    #_edge = len(edge_start)
    #n_rem = int(np.floor(n_edge * remove_rate))
    #remove_ind = random.sample(range(n_edge), n_rem)
    #edge_start = [edge_start[i] for i in range(n_edge) if i not in remove_ind]
    #edge_end = [edge_end[i] for i in range(n_edge) if i not in remove_ind]
    
    edge_start_new = edge_start + edge_end
    edge_end_new = edge_end + edge_start
    #print(len(edge_start))
    return (edge_start_new, edge_end_new)
    #return (edge_start, edge_end)

def create_edge_attr(df):
    nrow, ncol = df.shape
    edge_attr = []

    for i in range(nrow):
        for j in range(ncol):
            edge_attr.append([float(df.iloc[i,j])])
    edge_attr = edge_attr+edge_attr
    return edge_attr

def create_node(df):
    nrow, ncol = df.shape
    feature_ind = np.array(range(ncol))
    feature_node = np.zeros((ncol,ncol))
    feature_node[np.arange(ncol), feature_ind] = 1

    sample_node = [[1]*ncol for i in range(nrow)]
    node = sample_node + feature_node.tolist()
    return node

def get_dataset(uci_data="cancer", processed_df_file=None):
    if uci_data == "cancer":
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data',
                        header=None)
        #df = df.drop(df.columns[[0, 1]], axis=1)
        df = process(df)
        #nrow, ncol = df.shape
    else:
        df = pd.read_csv(processed_df_file)
    edge_start, edge_end = create_edge(df)
    edge_index = torch.tensor([edge_start, edge_end], dtype=torch.float)
    edge_attr = torch.tensor(create_edge_attr(df), dtype=torch.float)
    node_init = create_node(df) 
    #x = torch.tensor([[1] for x in range(nrow+ncol)], dtype=torch.float)
    x = torch.tensor(node_init, dtype=torch.float)
    # rev_edges = remove_edges(df, 90)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return [data]

def pre_transform(data):
    df = process(data)
    #nrow, ncol = df.shape
    edge_start, edge_end = create_edge(df)
    edge_index = torch.tensor([edge_start, edge_end], dtype=torch.float)
    edge_attr = torch.tensor(create_edge_attr(df), dtype=torch.float)
    node_init = create_node(df) 
    # x = torch.tensor([[1] for x in range(nrow+ncol)], dtype=torch.float)
    x = torch.tensor(node_init, dtype=torch.float)
    # rev_edges = remove_edges(df, 90)
    dataset = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    #print(dataset.is_undirected())
    nrow, ncol = df.shape
    dataset.num_obj = nrow
    dataset.num_att = ncol
    return dataset

def pre_transform_sim(data):
    df = data.drop(data.columns[[0]], axis=1)
    #df = df[df.columns[[1,2]]]
    #nrow, ncol = df.shape
    edge_start, edge_end = create_edge(df)
    edge_index = torch.tensor([edge_start, edge_end], dtype=torch.float)
    edge_attr = torch.tensor(create_edge_attr(df), dtype=torch.float)
    node_init = create_node(df) 
    # x = torch.tensor([[1] for x in range(nrow+ncol)], dtype=torch.float)
    x = torch.tensor(node_init, dtype=torch.float)
    # rev_edges = remove_edges(df, 90)
    dataset = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return dataset

class UCIDataset(Dataset):
    def __init__(self, root, pre_transform=pre_transform, transform=None,):
        super(UCIDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['uci_raw_1.pkl',]

    @property
    def processed_file_names(self):
        return ['data_1.pt',]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data',
                        header=None)

        file_path = folder = osp.join(self.root, 'raw',self.raw_file_names[0])
        df.to_pickle(file_path)

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = pd.read_pickle(raw_path)

            if self.pre_filter is not None and not self.pre_filter(data):
                 continue

            if self.pre_transform is not None:
                 data = self.pre_transform(data)
                 #print(type(data.edge_index))
                 #data = data.to_undirected()
                 print("Check directed/undirected")
                 print(data.is_undirected())

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


class SimDataset(Dataset):
    def __init__(self, root, pre_transform=pre_transform_sim, transform=None,):
        super(SimDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['sim_raw_1.pkl',]

    @property
    def processed_file_names(self):
        return ['simdata_1.pt',]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        df = pd.read_csv("./simulated_data.csv")
        file_path = folder = osp.join(self.root, 'raw',self.raw_file_names[0])
        df.to_pickle(file_path)

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = pd.read_pickle(raw_path)

            if self.pre_filter is not None and not self.pre_filter(data):
                 continue

            if self.pre_transform is not None:
                 data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
