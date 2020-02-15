from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch
import argparse
import pandas as pd
from uci import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--uci_data', type=str, default='housing') # 'pks', 'cancer', 'housing', 'wine'
parser.add_argument('--train_edge', type=float, default=0.7)
parser.add_argument('--train_y', type=float, default=0.7)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

df_X = pd.read_csv('./Data/uci/'+ args.uci_data +"/"+ args.uci_data +'.csv')
df_y = pd.read_csv('./Data/uci/'+ args.uci_data +"/"+ args.uci_data +'_target.csv', header=None)
dataset = get_dataset(df_X, df_y, args.train_edge, args.train_y, args.seed)
data = dataset[0]
print(data.x.shape)
print(data.y.shape,data.edge_index.shape,data.edge_attr.shape)
print(torch.sum(data.train_y_mask),data.train_edge_index.shape,data.train_edge_attr.shape)
print(torch.sum(data.test_y_mask),data.test_edge_index.shape,data.test_edge_attr.shape)


