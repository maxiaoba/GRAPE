import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from utils import objectview
import pandas as pd
from uci import get_data
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--uci_data', type=str, default='housing')
parser.add_argument('--train_edge', type=float, default=0.7)
parser.add_argument('--train_y', type=float, default=0.7)
parser.add_argument('--load_dir', type=str, default='y0')
parser.add_argument('--seed', type=int, default=0)
load_args = parser.parse_args()

load_path = './Data/uci/'+load_args.uci_data+'/'+load_args.load_dir+'/'

import joblib
result = joblib.load(load_path+'result.pkl')
result = objectview(result)
args = result.args

df_X = pd.read_csv('./Data/uci/'+ args.uci_data +"/"+ args.uci_data +'.csv')
df_y = pd.read_csv('./Data/uci/'+ args.uci_data +"/"+ args.uci_data +'_target.csv', header=None)
data = get_data(df_X, df_y, load_args.train_edge, load_args.train_y, load_args.seed)
n_row, n_col = data.df_X.shape
x = data.x.clone().detach()
y = data.y.clone().detach()
edge_index = data.edge_index.clone().detach()  
train_edge_index = data.train_edge_index.clone().detach()
train_edge_attr = data.train_edge_attr.clone().detach()
train_y_mask = data.train_y_mask.clone().detach()
test_y_mask = data.test_y_mask.clone().detach()

from gnn_model import GNNStack
model = GNNStack(data.num_node_features, args.node_dim,
                        args.edge_dim, args.edge_mode,
                        args.model_types, args.dropout)
model.load_state_dict(torch.load(load_path+'model.pt'))
model.eval()
from prediction_model import MLPNet
impute_model = MLPNet([args.node_dim, args.node_dim], 1,
                        hidden_layer_sizes=args.impute_hiddens, 
                        dropout=args.dropout)
impute_model.load_state_dict(torch.load(load_path+'impute_model.pt'))
impute_model.eval()
from prediction_model import MLPNet
predict_model = MLPNet([n_col], 1,
                        hidden_layer_sizes=args.predict_hiddens, 
                        dropout=args.dropout)
predict_model.load_state_dict(torch.load(load_path+'predict_model.pt'))
predict_model.eval()

x_embd = model(x, train_edge_attr, train_edge_index)
X = impute_model([x_embd[edge_index[0,:int(n_row*n_col)]],x_embd[edge_index[1,:int(n_row*n_col)]]])
X = torch.reshape(X, [n_row, n_col])
pred = predict_model(X)[:,0]
pred_test = pred[test_y_mask]
label_test = y[test_y_mask]
mse = F.mse_loss(pred_test, label_test, 'mse')
test_mse = mse.item()
l1 = F.l1_loss(pred_test, label_test, 'l1')
test_l1 = l1.item()
print("mse: ",test_mse," l1: ",test_l1)

