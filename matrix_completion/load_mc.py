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
from mc_data import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--data-name', type=str, default='flixster')
parser.add_argument('--log_dir', type=str, default='0')
load_args = parser.parse_args()

load_path = './Data/'+load_args.data_name+'/'+load_args.log_dir+'/'

import joblib
result = joblib.load(load_path+'result.pkl')
result = objectview(result)
args = result.args
for key in args.__dict__.keys():
    print(key,': ',args.__dict__[key])

plot_result(result, load_path)
print(np.min(result.test_rmse),np.argmin(result.test_rmse))

# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(args.seed)
# print(args)
# random.seed(args.seed)
# np.random.seed(args.seed)

# data = load_data(args)

# model = GNNStack(data.num_node_features, args.node_dim,
#                         args.edge_dim, args.edge_mode,
#                         args.model_types, args.dropout)
# model.load_state_dict(torch.load(load_path+'model.pt'))
# model.eval()
# impute_model = MLPNet([args.node_dim, args.node_dim], 1, 
#                         hidden_layer_sizes=args.impute_hiddens,
#                         dropout=args.dropout)
# impute_model.load_state_dict(torch.load(load_path+'impute_model.pt'))
# impute_model.eval()

# x = data.x.clone().detach()
# train_edge_index = data.train_edge_index.clone().detach()
# train_edge_attr = data.train_edge_attr.clone().detach()
# test_edge_index = data.test_edge_index.clone().detach()
# test_edge_attr = data.test_edge_attr.clone().detach()

# x_embd = model(x, train_edge_attr, train_edge_index)
# pred = impute_model([x_embd[test_edge_index[0],:],x_embd[test_edge_index[1],:]])
# pred_test = pred[:int(test_edge_attr.shape[0]/2)]
# label_test = test_edge_attr[:int(test_edge_attr.shape[0]/2)]

# Os = {}
# for indx in range(20):
#     i=test_edge_index[0,indx].detach().numpy()
#     j=test_edge_index[1,indx].detach().numpy()
#     true=label_test[indx].detach().numpy()
#     pred=pred_test[indx].detach().numpy()
#     xi=x_embd[i].detach().numpy()
#     xj=x_embd[j].detach().numpy()
#     if str(i) not in Os.keys():
#         Os[str(i)] = {'true':[],'pred':[],'x_j':[]}
#     Os[str(i)]['true'].append(true)
#     Os[str(i)]['pred'].append(pred)
#     Os[str(i)]['x_i'] = xi
#     Os[str(i)]['x_j'] += list(xj)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.subplot(1,3,1)
# for i in Os.keys():
#     plt.plot(Os[str(i)]['pred'],label='o'+str(i)+'pred')
# plt.legend()
# plt.subplot(1,3,2)
# for i in Os.keys():
#     plt.plot(Os[str(i)]['x_i'],label='o'+str(i)+'xi')
#     # print(Os[str(i)]['x_i'])
# plt.legend()
# plt.subplot(1,3,3)
# for i in Os.keys():
#     plt.plot(Os[str(i)]['x_j'],label='o'+str(i)+'xj')
#     # print(Os[str(i)]['x_j'])
# plt.legend()
# plt.savefig(load_path+'check_embedding.png')

