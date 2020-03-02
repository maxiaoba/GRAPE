import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from utils import objectview
import pandas as pd
from uci import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--uci_data', type=str, default='housing')
parser.add_argument('--log_dir', type=str, default='y0')
load_args = parser.parse_args()

load_path = './Data/uci/'+load_args.uci_data+'/'+load_args.log_dir+'/'

import joblib
result = joblib.load(load_path+'result.pkl')
result = objectview(result)
args = result.args
for key in args.__dict__.keys():
    print(key,': ',args.__dict__[key])

from plot_utils import plot_result
plot_result(result, load_path)

df_X = pd.read_csv('./Data/uci/'+ args.uci_data +"/"+ args.uci_data +'.csv')
df_y = pd.read_csv('./Data/uci/'+ args.uci_data +"/"+ args.uci_data +'_target.csv', header=None)
dataset = get_dataset(df_X, df_y, args.train_edge, args.train_y, args.seed)

from gnn_model import GNNStack
model = GNNStack(dataset[0].num_node_features, args.node_dim,
                        args.edge_dim, args.edge_mode,
                        args.model_types, args.dropout)
model.load_state_dict(torch.load(load_path+'model.pt'))
model.eval()

from prediction_model import MLPNet
predict_model = MLPNet([args.node_dim], 1, 
                        hidden_layer_sizes=args.predict_hiddens,
                        dropout=args.dropout)
predict_model.load_state_dict(torch.load(load_path+'predict_model.pt'))
predict_model.eval()

for data in dataset:
    x = data.x.clone().detach()
    y = data.y.clone().detach()
    train_edge_index = data.train_edge_index.clone().detach()
    train_edge_attr = data.train_edge_attr.clone().detach()

    x_embd = model(x, train_edge_attr, train_edge_index).detach()
    pred = predict_model(x_embd)[:data.y.shape[0],0]
    pred_test = pred[data.test_y_mask].detach()
    label_test = y[data.test_y_mask].detach()

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,1,1)
plt.plot(pred_test[::10],label='pred')
plt.plot(label_test[::10],label='true')
plt.legend()
plt.subplot(2,1,2)
for i in range(20):
    plt.plot(x_embd[i,:],label=str(i))
plt.legend()
plt.savefig(load_path+'check_embedding.png')

