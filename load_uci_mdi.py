import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from utils import objectview
import pandas as pd
from uci import get_data

parser = argparse.ArgumentParser()
parser.add_argument('--uci_data', type=str, default='housing')
parser.add_argument('--log_dir', type=str, default='0')
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
data = get_data(df_X, df_y, args.train_edge, args.train_y, args.seed)

from gnn_model import GNNStack
model = GNNStack(data.num_node_features, args.node_dim,
                        args.edge_dim, args.edge_mode,
                        args.model_types, args.dropout)
model.load_state_dict(torch.load(load_path+'model.pt'))
model.eval()
from prediction_model import MLPNet
impute_model = MLPNet([args.node_dim, args.node_dim],
                        hidden_layer_sizes=args.impute_hiddens, 
                        dropout=args.dropout)
impute_model.load_state_dict(torch.load(load_path+'impute_model.pt'))
impute_model.eval()

x = data.x.clone().detach()
train_edge_index = data.train_edge_index.clone().detach()
train_edge_attr = data.train_edge_attr.clone().detach()
test_edge_index = data.test_edge_index.clone().detach()
test_edge_attr = data.test_edge_attr.clone().detach()

x_embd = model(x, train_edge_attr, train_edge_index)
pred = impute_model([x_embd[test_edge_index[0],:],x_embd[test_edge_index[1],:]])
pred_test = pred[:int(test_edge_attr.shape[0]/2)]
label_test = test_edge_attr[:int(test_edge_attr.shape[0]/2)]

Os = {}
for indx in range(20):
    i=test_edge_index[0,indx].detach().numpy()
    j=test_edge_index[1,indx].detach().numpy()
    true=label_test[indx].detach().numpy()
    pred=pred_test[indx].detach().numpy()
    xi=x_embd[i].detach().numpy()
    xj=x_embd[j].detach().numpy()
    if str(i) not in Os.keys():
        Os[str(i)] = {'true':[],'pred':[],'x_j':[]}
    Os[str(i)]['true'].append(true)
    Os[str(i)]['pred'].append(pred)
    Os[str(i)]['x_i'] = xi
    Os[str(i)]['x_j'] += list(xj)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1,3,1)
for i in Os.keys():
    plt.plot(Os[str(i)]['pred'],label='o'+str(i)+'pred')
plt.legend()
plt.subplot(1,3,2)
for i in Os.keys():
    plt.plot(Os[str(i)]['x_i'],label='o'+str(i)+'xi')
    # print(Os[str(i)]['x_i'])
plt.legend()
plt.subplot(1,3,3)
for i in Os.keys():
    plt.plot(Os[str(i)]['x_j'],label='o'+str(i)+'xj')
    # print(Os[str(i)]['x_j'])
plt.legend()
plt.savefig(load_path+'check_embedding.png')

