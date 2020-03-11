import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from utils import objectview
import pandas as pd
from uci import get_data
import torch.nn.functional as F
import pdb

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

## old
# df_xold = pd.read_csv('./Data/uci/'+ args.uci_data +"/"+ args.uci_data +'.csv')
# df_yold = pd.read_csv('./Data/uci/'+ args.uci_data +"/"+ args.uci_data +'_target.csv', header=None)
## new
df_X = pd.read_csv('./Data/uci_all/energy/data/data.txt', header=None, sep='\t')
df_y = df_X.iloc[:, -1]
df_X = df_X.iloc[:, :-1]
pdb.set_trace()

data = get_data(df_X, df_y, args.train_edge, args.train_y, args.seed)
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
impute_model = MLPNet([args.node_dim, args.node_dim],
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

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,1,1)
plt.plot(pred_test.detach().numpy()[::10],label='pred')
plt.plot(label_test.detach().numpy()[::10],label='true')
plt.legend()
plt.subplot(2,1,2)
for i in range(20):
    plt.plot(x_embd.detach().numpy()[i,:],label=str(i))
plt.legend()
plt.savefig(load_path+'check_embedding.png')

