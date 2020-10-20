import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from utils.utils import objectview
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, default='uci')
parser.add_argument('--data', type=str, default='housing')
parser.add_argument('--log_dir', type=str, default='y0')
parser.add_argument('--log_extra', type=str, default='')
load_args = parser.parse_args()

load_path = './{}/test/{}/{}/'.format(load_args.domain,load_args.data,load_args.log_dir)

import joblib
result = joblib.load(load_path+'result.pkl')
result = objectview(result)
args = result.args
for key in args.__dict__.keys():
    print(key,': ',args.__dict__[key])

if args.domain == 'uci':
    from uci.uci_data import load_data
    data = load_data(args)

model = torch.load(load_path+'model'+load_args.log_extra+'.pt',map_location=torch.device('cpu'))
model.eval()
impute_model = torch.load(load_path+'impute_model'+load_args.log_extra+'.pt',map_location=torch.device('cpu'))
impute_model.eval()
predict_model = torch.load(load_path+'predict_model'+load_args.log_extra+'.pt',map_location=torch.device('cpu'))
predict_model.eval()

n_row, n_col = data.df_X.shape
x = data.x.clone().detach()
y = data.y.clone().detach()
edge_index = data.edge_index.clone().detach()  
train_edge_index = data.train_edge_index.clone().detach()
train_edge_attr = data.train_edge_attr.clone().detach()
train_y_mask = data.train_y_mask.clone().detach()
test_y_mask = data.test_y_mask.clone().detach()

x_embd = model(x, train_edge_attr, train_edge_index)
X = impute_model([x_embd[edge_index[0,:int(n_row*n_col)]],x_embd[edge_index[1,:int(n_row*n_col)]]])
X = torch.reshape(X, [n_row, n_col])
pred = predict_model(X)[:,0]
pred_test = pred[test_y_mask]
label_test = y[test_y_mask]

mse = F.mse_loss(pred_test, label_test)
test_rmse = np.sqrt(mse.item())
l1 = F.l1_loss(pred_test, label_test)
test_l1 = l1.item()
print("test rmse: ",test_rmse, " l1: ",test_l1)

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
plt.savefig(load_path+'check_embedding'+load_args.log_extra+'.png')

