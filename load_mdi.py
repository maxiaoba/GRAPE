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
parser.add_argument('--log_dir', type=str, default='0')
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
elif args.domain == 'mc':
    from mc.mc_data import load_data
    data = load_data(args)

model = torch.load(load_path+'model'+load_args.log_extra+'.pt')
model.eval()
impute_model = torch.load(load_path+'impute_model'+load_args.log_extra+'.pt')
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

mse = F.mse_loss(pred_test, label_test)
test_rmse = np.sqrt(mse.item())
l1 = F.l1_loss(pred_test, label_test)
test_l1 = l1.item()
print("test rmse: ",test_rmse, " l1: ",test_l1)

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

plt.figure()
plt.subplot(1,3,1)
for i in Os.keys():
    plt.plot(Os[str(i)]['pred'],label='o'+str(i)+'pred')
plt.legend()
plt.subplot(1,3,2)
for i in Os.keys():
    plt.plot(Os[str(i)]['x_i'],label='o'+str(i)+'xi')
plt.legend()
plt.subplot(1,3,3)
for i in Os.keys():
    plt.plot(Os[str(i)]['x_j'],label='o'+str(i)+'xj')
plt.legend()
plt.savefig(load_path+'check_embedding'+load_args.log_extra+'.png')

