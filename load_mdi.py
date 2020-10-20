import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from utils.utils import objectview
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, default='mc')
parser.add_argument('--data', type=str, default='douban')
parser.add_argument('--log_dir', type=str, default='0')
parser.add_argument('--log_extra', type=str, default='_best_valid_rmse')
load_args = parser.parse_args()

load_path = './{}/test/{}/{}/'.format(load_args.domain,load_args.data,load_args.log_dir)

import joblib
result = joblib.load(load_path+'result.pkl')
result = objectview(result)
args = result.args
if not hasattr(args,'one_hot_edge'):
    args.one_hot_edge = False
if not hasattr(args,'ce_loss'):
    args.ce_loss = False
if not hasattr(args,'node_mode'):
    args.node_mode = 0
for key in args.__dict__.keys():
    print(key,': ',args.__dict__[key])

### test rmse from saved curve

curves = result.curves
if load_args.log_extra == '':
    test_rmse = curves['test_rmse'][-1]
    test_mae = curves['test_l1'][-1]
    print('test rmse is {:.3g}, test mae is {:.3g}'.format(test_rmse,test_mae))
else:
    min_valid_rmse = np.min(curves['valid_rmse'])
    min_valid_rmse_index = np.argmin(curves['valid_rmse'])
    test_rmse = curves['test_rmse'][min_valid_rmse_index]
    print('test rmse is {:.3g} at {}'.format(test_rmse,min_valid_rmse_index))

## check loss jumping
# train_curve = curves['train_loss']
# for epoch in range(len(train_curve)):
#     train_loss = train_curve[epoch]
#     Train_loss = train_curve[:epoch]
#     if (len(Train_loss)>200):
#         jump = train_loss-Train_loss[-1]
#         pre20jump = sum(abs(np.array([Train_loss[-i-1]-Train_loss[-i] for i in range(1,11)])))
#         # print("jump is {:.3g}, total pre20 jump is {:.3g}".format(jump,pre20jump))
#         if jump > pre20jump:
#             print("loss jumping at epoch {} of value {:.3g}".format(epoch,train_loss))

### test rmse from saved prediction

# if args.domain == 'uci':
#     from uci.uci_data import load_data
#     data = load_data(args)
# elif args.domain == 'mc':
#     from mc.mc_data import load_data
#     data = load_data(args)

# test_labels = data.test_labels.clone().detach()
# if hasattr(data,'class_values'):
#     class_values = data.class_values.clone().detach()
# if hasattr(args,'ce_loss') and args.ce_loss:
#     label_test = class_values[test_labels]
# elif hasattr(args,'norm_label') and args.norm_label:
#     label_test = test_labels
#     label_test = label_test * max(class_values)
# else:
#     label_test = test_labels

# outputs = result.outputs
# pred_test = torch.tensor(outputs['best_valid_rmse_pred_test'])
# mse = F.mse_loss(pred_test, label_test)
# test_rmse = np.sqrt(mse.item())
# l1 = F.l1_loss(pred_test, label_test)
# test_l1 = l1.item()

# mse = F.mse_loss(pred_test, label_test)
# test_rmse = np.sqrt(mse.item())
# l1 = F.l1_loss(pred_test, label_test)
# test_l1 = l1.item()
# print("test rmse: ",test_rmse, " l1: ",test_l1)

### test rmse from saved model

# if args.domain == 'uci':
#     from uci.uci_data import load_data
#     data = load_data(args)
# elif args.domain == 'mc':
#     from mc.mc_data import load_data
#     data = load_data(args)

# model = torch.load(load_path+'model'+load_args.log_extra+'.pt',map_location=torch.device('cpu'))
# model.eval()
# impute_model = torch.load(load_path+'impute_model'+load_args.log_extra+'.pt',map_location=torch.device('cpu'))
# impute_model.eval()

# x = data.x.clone().detach()
# train_edge_index = data.train_edge_index.clone().detach()
# train_edge_attr = data.train_edge_attr.clone().detach()
# train_labels = data.train_labels.clone().detach()
# test_edge_index = data.test_edge_index.clone().detach()
# test_edge_attr = data.test_edge_attr.clone().detach()
# test_labels = data.test_labels.clone().detach()
# if hasattr(data,'class_values'):
#     class_values = data.class_values.clone().detach()

# x_embd = model(x, train_edge_attr, train_edge_index)
# pred = impute_model([x_embd[test_edge_index[0],:],x_embd[test_edge_index[1],:]])
# if hasattr(args,'ce_loss') and args.ce_loss:
#     pred_test = class_values[pred[:int(test_edge_attr.shape[0] / 2)].max(1)[1]]
#     label_test = class_values[test_labels]
# elif hasattr(args,'norm_label') and args.norm_label:
#     pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
#     pred_test = pred_test * max(class_values)
#     label_test = test_labels
#     label_test = label_test * max(class_values)
# else:
#     pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
#     label_test = test_labels

# mse = F.mse_loss(pred_test, label_test)
# test_rmse = np.sqrt(mse.item())
# l1 = F.l1_loss(pred_test, label_test)
# test_l1 = l1.item()

# mse = F.mse_loss(pred_test, label_test)
# test_rmse = np.sqrt(mse.item())
# l1 = F.l1_loss(pred_test, label_test)
# test_l1 = l1.item()
# print("test rmse: ",test_rmse, " l1: ",test_l1)

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

# plt.figure()
# plt.subplot(1,3,1)
# for i in Os.keys():
#     plt.plot(Os[str(i)]['pred'],label='o'+str(i)+'pred')
# plt.legend()
# plt.subplot(1,3,2)
# for i in Os.keys():
#     plt.plot(Os[str(i)]['x_i'],label='o'+str(i)+'xi')
# plt.legend()
# plt.subplot(1,3,3)
# for i in Os.keys():
#     plt.plot(Os[str(i)]['x_j'],label='o'+str(i)+'xj')
# plt.legend()
# plt.savefig(load_path+'check_embedding'+load_args.log_extra+'.png')

