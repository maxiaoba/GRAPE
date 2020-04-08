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

load_dirs = load_args.log_dir.split('_')
load_paths = ['./{}/test/{}/{}/'.format(load_args.domain,load_args.data,load_dir) for load_dir in load_dirs]

import joblib
result = joblib.load(load_paths[0]+'result.pkl')
result = objectview(result)
args = result.args
if not hasattr(args,'one_hot_edge'):
    args.one_hot_edge = False
if not hasattr(args,'ce_loss'):
    args.ce_loss = False

if args.domain == 'uci':
    from uci.uci_data import load_data
    data = load_data(args)
elif args.domain == 'mc':
    from mc.mc_data import load_data
    data = load_data(args)

test_labels = data.test_labels.clone().detach()
if hasattr(data,'class_values'):
    class_values = data.class_values.clone().detach()
if hasattr(args,'ce_loss') and args.ce_loss:
    label_test = class_values[test_labels]
elif hasattr(args,'norm_label') and args.norm_label:
    label_test = test_labels
    label_test = label_test * max(class_values)
else:
    label_test = test_labels

preds = torch.zeros(test_labels.shape[0],dtype=float)
for load_path in load_paths:
	outputs = result.outputs
	pred_test = torch.tensor(outputs['best_valid_rmse_pred_test'])
	preds += pred_test

pred_test = preds/float(len(load_paths))
mse = F.mse_loss(pred_test, label_test)
test_rmse = np.sqrt(mse.item())
l1 = F.l1_loss(pred_test, label_test)
test_l1 = l1.item()

mse = F.mse_loss(pred_test, label_test)
test_rmse = np.sqrt(mse.item())
l1 = F.l1_loss(pred_test, label_test)
test_l1 = l1.item()
print("test rmse: ",test_rmse, " l1: ",test_l1)

# x = data.x.clone().detach()
# train_edge_index = data.train_edge_index.clone().detach()
# train_edge_attr = data.train_edge_attr.clone().detach()
# train_labels = data.train_labels.clone().detach()
# test_edge_index = data.test_edge_index.clone().detach()
# test_edge_attr = data.test_edge_attr.clone().detach()
# test_labels = data.test_labels.clone().detach()
# if hasattr(data,'class_values'):
#     class_values = data.class_values.clone().detach()

# preds = torch.zeros((test_edge_index.shape[1],1),dtype=float)
# for load_path in load_paths:
# 	model = torch.load(load_path+'model'+load_args.log_extra+'.pt',map_location=torch.device('cpu'))
# 	model.eval()
# 	impute_model = torch.load(load_path+'impute_model'+load_args.log_extra+'.pt',map_location=torch.device('cpu'))
# 	impute_model.eval()
# 	x_embd = model(x, train_edge_attr, train_edge_index)
# 	pred = impute_model([x_embd[test_edge_index[0],:],x_embd[test_edge_index[1],:]])
# 	preds += pred
# pred = preds/float(len(load_paths))

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
