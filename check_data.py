import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from mc.mc_subparser import add_mc_subparser
from uci.uci_subparser import add_uci_subparser

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
subparsers = parser.add_subparsers()
add_uci_subparser(subparsers)
add_mc_subparser(subparsers)
args = parser.parse_args()
print(args)

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

if args.domain == 'uci':
    from uci.uci_data import load_data
    data = load_data(args)
elif args.domain == 'mc':
    from mc.mc_data import load_data
    data = load_data(args)

# print(data)
print('node dim: ',data.num_node_features,' edge dim: ',data.edge_attr_dim)
for i,key in enumerate(data.keys):
	if torch.is_tensor(data[key]):
	    print(key,': ',data[key].shape)
	# else:
	# 	print(key,': ',data[key])

if hasattr(args,'split_sample') and args.split_sample > 0.:
	lower_y = data.y[data.lower_y_index]
	higher_y = data.y[data.higher_y_index]
	print('lower_y: {} {} {}'.format(lower_y.shape,torch.min(lower_y),torch.max(lower_y)))
	print('higher_y: {} {} {}'.format(higher_y.shape,torch.min(higher_y),torch.max(higher_y)))

	lower_train_edge_index = data.lower_train_edge_index
	lower_train_edge_start1 = lower_train_edge_index[0,:int(lower_train_edge_index.shape[1]/2)]
	lower_train_edge_start2 = lower_train_edge_index[1,int(lower_train_edge_index.shape[1]/2):]
	assert torch.all(torch.eq(lower_train_edge_start1, lower_train_edge_start2))
	for start in lower_train_edge_start1:
		assert start in data.lower_y_index

	higher_train_edge_index = data.higher_train_edge_index
	higher_train_edge_start1 = higher_train_edge_index[0,:int(higher_train_edge_index.shape[1]/2)]
	higher_train_edge_start2 = higher_train_edge_index[1,int(higher_train_edge_index.shape[1]/2):]
	assert torch.all(torch.eq(higher_train_edge_start1, higher_train_edge_start2))
	for start in higher_train_edge_start1:
		assert start in data.higher_y_index

	lower_test_edge_index = data.lower_test_edge_index
	lower_test_edge_start1 = lower_test_edge_index[0,:int(lower_test_edge_index.shape[1]/2)]
	lower_test_edge_start2 = lower_test_edge_index[1,int(lower_test_edge_index.shape[1]/2):]
	assert torch.all(torch.eq(lower_test_edge_start1, lower_test_edge_start2))
	for start in lower_test_edge_start1:
		assert start in data.lower_y_index

	higher_test_edge_index = data.higher_test_edge_index
	higher_test_edge_start1 = higher_test_edge_index[0,:int(higher_test_edge_index.shape[1]/2)]
	higher_test_edge_start2 = higher_test_edge_index[1,int(higher_test_edge_index.shape[1]/2):]
	assert torch.all(torch.eq(higher_test_edge_start1, higher_test_edge_start2))
	for start in higher_test_edge_start1:
		assert start in data.higher_y_index


