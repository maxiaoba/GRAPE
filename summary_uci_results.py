import os.path as osp
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# pre_path = "./uci/mdi_results"
# methods = ["knn","mean","mice","svd","spectral","gain","gnn_mdi"]
# method_names = ["KNN","Mean","MICE","SVD","Spectral","GAIN","GNN"]
# comment = '_v2train0.7'

# pre_path = "./uci/mdi_results"
# methods = ["knn_v2train0.7splitrandom0.7test",
# 			"mean_v2train0.7splitrandom0.7test",
# 			"mice_v2train0.7splitrandom0.7test",
# 			"svd_v2train0.7splitrandom0.7test",
# 			"spectral_v2train0.7splitrandom0.7test",
# 			"gain_v2train0.7splitrandom0.7train",
# 			"gnn_mdi_v2train0.7splitrandom0.7traintest"
# 			]
# method_names = ["KNN","Mean","MICE","SVD","Spectral","GAIN","MD-GNN"]
# comment = ''

# pre_path = "./uci/y_results"
# methods = ["knn","mean","mice","svd","spectral","gain","tree","gnn"]
# method_names = ["KNN","Mean","MICE","SVD","Spectral","GAIN","Tree","GNN N2N"]
# comment = '_v2train0.3'

# pre_path = "./uci/mdi_results"
# methods = ["gnn_mdi_v2train0.7known1.0","gnn_mdi_v2train0.7"]
# method_names = ["MD-GNN with edge dropout","MD-GNN without edge dropout"]
# comment = ''

pre_path = "./uci/y_results"
methods = ["gnn_mdi","gnn"]
method_names = ["MD-GNN End-to-End","MD-GNN with Linear Regression"]
comment = '_v2train0.7'

datasets = ["concrete","energy","housing","kin8nm","naval","power",
			"protein","wine","yacht"]
seeds = [0,1,2,3,4]

data = np.zeros((len(methods),len(datasets),len(seeds)))
for i,(method,method_name) in enumerate(zip(methods,method_names)):
	for j,dataset in enumerate(datasets):
		for k,seed in enumerate(seeds):
			load_path = '{}/results/{}{}/{}/{}/'.format(pre_path,method, comment,dataset, seed)
			obj = joblib.load(load_path+'result.pkl')
			# if method.startswith('gnn_mdi'):
			if method == 'gnn':
				data[i,j,k] = obj['curves']['test_l1'][-1]
			elif method.startswith('gain'):
				# data[i,j,k] = obj['mdi_mae']
				data[i,j,k] = obj['reg_mae']
			else:
				data[i,j,k] = obj['mae']

data_avg = np.mean(data,-1)
data_std = np.std(data,-1)
data_rel_std = data_std/data_avg
data_avg_std = np.mean(data_std,-1)
data_diff = data_avg - data_avg[-1,:]
data_rel_diff = data_diff/data_avg
data_avg_rel_diff = np.mean(data_rel_diff,-1)
# print(data_rel_diff)
print(data_avg_rel_diff)
# print(data_avg_std)
