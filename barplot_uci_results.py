import os.path as osp
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# pre_path = "./uci/mdi_results"
# methods = ["knn","mean","mice","svd","spectral","gain","gnn_mdi"]
# method_names = ["KNN","Mean","MICE","SVD","Spectral","GAIN","GNN"]
# comment = '_v2train0.7'
# colors = ['b', 'g', 'c', 'm', 'y', 'pink', 'blue']
# plot_name = 'mdi_known0.7_bar'

# pre_path = "./uci/y_results"
# methods = ["knn","mean","mice","svd","spectral","gain","gnn_mdi","gnn"]
# method_names = ["KNN","Mean","MICE","SVD","Spectral","GAIN","GNN","GNN N2N"]
# base_method = methods[1]
# comment = '_v2train0.7'
# colors = ['b', 'g', 'c', 'm', 'y', 'pink', 'blue', 'red']
# plot_name = 'reg_known0.7_bar'

# pre_path = "./uci/mdi_results"
# methods = ["knn_v2train0.7splitrandom0.7test",
# 			"mean_v2train0.7splitrandom0.7test",
# 			"mice_v2train0.7splitrandom0.7test",
# 			"svd_v2train0.7splitrandom0.7test",
# 			"spectral_v2train0.7splitrandom0.7test",
# 			"gain_v2train0.7splitrandom0.7train",
# 			"gnn_mdi_v2train0.7splitrandom0.7traintest"
# 			]
# method_names = ["KNN","Mean","MICE","SVD","Spectral","GAIN","GNN"]
# base_method = methods[1]
# comment = ''
# colors = ['b', 'g', 'c', 'm', 'y', 'pink', 'blue']
# plot_name = 'mdi_known0.7split0.7_bar'

pre_path = "./uci/mdi_results"
methods = ["gnn_mdi_v2train0.7","gnn_mdi_v2train0.7known1.0"]
method_names = ["GNN","GNN without edge-dropout"]
base_method = methods[1]
comment = ''
colors = ['red', 'blue']
plot_name = 'mdi_known0.7dropout_bar'

data = {"Dataset":[],"Test MAE":[],"Method":[]}
datasets = ["concrete","energy","housing","kin8nm","naval","power",
			"protein","wine","yacht"]
seeds = [0,1,2,3,4]
norm_bases = np.zeros(len(datasets))
for i,dataset in enumerate(datasets):
	for seed in seeds:
		load_path = '{}/results/{}{}/{}/{}/'.format(pre_path,base_method, comment,dataset, seed)
		obj = joblib.load(load_path+'result.pkl')
		# norm_bases[i] += obj['mae']
		if base_method.startswith('gnn_mdi'):
		# if method == 'gnn':
			norm_bases[i] += obj['curves']['test_l1'][-1]
		elif base_method.startswith('gain'):
			norm_bases[i] += obj['mdi_mae']
			# data["Test MAE"].append(obj['reg_mae']/norm_bases[i])
		else:
			norm_bases[i] += obj['mae']
	norm_bases[i] = norm_bases[i]/float(len(seeds))

for method,method_name in zip(methods,method_names):
	for i,dataset in enumerate(datasets):
		for seed in seeds:
			data["Method"].append(method_name)
			data["Dataset"].append(dataset)
			load_path = '{}/results/{}{}/{}/{}/'.format(pre_path,method, comment,dataset, seed)
			obj = joblib.load(load_path+'result.pkl')
			if method.startswith('gnn_mdi'):
			# if method == 'gnn':
				data["Test MAE"].append(obj['curves']['test_l1'][-1]/norm_bases[i])
			elif method.startswith('gain'):
				data["Test MAE"].append(obj['mdi_mae']/norm_bases[i])
				# data["Test MAE"].append(obj['reg_mae']/norm_bases[i])
			else:
				data["Test MAE"].append(obj['mae']/norm_bases[i])
df = pd.DataFrame(data=data)

import seaborn as sns
sns.set(style="ticks", font_scale=3)
plt.figure(figsize=(20,10))
ax = sns.barplot(x="Dataset", y="Test MAE", hue="Method", data=df, palette=colors,
				 errwidth=0.8)
plt.legend(loc='right', bbox_to_anchor=(1.25, 0.5), ncol=1)
# plt.legend(loc='upper center', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.savefig("{}/plots/{}.png".format(pre_path,plot_name), dpi=150, bbox_inches='tight')

