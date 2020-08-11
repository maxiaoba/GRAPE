import os.path as osp
import numpy as np
import joblib

pre_path = "./uci/mdi_results"
# pre_path = "./uci/y_results"

# methods = ["knn","mean","mice","svd","spectral","gain","gnn_mdi"]
# method_names = ["knn","mean","mice","svd","spectral","gain","gnn"]

# methods = ["knn","mean","mice","svd","spectral","gain","gnn_mdi","gnn"]
# method_names = ["knn","mean","mice","svd","spectral","gain","gnn","gnn n2n"]

# methods = ["gnn_v2train0.7","gnn_mdi_v2train0.7"]
# method_names = ["MD-GNN End-to-End","MD-GNN with Linear Regression"]
# table_name = 'n2n'

# methods = ["gnn_mdi_v2train0.7","gnn_mdi_v2train0.7known1.0"]
# method_names = ["MD-GNN with edge dropout","MD-GNN without edge dropout"]
# table_name = 'edgedropout'

# methods = ["knn_v2train0.7splitrandom0.7","knn_v2train0.7splitrandom0.7test",
# 			"mean_v2train0.7splitrandom0.7","mean_v2train0.7splitrandom0.7test",
# 			"mice_v2train0.7splitrandom0.7","mice_v2train0.7splitrandom0.7test",
# 			"svd_v2train0.7splitrandom0.7","svd_v2train0.7splitrandom0.7test",
# 			"spectral_v2train0.7splitrandom0.7","spectral_v2train0.7splitrandom0.7test",
# 			"gain_v2train0.7splitrandom0.7","gain_v2train0.7splitrandom0.7train",
# 			"gnn_mdi_v2train0.7splitrandom0.7train","gnn_mdi_v2train0.7splitrandom0.7test",
# 			"gnn_mdi_v2train0.7splitrandom0.7traintest"
# 			]
# method_names = ["knn","knn test","mean","mean test",
# 				"mice","mice test","svd", "svd test",
# 				"spectral","spectral test","gain","gain train",
# 				"gnn train","gnn test","gnn train test",]
# table_name = 'train0.7splitrandom0.7'

methods = ["knn_v2train0.7","mean_v2train0.7","mice_v2train0.7","svd_v2train0.7","spectral_v2train0.7","gain_v2train0.7","miwae_v2train0.7","gnn_mdi_v2train0.7"]
method_names = ["knn","mean","mice","svd","spectral","gain","miwae","gnn"]
table_name = 'v2train0.7_2'

datasets = ["concrete","energy","housing","kin8nm","naval","power",
			"protein","wine","yacht"]
seeds = [0,1,2,3,4]

with open("{}/tables/mae_{}.txt".format(pre_path,table_name), "w") as text_file:
	text_file.write(' & ')
	for i,dataset in enumerate(datasets):
		if i == len(datasets)-1:
			text_file.write(dataset+' \\\\'+'\n')
		else:
		    text_file.write(dataset+' & ')
	for method,method_name in zip(methods,method_names):
		text_file.write(method_name+' & ')
		for i,dataset in enumerate(datasets):
			result = []
			for seed in seeds:
				load_path = '{}/results/{}/{}/{}/'.format(pre_path,method, dataset, seed)
				obj = joblib.load(load_path+'result.pkl')
				if method.startswith('gnn_mdi'):
				# if method.startswith('gnn') and not method.startswith('gnn_mdi'):
					result.append(obj['curves']['test_l1'][-1])
				elif method.startswith('gain'):
					result.append(obj['mdi_mae'])
					# result.append(obj['reg_mae'])
				else:
					result.append(obj['mae'])
			mean = np.mean(result)
			std = np.std(result)
			# if i == len(datasets)-1:
			# 	text_file.write("${:.3g} \\pm {:.1g} $\\\\ \n".format(mean,std))
			# else:
			# 	text_file.write("${:.3g} \\pm {:.1g} $& ".format(mean,std))
			if i == len(datasets)-1:
				text_file.write("${:.3g}$\\\\ \n".format(mean))
			else:
				text_file.write("${:.3g}$& ".format(mean))


