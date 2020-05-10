import os.path as osp
import numpy as np
import joblib

# pre_path = "./uci/mdi_results"
pre_path = "./uci/y_results"
datasets = ["concrete","energy","housing","kin8nm","naval","power",
			"protein","wine","yacht"]
# methods = ["knn","mean","mice","svd","spectral","gain","gnn_mdi"]
methods = ["knn","mean","mice","svd","spectral","gain","gnn_mdi","gnn"]
# method_names = ["knn","mean","mice","svd","spectral","gain","gnn"]
method_names = ["knn","mean","mice","svd","spectral","gain","gnn","gnn n2n"]
comment = 'v2train0.3'
seeds = [0,1,2,3,4]

with open("{}/tables/mae_{}.txt".format(pre_path,comment), "w") as text_file:
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
				load_path = '{}/results/{}_{}/{}/{}/'.format(pre_path,method, comment, dataset, seed)
				obj = joblib.load(load_path+'result.pkl')
				# if method == 'gnn_mdi':
				if method == 'gnn':
					result.append(obj['curves']['test_l1'][-1])
				elif method == 'gain':
					# result.append(obj['mdi_mae'])
					result.append(obj['reg_mae'])
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


