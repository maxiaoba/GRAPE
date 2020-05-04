import os.path as osp
import numpy as np
import joblib

pre_path = "./uci/mdi_results"
datasets = ["concrete","energy","housing","kin8nm","naval","power",
			"protein","wine","yacht"]
# methods = ["knn","mean","mice","svd","gnn_mdi"]
methods = ["knn","mean","mice","svd","spectral"]
# method_names = ["knn","mean","mice","svd","gnn"]
method_names = ["knn","mean","mice","svd","spectral"]
comments = ['v2lv0','v2lv1','v2lv2']
seeds = [0,1,2,3,4]

with open("{}/tables/compare_mae.txt".format(pre_path), "w") as text_file:
	text_file.write(' & ')
	for i,dataset in enumerate(datasets):
		if i == len(datasets)-1:
			text_file.write(dataset+' \\\\'+'\n')
		else:
		    text_file.write(dataset+' & ')
	for method,method_name in zip(methods,method_names):
		text_file.write(method_name+' & ')
		for i,dataset in enumerate(datasets):
			results = []
			for comment in comments:
				result = []
				for seed in seeds:
					load_path = './uci/mdi_results/results/{}_{}/{}/{}/'.format(method, comment, dataset, seed)
					obj = joblib.load(load_path+'result.pkl')
					if method.startswith('gnn'):
						result.append(obj['curves']['test_l1'][-1])
					else:
						result.append(obj['mae'])
				mean = np.mean(result)
				std = np.std(result)
				results.append(mean)
			best_version = comments[np.argmin(results)]
			if i == len(datasets)-1:
				text_file.write("{}\\\\ \n".format(best_version))
			else:
				text_file.write("{}& ".format(best_version))


