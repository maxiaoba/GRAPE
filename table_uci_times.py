import os.path as osp
import numpy as np
import joblib

pre_path = "./uci/y_results"

methods = ["mean","knn","mice","svd","spectral","gain","gnn"]
method_names = ["Mean","KNN","MICE","SVD","Spectral","GAIN","GRAPE"]
comment = '_v2train0.7'
metric = 'reg_time'
table_name = 'v2train0.7'

datasets = ["concrete","energy","housing","kin8nm","naval","power",
			"protein","wine","yacht"]
seeds = [0,1,2,3,4]

with open("{}/tables/{}_{}.txt".format(pre_path,metric,table_name), "w") as text_file:
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
				load_path = '{}/results/{}{}/{}/{}/'.format(pre_path,method,comment,dataset,seed)
				obj = joblib.load(load_path+'result.pkl')
				result.append(obj[metric])
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


