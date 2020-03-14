import os.path as osp
import numpy as np

pre_path = "./Data/mdi_results"
datasets = ["concrete","energy","housing","kin8nm","naval","power",
			"protein","wine","yacht"]
methods = ["knn_v1","mean_v1","mice_v1","svd_v1","gnn_v1"]
method_names = ["knn","mean","mice","svd","gnn"]
seeds = [0,1,2,3,4]

with open("{}/results_mae.txt".format(pre_path), "w") as text_file:
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
				f = open(osp.join(pre_path,method,dataset,str(seed),"results.txt"),"r")
				result_i = f.readline()
				result.append(float(result_i))
			result = np.mean(result,0)
			if i == len(datasets)-1:
				text_file.write("{:.3g} \\\\ \n".format(result))
			else:
				text_file.write("{:.3g} & ".format(result))


