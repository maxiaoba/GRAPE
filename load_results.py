import os.path as osp
import numpy as np

pre_path = "./Data/results"
datasets = ["concrete","energy","housing","kin8nm","naval","power",
			"protein","wine","yacht"]
methods = ["gnn_v1","knn_v1","mean_v1","mice_v1","svd_v1"]
seeds = [0,1,2,3,4]

for dataset in datasets:
	print(dataset)
	for method in methods:
		result = []
		for seed in seeds:
			f = open(osp.join(pre_path,method,dataset,str(seed),"results.txt"),"r")
			result_i = f.readline()
			result.append(list(map(float,result_i.split(','))))
		print(method,": ",np.mean(result,0))