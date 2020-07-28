import os
import pickle
import numpy as np

known_rates = [0.3,0.5,0.7,0.9]
datasets = ["concrete","energy","housing","kin8nm","naval","power",
			"protein","wine","yacht"]
seeds = [0,1,2,3,4]


for known_rate in known_rates:
	file = open("./tree_{}.txt".format(known_rate),"r")
	file.readline()
	for dataset in datasets:
		maes = list(map(float,file.readline().split(',')[1:]))
		for i,seed in enumerate(seeds):
			mae = maes[i]
			# don't have orig data, do this for the ease of plot
			log_path = "tree_v2train{}/{}/{}/".format(known_rate,dataset,seed)
			if not os.path.isdir(log_path):
				os.makedirs(log_path)
			obj = dict()
			obj['mae'] = mae
			pickle.dump(obj, open(log_path + 'result.pkl', "wb"))




