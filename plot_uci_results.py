import os.path as osp
import numpy as np
import joblib
import matplotlib.pyplot as plt

pre_path = "./uci/mdi_results"
datasets = ["concrete","energy","housing","kin8nm","naval","power",
			"protein","wine","yacht"]
methods = ["knn","mean","mice","svd","spectral","gain","gnn_mdi"]
method_names = ["knn","mean","mice","svd","spectral","gain","gnn"]
comments = ['v2train0.3','v2train0.5','v2train0.7','v2train0.9']
xs = [0.3,0.5,0.7,0.9]
xlabel = 'known ratio'
ylabel = 'test mae'
plot_name = 'known_ratio'
seeds = [0,1,2,3,4]


for dataset in datasets:
	plt.figure()
	for method,method_name in zip(methods,method_names):
		ys = []
		for comment in comments:
			result = []
			for seed in seeds:
				load_path = './uci/mdi_results/results/{}_{}/{}/{}/'.format(method, comment, dataset, seed)
				obj = joblib.load(load_path+'result.pkl')
				if method.startswith('gnn'):
					result.append(obj['curves']['test_l1'][-1])
				elif method == 'gain':
					result.append(obj['mdi_mae'])
				else:
					result.append(obj['mae'])
			mean = np.mean(result)
			std = np.std(result)
			ys.append(mean)
		plt.plot(xs, ys, label=method_name)
	plt.legend()
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(dataset)
	plt.savefig("{}/plots/{}_{}.png".format(pre_path,plot_name,dataset))

