import os.path as osp
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tikzplotlib
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")

pre_path = "./uci/mdi_results"
methods = ["knn","mean","mice","svd","spectral","gain","gnn_mdi"]
method_names = ["KNN","Mean","MICE","SVD","Spectral","GAIN","MD-GNN"]
colors = [(0., 0., 0.5), 'g', 'c', 'm', 'y', 'pink', 'blue']
ylabel = 'MDI Test MAE'
plot_name = 'mdi_known_ratio'

# pre_path = "./uci/y_results"
# methods = ["knn","mean","mice","svd","spectral","gain","gnn"]
# method_names = ["KNN","Mean","MICE","SVD","Spectral","GAIN","MD-GNN"]
# colors = [(0., 0., 0.5), 'g', 'c', 'm', 'y', 'pink', 'blue']
# ylabel = "Downstream Regression Test MAE"
# plot_name = 'reg_known_ratio'

comments = ['v2train0.9','v2train0.7','v2train0.5','v2train0.3']
xs = [0.1,0.3,0.5,0.7]
# datasets = ["concrete","energy","housing","kin8nm","naval","power",
# 			"protein","wine","yacht"]
# datasets = ["concrete"]
datasets = ["protein"]
seeds = [0,1,2,3,4]
xlabel = 'Missing data ratio'
use_ylabel = False
use_legend = True

for dataset in datasets:
	print(dataset)
	plt.figure(figsize=(10,8))
	for method,method_name, color in zip(methods,method_names,colors):
		ys = []
		for comment in comments:
			result = []
			for seed in seeds:
				load_path = '{}/results/{}_{}/{}/{}/'.format(pre_path, method, comment, dataset, seed)
				obj = joblib.load(load_path+'result.pkl')
				if method == 'gnn_mdi':
				# if method == 'gnn':
					result.append(obj['curves']['test_l1'][-1])
				elif method == 'gain':
					result.append(obj['mdi_mae'])
					# result.append(obj['reg_mae'])
				else:
					result.append(obj['mae'])
			mean = np.mean(result)
			std = np.std(result)
			ys.append(mean)
		plt.errorbar(xs, ys, std, label=method_name, color=color)
	if use_legend:
		plt.legend(loc='right', bbox_to_anchor=(1.4, 0.5), ncol=1,prop={'size': 28})
	plt.xlabel(xlabel, fontsize=36)
	if use_ylabel:
		plt.ylabel(ylabel, fontsize=36)
	plt.xticks(xs, fontsize=30)
	plt.yticks(fontsize=30)
	ax = plt.gca()
	ax.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
	ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
	plt.title(dataset, fontsize=36)
	plot_file = "{}/plots/{}_{}".format(pre_path,plot_name,dataset)
	if use_ylabel:
		plot_file = plot_file + "_label"
	if use_legend:
		plot_file = plot_file + '_legend'
	plt.savefig("{}.png".format(plot_file), dpi=150, bbox_inches='tight')
	tikzplotlib.save("{}.tex".format(plot_file))

