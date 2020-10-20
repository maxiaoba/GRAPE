GRAPE -- Handling Missing Data with Graph Representation Learning
===============================================================================

About
-----

GRAPE is a framework for __feature imputation__ as well as __label prediction__. GRAPE tackles the missing data problem using a __graph representation__, where the observations and features are viewed as two types of nodes in a bipartite graph, and the observed feature values as edges. Under the GRAPE framework, the __feature imputation__ is formulated as an __edge-level prediction__ task and the __label prediction__ as a __node-level prediction__ task. These tasks are then solved with Graph Neural Networks.

Installation
------------
At the root folder:
```console
conda env create -f environment.yml
conda activate grape
```

Install [PyTorch](https://pytorch.org/)

Install [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Usages
------

To train feature imputation on uci datasets:
```console
python train_mdi.py uci --data dataset_name
```

To train label prediction on uci datasets:
```console
python train_y.py uci --data dataset_name
```

To train feature imputation on Flixster, Douban and YahooMusic:
```console
python train_mdi.py mc --data flixster/douban/yahoo_music
```

The results will be saved in "uci/test/dataset_name" or "mc/test/dataset_name". For more training options, look at the arguments in "train_mdi.py" and "train_y.py" as well as "uci/uci_subparser.py" and "mc/mc_subparser.py".
