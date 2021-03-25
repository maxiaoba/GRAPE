GRAPE -- Handling Missing Data with Graph Representation Learning
===============================================================================

About
-----
This repository is the official PyTorch implementation of "Handling Missing Data with Graph Representation Learning". [**[Project webpage](http://snap.stanford.edu/grape/)**].

[Jiaxuan You*](https://cs.stanford.edu/~jiaxuan/), Xiaobai Ma\*, Daisy Ding*, [Mykel Kochenderfer](https://mykel.kochenderfer.com/), [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html), Handling Missing Data with Graph Representation Learning, NeurIPS 2020.

GRAPE is a framework for __feature imputation__ as well as __label prediction__. GRAPE tackles the missing data problem using a __graph representation__, where the observations and features are viewed as two types of nodes in a bipartite graph, and the observed feature values as edges. Under the GRAPE framework, the __feature imputation__ is formulated as an __edge-level prediction__ task and the __label prediction__ as a __node-level prediction__ task. These tasks are then solved with Graph Neural Networks.

Installation
------------
At the root folder:
```bash
conda env create -f environment.yml
conda activate grape
```

Install [PyTorch](https://pytorch.org/)

Install [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Usages
------

To train feature imputation on uci datasets:
```bash
# UCI: concrete, energy, housing, kin8nm, naval, power, protein, wine, yacht
python train_mdi.py uci --data concrete
```

To train label prediction on uci datasets:
```bash
# UCI: concrete, energy, housing, kin8nm, naval, power, protein, wine, yacht
python train_y.py uci --data concrete
```

To train feature imputation on Flixster, Douban and YahooMusic:
```bash
# flixster, douban, yahoo_music
python train_mdi.py mc --data flixster
```

The results will be saved in "uci/test/dataset_name" or "mc/test/dataset_name". For more training options, look at the arguments in "train_mdi.py" and "train_y.py" as well as "uci/uci_subparser.py" and "mc/mc_subparser.py".

## Citation
If you find this work useful, please cite our paper:
```latex
@article{you2020handling,
  title={Handling Missing Data with Graph Representation Learning},
  author={You, Jiaxuan and Ma, Xiaobai and Ding, Daisy and Kochenderfer, Mykel and Leskovec, Jure},
  journal={NeurIPS},
  year={2020}
}
