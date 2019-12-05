from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch

from uci import get_dataset, UCIDataset
dataset = UCIDataset(root='/tmp/UCI')
# dataset = dataset.shuffle()

# the batch size doesn't matter since there is only one graph
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
	print(batch)
	print(batch.batch)
	print(torch.max(batch.edge_index[0,:]))


