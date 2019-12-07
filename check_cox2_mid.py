from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch

dataset = TUDataset(root='/tmp/COX2_MD', name='COX2_MD', use_node_attr=True, use_edge_attr=True)
dataset = dataset.shuffle()

print(dataset[0])
dataset[0].edge_attr = dataset[0].edge_attr[:,0] # not working
print(dataset[0].edge_index)
# print(dataset[0].edge_attr)
# print(dataset.__dict__.keys())
# print(dataset.num_node_features)
# print(dataset[0].edge_attr.shape[0])
# print(dataset[0].x)

# not sure about its behavior
# loader = DataLoader(dataset, batch_size=2, shuffle=True)
# for batch in loader:
# 	print(batch)
# 	print(batch.batch)
# 	print(torch.max(batch.edge_index[0,:]))

# print(len(dataset))
# for data in dataset:
# 	print(data)
# 	known_mask = torch.FloatTensor(data.edge_attr.shape[0], 1).uniform_() < 0.8
# 	pred_mask = ~known_mask
# 	edge_attr = data.edge_attr
# 	edge_attr = edge_attr[:,0].view(-1,1)

# 	known_edge_attr = edge_attr.clone().detach()
# 	known_edge_attr[pred_mask] = 0.

# 	x = data.x
# 	edge_index = data.edge_index
# 	print(edge_index)
# 	print(x[edge_index[0],:])

