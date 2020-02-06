import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_activation

class MLPNet(torch.nn.Module):
    def __init__(self, 
         		input_dims, output_dim,
         		hidden_layer_sizes=(64,),
         		hidden_activation='relu',
         		output_activation=None,
                dropout=0.):
        super(MLPNet, self).__init__()

        self.hidden_activation = get_activation(hidden_activation)
        self.output_activation = get_activation(output_activation)
        layers = nn.ModuleList()

        input_dim = np.sum(input_dims)

        for layer_size in hidden_layer_sizes:
        	hidden_dim = layer_size
        	layer = nn.Sequential(
        				nn.Linear(input_dim, hidden_dim),
        				self.hidden_activation,
        				nn.Dropout(dropout),
        				)
        	layers.append(layer)
        	input_dim = hidden_dim

        layer = nn.Sequential(
        				nn.Linear(input_dim, output_dim),
        				self.output_activation,
        				)
       	layers.append(layer)
       	self.layers = layers

    def forward(self, inputs):
    	if torch.is_tensor(inputs):
    		inputs = [inputs]
    	input_var = torch.cat(inputs,-1)
    	for layer in self.layers:
    		input_var = layer(input_var)
    	return input_var

    # def predict_edge(self, x, edge_attr, edge_index):
    #     if self.predict_mode == 0:
    #         x_i = x[edge_index[0],:]
    #         x_j = x[edge_index[1],:]
    #         x = torch.cat((x_i,x_j),dim=-1)
    #     else:
    #         assert edge_attr.shape[0] == edge_index.shape[1]
    #         x = edge_attr
    #     y = self.edge_predict_mlp(x)
    #     return y

    # def loss(self, pred, label):
    #     return F.mse_loss(pred, label)

    # def metric(self, pred, label, metric='mse'):
    #     if metric == 'mse':
    #         return F.mse_loss(pred, label)
    #     elif metric == 'l1':
    #         return F.l1_loss(pred, label)




