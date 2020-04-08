import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import get_activation

class MLPNet(torch.nn.Module):
    def __init__(self, 
         		input_dims, output_dim,
         		hidden_layer_sizes=(64,),
         		hidden_activation='relu',
         		output_activation=None,
                dropout=0.):
        super(MLPNet, self).__init__()

        layers = nn.ModuleList()

        input_dim = np.sum(input_dims)

        for layer_size in hidden_layer_sizes:
        	hidden_dim = layer_size
        	layer = nn.Sequential(
        				nn.Linear(input_dim, hidden_dim),
        				get_activation(hidden_activation),
        				nn.Dropout(dropout),
        				)
        	layers.append(layer)
        	input_dim = hidden_dim

        layer = nn.Sequential(
        				nn.Linear(input_dim, output_dim),
        				get_activation(output_activation),
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




