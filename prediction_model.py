import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_activateion

class MLPNet(torch.nn.Module):
    def __init__(self, 
         		input_dims, output_dim,
         		hidden_layer_sizes=(64,),
         		hidden_activation='relu',
         		output_activation=None,
                args):
        super(EdgePredictor, self).__init__()

        self.predict_mode = predict_mode
        self.hidden_activation = get_activation(hidden_activation)
        self.output_activation = get_activation(output_activation)
        layers = nn.ModuleList()

        input_dim = np.sum(input_dims)

        for layer_size in hidden_layer_sizes:
        	hidden_dim = layer_size
        	layer = nn.Sequential(
        				nn.Linear(input_dim, hidden_dim),
        				self.hidden_activation,
        				nn.Dropout(args.dropout),
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




