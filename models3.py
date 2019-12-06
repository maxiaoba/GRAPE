import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from egcn import EGCNConv
from egsage import EGraphSage

from models2 import GNNStack

class GNNStackSplit(GNNStack):
    # split the graph into two directed graphs
    def __init__(self, 
                node_input_dim, node_dim,
                edge_dim, edge_mode, predict_mode,
                update_edge, 
                args):
        super(GNNStack, self).__init__() # just call the grandparent's initializer
        self.dropout = args.dropout
        self.model_types = args.model_types
        self.gnn_layer_num = len(args.model_types)
        self.update_edge = update_edge

        # convs
        self.obj_convs = self.build_convs(node_input_dim, node_dim, edge_dim, edge_mode, args.model_types)
        self.att_convs = self.build_convs(node_dim, node_dim, edge_dim, edge_mode, args.model_types)
        # use node_dim in att_convs since for input_dim since x is updated by obj_convs first

        # post node update
        self.node_post_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim), 
            nn.ReLU(),
            nn.Dropout(args.dropout), 
            nn.Linear(node_dim, node_dim)
            )

        # edge update
        self.obj_edge_update_mlps = self.build_edge_update_mlps(update_edge, node_dim, edge_dim, self.gnn_layer_num)
        self.att_edge_update_mlps = self.build_edge_update_mlps(update_edge, node_dim, edge_dim, self.gnn_layer_num)

        # edge prediction
        self.predict_mode = predict_mode
        if predict_mode == 0:
            self.edge_predict_mlp = nn.Sequential(
                nn.Linear(2*node_dim, node_dim), 
                nn.ReLU(),
                nn.Dropout(args.dropout), 
                nn.Linear(node_dim, 1)
                )
        elif predict_mode == 1:
            self.edge_predict_mlp = nn.Sequential(
                nn.Linear(edge_dim, edge_dim), 
                nn.ReLU(),
                nn.Dropout(args.dropout), 
                nn.Linear(edge_dim, 1)
                )

    def forward(self, x, edge_attr, edge_index, predict_edge_index, return_x=False):
        for l,(conv_name,obj_conv,att_conv) in \
            enumerate(zip(self.model_types,self.obj_convs,self.att_convs)):
            # self.check_input(x,edge_attr,edge_index)
            # split edge_attr and edge_index into two sets
            obj_edge_attr = edge_attr[int(edge_attr.shape[0]/2):,:]
            att_edge_attr = edge_attr[:int(edge_attr.shape[0]/2),:]
            obj_edge_index = edge_index[:,int(edge_index.shape[1]/2):]
            att_edge_index = edge_index[:,:int(edge_index.shape[1]/2)]
            
            if conv_name == 'EGCN' or conv_name == 'EGSAGE':
                x = obj_conv(x, obj_edge_attr, obj_edge_index)
                x = att_conv(x, att_edge_attr, att_edge_index)
            else:
                x = obj_conv(x, obj_edge_index)
                x = att_conv(x, att_edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.update_edge:
                obj_edge_attr = self.update_edge_attr(x, obj_edge_attr, obj_edge_index, self.obj_edge_update_mlps[l])
                att_edge_attr = self.update_edge_attr(x, att_edge_attr, att_edge_index, self.att_edge_update_mlps[l])
                edge_attr = torch.cat((att_edge_attr,obj_edge_attr),dim=0)
            #print(edge_attr.shape)
        x = self.node_post_mlp(x)
        y = self.predict_edge(x, edge_attr, predict_edge_index)
        # self.check_input(x,edge_attr,edge_index)
        if return_x:
            return x,y
        else:
            return y



