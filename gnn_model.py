import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from egcn import EGCNConv
from egsage import EGraphSage

class GNNStack(torch.nn.Module):
    def __init__(self, 
                node_input_dim, node_dim,
                edge_dim, edge_mode,
                model_types, dropout, pre_linear_dim):
        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.model_types = model_types
        self.gnn_layer_num = len(model_types)

        # pre-linear
        self.pre_linear = nn.Linear(node_input_dim, pre_linear_dim)

        # convs
        self.convs = self.build_convs(pre_linear_dim, node_dim, edge_dim, edge_mode, model_types)

        # post node update
        self.node_post_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(node_dim, node_dim)
            )

        self.edge_update_mlps = self.build_edge_update_mlps(node_dim, edge_dim, self.gnn_layer_num)

    def build_convs(self, node_input_dim, node_dim, edge_dim, edge_mode, model_types):
        convs = nn.ModuleList()
        conv = self.build_conv_model(model_types[0],node_input_dim,node_dim,
                                    1, edge_mode)
        convs.append(conv)
        for l in range(1,len(model_types)):
            conv = self.build_conv_model(model_types[l],node_dim, node_dim,
                                    edge_dim, edge_mode)
            convs.append(conv)
        return convs

    def build_conv_model(self, model_type, node_in_dim, node_out_dim, edge_dim, edge_mode):
        #print(model_type)
        if model_type == 'GCN':
            return pyg_nn.GCNConv(node_in_dim,node_out_dim)
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv(node_in_dim,node_out_dim)
        elif model_type == 'GAT':
            return pyg_nn.GATConv(node_in_dim,node_out_dim)
        elif model_type == 'EGCN':
            return EGCNConv(node_in_dim,node_out_dim,edge_dim,edge_mode)
        elif model_type == 'EGSAGE':
            return EGraphSage(node_in_dim,node_out_dim,edge_dim,edge_mode)

    def build_edge_update_mlps(self, node_dim, edge_dim, gnn_layer_num):
        edge_update_mlps = nn.ModuleList()
        edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+1,edge_dim),
                nn.ReLU(),
                )
        edge_update_mlps.append(edge_update_mlp)
        for l in range(1,gnn_layer_num):
            edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+edge_dim,edge_dim),
                nn.ReLU(),
                )
            edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = mlp(torch.cat((x_i,x_j,edge_attr),dim=-1))
        edge_attr = F.relu(edge_attr)
        edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)
        return edge_attr

    def forward(self, x, edge_attr, edge_index):
        x = self.pre_linear(x)
        
        for l,(conv_name,conv) in enumerate(zip(self.model_types,self.convs)):
            # self.check_input(x,edge_attr,edge_index)
            if conv_name == 'EGCN' or conv_name == 'EGSAGE':
                x = conv(x, edge_attr, edge_index)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
            #print(edge_attr.shape)
        x = self.node_post_mlp(x)
        # self.check_input(x,edge_attr,edge_index)
        return x

    def check_input(self, xs, edge_attr, edge_index):
        Os = {}
        for indx in range(128):
            i=edge_index[0,indx].detach().numpy()
            j=edge_index[1,indx].detach().numpy()
            xi=xs[i].detach().numpy()
            xj=list(xs[j].detach().numpy())
            eij=list(edge_attr[indx].detach().numpy())
            if str(i) not in Os.keys():
                Os[str(i)] = {'x_j':[],'e_ij':[]}
            Os[str(i)]['x_i'] = xi
            Os[str(i)]['x_j'] += xj
            Os[str(i)]['e_ij'] += eij

        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1,3,1)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_i'],label=str(i))
            plt.title('x_i')
        plt.legend()
        plt.subplot(1,3,2)
        for i in Os.keys():
            plt.plot(Os[str(i)]['e_ij'],label=str(i))
            plt.title('e_ij')
        plt.legend()
        plt.subplot(1,3,3)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_j'],label=str(i))
            plt.title('x_j')
        plt.legend()
        plt.show()

