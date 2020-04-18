import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from models.egcn import EGCNConv
from models.egsage import EGraphSage
from utils.utils import get_activation

def get_gnn(data, args):
    model_types = args.model_types.split('_')
    if args.norm_embs is None:
        norm_embs = [True,]*len(model_types)
    else:
        norm_embs = list(map(bool,map(int,args.norm_embs.split('_'))))
    if args.post_hiddens is None:
        post_hiddens = [args.node_dim]
    else:
        post_hiddens = list(map(int,args.post_hiddens.split('_')))
    print(model_types, norm_embs, post_hiddens)
    # build model
    if args.dual_gnn:
        model = GNNDualStack(data.user_num, data.num_node_features, data.edge_attr_dim,
                            args.node_dim, args.edge_dim, args.edge_mode,
                            model_types, args.dropout, args.gnn_activation,
                            args.concat_states,
                            norm_embs)
    else:
        model = GNNStack(data.num_node_features, data.edge_attr_dim,
                            args.node_dim, args.edge_dim, args.edge_mode,
                            model_types, args.dropout, args.gnn_activation,
                            args.concat_states, post_hiddens,
                            norm_embs)
    return model

class GNNStack(torch.nn.Module):
    def __init__(self, 
                node_input_dim, edge_input_dim,
                node_dim, edge_dim, edge_mode,
                model_types, dropout, activation,
                concat_states, node_post_mlp_hiddens,
                normalize_embs,
                ):
        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.concat_states = concat_states
        self.model_types = model_types
        self.gnn_layer_num = len(model_types)

        # convs
        self.convs = self.build_convs(node_input_dim, edge_input_dim,
                                    node_dim, edge_dim, edge_mode,
                                    model_types, normalize_embs, activation)

        # post node update
        if concat_states:
            self.node_post_mlp = self.build_node_post_mlp(int(node_dim*len(model_types)), int(node_dim*len(model_types)), node_post_mlp_hiddens, dropout, activation)
        else:
            self.node_post_mlp = self.build_node_post_mlp(node_dim, node_dim, node_post_mlp_hiddens, dropout, activation)

        self.edge_update_mlps = self.build_edge_update_mlps(node_dim, edge_input_dim, edge_dim, self.gnn_layer_num, activation)

    def build_node_post_mlp(self, input_dim, output_dim, hidden_dims, dropout, activation):
        if 0 in hidden_dims:
            return get_activation('none')
        else:
            layers = []
            for hidden_dim in hidden_dims:
                layer = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            get_activation(activation),
                            nn.Dropout(dropout),
                            )
                layers.append(layer)
                input_dim = hidden_dim
            layer = nn.Linear(input_dim, output_dim)
            layers.append(layer)
            return nn.Sequential(*layers)

    def build_convs(self, node_input_dim, edge_input_dim,
                     node_dim, edge_dim, edge_mode,
                     model_types, normalize_embs, activation):
        convs = nn.ModuleList()
        conv = self.build_conv_model(model_types[0],node_input_dim,node_dim,
                                    edge_input_dim, edge_mode, normalize_embs[0], activation)
        convs.append(conv)
        for l in range(1,len(model_types)):
            conv = self.build_conv_model(model_types[l],node_dim, node_dim,
                                    edge_dim, edge_mode, normalize_embs[l], activation)
            convs.append(conv)
        return convs

    def build_conv_model(self, model_type, node_in_dim, node_out_dim, edge_dim, edge_mode, normalize_emb, activation):
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
            return EGraphSage(node_in_dim,node_out_dim,edge_dim,activation,edge_mode,normalize_emb)

    def build_edge_update_mlps(self, node_dim, edge_input_dim, edge_dim, gnn_layer_num, activation):
        edge_update_mlps = nn.ModuleList()
        edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+edge_input_dim,edge_dim),
                get_activation(activation),
                )
        edge_update_mlps.append(edge_update_mlp)
        for l in range(1,gnn_layer_num):
            edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+edge_dim,edge_dim),
                get_activation(activation),
                )
            edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = mlp(torch.cat((x_i,x_j,edge_attr),dim=-1))
        return edge_attr

    def forward(self, x, edge_attr, edge_index):
        if self.concat_states:
            concat_x = []
        for l,(conv_name,conv) in enumerate(zip(self.model_types,self.convs)):
            # self.check_input(x,edge_attr,edge_index)
            if conv_name == 'EGCN' or conv_name == 'EGSAGE':
                x = conv(x, edge_attr, edge_index)
            else:
                x = conv(x, edge_index)
            if self.concat_states:
                concat_x.append(x)
            edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
            #print(edge_attr.shape)
        if self.concat_states:
            x = torch.cat(concat_x, 1)
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

class GNNDualStack(GNNStack):
    def __init__(self, user_num,
                node_input_dim, edge_input_dim,
                node_dim, edge_dim, edge_mode,
                model_types, dropout, activation,
                concat_states,
                normalize_embs,
                ):
        super(GNNStack, self).__init__()
        self.user_num = user_num

        self.dropout = dropout
        self.activation = activation
        self.concat_states = concat_states
        self.model_types = model_types
        self.gnn_layer_num = len(model_types)

        # convs
        self.convs1 = self.build_convs(node_input_dim, edge_input_dim,
                                    node_dim, edge_dim, edge_mode,
                                    model_types, normalize_embs, activation)
        self.convs2 = self.build_convs(node_input_dim, edge_input_dim,
                                    node_dim, edge_dim, edge_mode,
                                    model_types, normalize_embs, activation)

        self.edge_update_mlps1 = self.build_edge_update_mlps(node_dim, edge_input_dim, edge_dim, self.gnn_layer_num, activation)
        self.edge_update_mlps2 = self.build_edge_update_mlps(node_dim, edge_input_dim, edge_dim, self.gnn_layer_num, activation)

    def forward(self, x, edge_attr, edge_index):
        edge_num = int(edge_attr.shape[0]/2)
        # pay attention to order, edge index is [(u_ind,p_ind),(p_ind,u_ind)]
        # the gnn is source_to_target, so edge_index[0,:] is source
        edge_attr1 = edge_attr[edge_num:]
        edge_index1 = edge_index[:,edge_num:]
        edge_attr2 = edge_attr[:edge_num]
        edge_index2 = edge_index[:,:edge_num]
        # assert torch.all(torch.eq(edge_attr1,edge_attr2))
        # assert torch.all(torch.eq(edge_index1[0,:],edge_index2[1,:]))
        # assert torch.all(torch.eq(edge_index2[1,:],edge_index1[0,:]))
        if self.concat_states:
            concat_x = []
        for l,(conv_name,conv1,conv2) in enumerate(zip(self.model_types,self.convs1,self.convs2)):
            # self.check_input(x,edge_attr1,edge_index1)
            # self.check_input(x,edge_attr2,edge_index2)
            if conv_name == 'EGCN' or conv_name == 'EGSAGE':
                print(edge_index1.shape)
                x1 = conv1(x, edge_attr1, edge_index1)
                x2 = conv2(x, edge_attr2, edge_index2)
            else:
                x1 = conv1(x, edge_index1)
                x2 = conv2(x, edge_index2)
            x = torch.cat((x1[:self.user_num],x2[self.user_num:]),0)
            if self.concat_states:
                concat_x.append(x)
            edge_attr1 = self.update_edge_attr(x, edge_attr1, edge_index1, self.edge_update_mlps1[l])
            edge_attr2 = self.update_edge_attr(x, edge_attr2, edge_index2, self.edge_update_mlps2[l])
            #print(edge_attr.shape)
        if self.concat_states:
            x = torch.cat(concat_x, 1)
        # self.check_input(x,edge_attr,edge_index)
        return x


