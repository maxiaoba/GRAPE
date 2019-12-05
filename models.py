import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from egcn import EGCNConv

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GNNStack, self).__init__()
        self.build_convs(input_dim, hidden_dim, output_dim, args)

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        # edge prediction
        self.edge_mlps = nn.ModuleList()
        self.edge_mlps.append(nn.Linear(2*output_dim,output_dim))
        self.edge_mlps.append(nn.Linear(output_dim,output_dim))
        self.edge_mlps.append(nn.Linear(output_dim,output_dim))
        self.edge_mlps.append(nn.Linear(output_dim,1))

        self.dropout = args.dropout
        self.model_types = args.model_types

    def build_convs(self, input_dim, hidden_dim, output_dim, args):
        self.convs = nn.ModuleList()
        conv = self.build_conv_model(args.model_types[0],input_dim,hidden_dim,1)
        self.convs.append(conv)
        for l in range(1,len(args.model_types)):
            conv = self.build_conv_model(args.model_types[l],hidden_dim, hidden_dim,1)
            self.convs.append(conv)

    def build_conv_model(self, model_type, in_dim, out_dim, edge_dim):
        print(model_type)
        if model_type == 'GCN':
            return pyg_nn.GCNConv(in_dim,out_dim)
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv(in_dim,out_dim)
        elif model_type == 'GAT':
            return pyg_nn.GATConv(in_dim,out_dim)
        elif model_type == 'EGCN':
            return EGCNConv(in_dim,out_dim,edge_dim)

    def forward(self, x, edge_attr, edge_index, predict_edge_index):
        for conv_name,conv in zip(self.model_types,self.convs):
            if conv_name == 'EGCN':
                x = conv(x, edge_attr, edge_index)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.post_mp(x)
        x = self.predict_edge(x, predict_edge_index)

        return x

    def predict_edge(self, x, edge_index):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        x = torch.cat((x_i,x_j),dim=-1)
        for edge_mlp in self.edge_mlps:
            x = edge_mlp(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def loss(self, pred, label):
        return F.mse_loss(pred, label)

    def metric(self, pred, label, metric='mse'):
        if metric == 'mse':
            return F.mse_loss(pred, label)
        elif metric == 'l1':
            return F.l1_loss(pred, label)

class EGNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(EGNNStack, self).__init__()
        self.convs = nn.ModuleList()
        conv = self.build_conv_model(args.model_types[0],input_dim,hidden_dim,1)
        self.convs.append(conv)
        self.edge_mlps = nn.ModuleList()
        edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim+hidden_dim+1,hidden_dim),
                nn.ReLU(),
                )
        self.edge_mlps.append(edge_mlp)
        for l in range(1,len(args.model_types)):
            conv = self.build_conv_model(args.model_types[l],hidden_dim, hidden_dim, hidden_dim)
            self.convs.append(conv)
            edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim+hidden_dim+hidden_dim,hidden_dim),
                nn.ReLU(),
                )
            self.edge_mlps.append(edge_mlp)

        # edge prediction
        self.edge_mlps2 = nn.ModuleList()
        self.edge_mlps2.append(nn.Linear(2*output_dim,output_dim))
        self.edge_mlps2.append(nn.Linear(output_dim,output_dim))
        self.edge_mlps2.append(nn.Linear(output_dim,output_dim))
        self.edge_mlps2.append(nn.Linear(output_dim,1))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, 1))

        self.dropout = args.dropout
        self.model_types = args.model_types

    def build_conv_model(self, model_type, in_dim, out_dim, edge_dim):
        print(model_type)
        if model_type == 'GCN':
            return pyg_nn.GCNConv(in_dim,out_dim)
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv(in_dim,out_dim)
        elif model_type == 'GAT':
            return pyg_nn.GATConv(in_dim,out_dim)
        elif model_type == 'EGCN':
            return EGCNConv(in_dim,out_dim,edge_dim)

    def forward(self, x, edge_attr, edge_index, predict_edge_index):
        for conv_name,conv,edge_mlp in zip(self.model_types,self.convs,self.edge_mlps):
            if conv_name == 'EGCN':
                x = conv(x, edge_attr, edge_index)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_i = x[edge_index[0],:]
            x_j = x[edge_index[1],:]
            edge_attr = edge_mlp(torch.cat((x_i,x_j,edge_attr),dim=-1))
            edge_attr = F.relu(edge_attr)
            edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

        # edge_attr = self.post_mp(edge_attr)
        y = self.predict_edge(x, predict_edge_index)
        return y

    def predict_edge(self, x, edge_index):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        x = torch.cat((x_i,x_j),dim=-1)
        for edge_mlp in self.edge_mlps2:
            x = edge_mlp(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def loss(self, pred, label):
        return F.mse_loss(pred, label)

    def metric(self, pred, label, metric='mse'):
        if metric == 'mse':
            return F.mse_loss(pred, label)
        elif metric == 'l1':
            return F.l1_loss(pred, label)

