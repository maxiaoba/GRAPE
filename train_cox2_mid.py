import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

from models import GNNStack
from utils import build_optimizer

def train(dataset, args):

    # build model
    model = GNNStack(dataset.num_node_features, args.hidden_dim, args.embed_dim, 
                            args)
    scheduler, opt = build_optimizer(args, model.parameters())

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # train
    train_loss = []
    train_valid = []
    for epoch in range(args.epochs):
        total_loss = 0
        total_valid = 0
        model.train()
        # for data in dataset: #this is bad, still need dataloader
        # if True:
        #     data = dataset[0]
        for data in loader:
            train_mask = torch.FloatTensor(data.edge_attr.shape[0], 1).uniform_() < 0.8
            # train_mask = torch.FloatTensor(data.edge_attr.shape[0], 1).uniform_() < 1.1
            valid_mask = ~train_mask
            
            edge_attr = data.edge_attr.clone().detach()
            edge_attr = edge_attr[:,0].view(-1,1)
            # edge_attr = edge_attr/torch.max(edge_attr)

            known_edge_attr = edge_attr.clone().detach()
            known_edge_attr[valid_mask] = 0.

            x = data.x.clone().detach()
            edge_index = data.edge_index.clone().detach()

            opt.zero_grad()
            pred = model(x, known_edge_attr, edge_index)
            label = edge_attr

            pred_valid = pred[valid_mask]
            label_valid = label[valid_mask]
            vald_score = model.loss(pred_valid, label_valid)
            total_valid += vald_score.item()

            pred_train = pred[train_mask]
            label_train = label[train_mask]
            loss = model.loss(pred_train, label_train)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        # total_loss /= len(dataset)
        train_loss.append(total_loss)
        train_valid.append(total_valid)
        print('epoch: ',epoch)
        print('loss: ',total_loss)
        print('valid: ',total_valid)

    import matplotlib.pyplot as plt
    plt.figure()
    plot1, = plt.plot(train_loss)
    plot2, =plt.plot(train_valid)
    plt.legend([plot1,plot2],['trian','valid'])
    plt.title('train valid curve')
    plt.figure()
    plot1, = plt.plot(pred_train.detach().numpy())
    plot2, = plt.plot(label_train.detach().numpy())
    plt.legend([plot1,plot2],['pred','label'])
    plt.title('final train result')
    plt.figure()
    plot1, = plt.plot(pred_valid.detach().numpy())    
    plot2, = plt.plot(label_valid.detach().numpy())
    plt.legend([plot1,plot2],['pred','label'])
    plt.title('final valid result')
    plt.show()
  
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def main():
  for args in [
      {'model_type': 'EGCN', 'num_layers': 3, 'batch_size': 32, 'hidden_dim': 1, 'embed_dim': 6, 'dropout': 0., 'epochs': 5000, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 0., 'lr': 0.001},
  ]:
    args = objectview(args)
    dataset = TUDataset(root='/tmp/COX2_MD', name='COX2_MD', use_node_attr=True, use_edge_attr=True)
    dataset = dataset.shuffle()   # add this line!
    train(dataset, args) 

if __name__ == '__main__':
    main()

