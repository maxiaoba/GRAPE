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
    # use mask to split train/validation/test
    test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print('total node: ',dataset[0])
    print('train node: ',torch.sum(dataset[0].train_mask==True).item())
    print('val node: ',torch.sum(dataset[0].val_mask==True).item())
    print('test node: ',torch.sum(dataset[0].test_mask==True).item())

    # build model
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, 
                            args, task=task)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    train_loss = []
    valid_score = []
    test_score = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        # print(total_loss)
        train_loss.append(total_loss)
        print('epoch: ',epoch)
        print('loss: ',total_loss)

        if epoch % 10 == 0:
            # test_acc = test(loader, model)
            # print(test_acc,   '  test')
            valid_acc = test(test_loader, model, True)
            valid_score.append(valid_acc)
            print('valid: ',valid_acc)
            test_acc = test(test_loader, model)
            test_score.append(test_acc)
            print('test: ',test_acc)

    np.save(args.dataset+args.model_type+'train.npy',train_loss)
    np.save(args.dataset+args.model_type+'valid.npy',valid_score)
    np.save(args.dataset+args.model_type+'test.npy',test_score)

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            # total += torch.sum(data.test_mask).item()
            total += torch.sum(data.val_mask if is_validation else data.test_mask).item()
    return correct / total
  
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def main():
  for args in [
      {'model_type': 'EGCN', 'dataset': 'cox', 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0., 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01},
  ]:
    args = objectview(args)
    if args.dataset == 'cox':
        dataset = TUDataset(root='/tmp/COX2_MD', name='COX2_MD')
        dataset = dataset.shuffle()   # add this line!
    train(dataset, args) 

if __name__ == '__main__':
    main()

