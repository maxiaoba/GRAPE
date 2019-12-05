import time
import argparse
import os

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

from models import GNNStack, EGNNStack
from utils import build_optimizer, objectview

def train(dataset, args):
    log_path = './Data/uci/'+args.log_dir+'/'
    os.mkdir(log_path)

    # build model
    if args.gnn_type == 'GNN':
        model = GNNStack(dataset.num_node_features, args.hidden_dim, args.embed_dim, 
                                args)
    elif args.gnn_type == 'EGNN':
        model = EGNNStack(dataset.num_node_features, args.hidden_dim, args.embed_dim, 
                                args)
    scheduler, opt = build_optimizer(args, model.parameters())

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # train
    Train_loss = []
    Valid_mse = []
    Valid_l1 = []

    mask_defined = False
    for epoch in range(args.epochs):
        train_loss = 0.
        valid_mse = 0.
        valid_l1 = 0.
        for data in loader:
            model.train()
            if (not mask_defined) or (args.fix_train_mask == 0):
                if args.load_train_mask == 1:
                    print('loading train validation mask')
                    train_mask = np.load(args.train_mask_dir)
                    train_mask = torch.BoolTensor(train_mask).view(-1)
                else:
                    print('defining train validation mask')
                    train_mask = (torch.FloatTensor(data.edge_attr.shape[0], 1).uniform_() < (1-args.valid)).view(-1)
                valid_mask = ~train_mask
                mask_defined = True

            known_mask = train_mask.clone().detach()
            known_mask[train_mask] = (torch.FloatTensor(torch.sum(train_mask).item()).uniform_() < args.known)
            # known mask is a mask that masks train mask
            
            x = torch.FloatTensor(np.copy(data.x))
            edge_attr = data.edge_attr.clone().detach()
            edge_index = torch.tensor(np.copy(data.edge_index),dtype=int)


            if args.remove_unknown_edge == 1:
                known_edge_index = edge_index[:,known_mask]
                known_edge_attr = edge_attr[known_mask]
                train_edge_index = edge_index[:,train_mask]
                train_edge_attr = edge_attr[train_mask]
            else:
                train_edge_index = edge_index
                train_edge_attr = edge_attr.clone().detach()
                train_edge_attr[valid_mask] = 0.
                known_edge_index = edge_index
                known_edge_attr = edge_attr.clone().detach()
                known_edge_attr[~known_mask] = 0.


            opt.zero_grad()
            pred = model(x, known_edge_attr, known_edge_index, edge_index)
            label = edge_attr

            pred_train = pred[train_mask]
            label_train = label[train_mask]
            loss = model.loss(pred_train, label_train)
            loss.backward()
            opt.step()
            train_loss += loss.item()

            model.eval()
            pred = model(x, train_edge_attr, train_edge_index, edge_index)
            pred_valid = pred[valid_mask]
            label_valid = label[valid_mask]
            mse = model.metric(pred_valid, label_valid, 'mse')
            valid_mse += mse.item()
            l1 = model.metric(pred_valid, label_valid, 'l1')
            valid_l1 += l1.item()

        train_loss /= len(dataset)

        Train_loss.append(train_loss)
        Valid_mse.append(valid_mse)
        Valid_l1.append(valid_l1)
        print('epoch: ',epoch)
        print('loss: ',train_loss)
        print('valid mse: ',valid_mse)
        print('valid l1: ',valid_l1)

    pred_train = pred_train.detach().numpy()
    label_train = label_train.detach().numpy()
    pred_valid = pred_valid.detach().numpy()
    label_valid = label_valid.detach().numpy()

    import pickle
    obj = dict()
    obj['args'] = args
    obj['train_loss'] = Train_loss
    obj['valid_mse'] = Valid_mse
    obj['valid_l1'] = Valid_l1
    obj['pred_train'] = pred_train
    obj['label_train'] = label_train
    obj['pred_valid'] = pred_valid
    obj['label_valid'] = label_valid
    pickle.dump(obj, open(log_path+'result.pkl', "wb" ))

    torch.save(model.state_dict(), log_path+'model.pt')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(Train_loss,linewidth=1.)
    plt.title('train mse')
    plt.subplot(3,1,2)
    plt.plot(Valid_mse,linewidth=1.)
    plt.title('valid mse')
    plt.subplot(3,1,3)
    plt.plot(Valid_l1,linewidth=1.)
    plt.title('valid mae')
    plt.savefig(log_path+'curve.png')
    plt.close()

    plt.figure()
    plot1, = plt.plot(pred_train[::100],linewidth=1.)
    plot2, = plt.plot(label_train[::100],linewidth=1.)
    plt.legend([plot1,plot2],['pred','label'])
    plt.title('final train result')
    plt.savefig(log_path+'final_train.png')
    plt.close()
    plt.figure()
    plot1, = plt.plot(pred_valid[::25],linewidth=1.)    
    plot2, = plt.plot(label_valid[::25],linewidth=1.)
    plt.legend([plot1,plot2],['pred','label'])
    plt.title('final valid result')
    plt.savefig(log_path+'final_valid.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_type', type=str, default='GNN')
    parser.add_argument('--model_types', type=str, default='EGCN_EGCN_EGCN')
    parser.add_argument('--hidden_dim', type=int, default=6)
    parser.add_argument('--embed_dim', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32) # doesn't matter here
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=0)
    parser.add_argument('--opt_decay_rate', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--valid', type=float, default=0.3)
    parser.add_argument('--known', type=float, default=1.1)
    parser.add_argument('--fix_train_mask', type=int, default=1) # 1: yes, 0: no
    parser.add_argument('--load_train_mask', type=int, default=1)
    parser.add_argument('--train_mask_dir', type=str, default='./Data/uci/len6336rate0.7seed0.npy')
    parser.add_argument('--remove_unknown_edge', type=int, default=1)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='1')
    args = parser.parse_args()
    args.model_types = args.model_types.split('_')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    
    from uci import get_dataset, UCIDataset
    dataset = UCIDataset(root='/tmp/UCI')
    # dataset = dataset.shuffle()   # add this line!
    train(dataset, args) 

if __name__ == '__main__':
    main()

