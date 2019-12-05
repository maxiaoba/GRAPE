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

from utils import build_optimizer, objectview

def train(dataset, args, log_path):
    # build model
    if args.gnn_type == 'gnn':
        from models2 import GNNStack
        model = GNNStack(dataset.num_node_features, args.node_dim,
                                args.edge_dim, args.edge_mode,
                                args.predict_mode,
                                (args.update_edge==1),
                                args)
    elif args.gnn_type == 'gnn_split':
        from models3 import GNNStackSplit
        model = GNNStackSplit(dataset.num_node_features, args.node_dim,
                                args.edge_dim, args.edge_mode,
                                args.predict_mode,
                                (args.update_edge==1),
                                dataset[0].num_obj,
                                dataset[0].num_att,
                                args)
        print(dataset[0].num_obj,dataset[0].num_att)

    if args.mode == 'load':
        print('loading model param')
        model.load_state_dict(torch.load(log_path+'model.pt'))
    # build optimizer
    scheduler, opt = build_optimizer(args, model.parameters())
    # build data loader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # train
    if args.mode == 'new':
        Train_loss = []
        Valid_mse = []
        Valid_l1 = []
    elif args.mode == 'load':
        import joblib
        result = joblib.load(log_path+'result.pkl')
        result = objectview(result)
        Train_loss = result.train_loss
        Valid_mse = result.valid_mse
        Valid_l1 = result.valid_l1

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
                    train_mask = (torch.FloatTensor(int(data.edge_attr.shape[0]/2), 1).uniform_() < (1-args.valid)).view(-1)
                    #print(data.edge_attr.shape[0])
                #print(len(train_mask))
                valid_mask = ~train_mask
                mask_defined = True

            known_mask = train_mask.clone().detach()
            known_mask[train_mask] = (torch.FloatTensor(torch.sum(train_mask).item()).uniform_() < args.known)
            # known mask is a mask that masks train mask

            # now concat all masks by it self
            double_train_mask = torch.cat((train_mask, train_mask),dim=0)
            double_valid_mask = torch.cat((valid_mask, valid_mask),dim=0)
            double_known_mask = torch.cat((known_mask, known_mask),dim=0)
            
            x = torch.FloatTensor(np.copy(data.x))
            edge_attr = data.edge_attr.clone().detach()
            edge_index = torch.tensor(np.copy(data.edge_index),dtype=int)

            if args.remove_unknown_edge == 1:
                known_edge_index = edge_index[:,double_known_mask]
                known_edge_attr = edge_attr[double_known_mask]
                train_edge_index = edge_index[:,double_train_mask]
                train_edge_attr = edge_attr[double_train_mask]
            else:
                train_edge_index = edge_index
                train_edge_attr = edge_attr.clone().detach()
                train_edge_attr[double_valid_mask] = 0.
                known_edge_index = edge_index
                known_edge_attr = edge_attr.clone().detach()
                known_edge_attr[~double_known_mask] = 0.


            opt.zero_grad()
            pred = model(x, known_edge_attr, known_edge_index, edge_index[:,:int(edge_index.shape[1]/2)])
            label = edge_attr[:int(edge_attr.shape[0]/2)]

            pred_train = pred[train_mask]
            label_train = label[train_mask]
            loss = model.loss(pred_train, label_train)
            loss.backward()
            opt.step()
            train_loss += loss.item()

            model.eval()
            pred = model(x, train_edge_attr, train_edge_index, edge_index[:,:int(edge_index.shape[1]/2)])
            pred_valid = pred[valid_mask]
            label_valid = label[valid_mask]
            mse = model.metric(pred_valid, label_valid, 'mse')
            valid_mse += mse.item()
            l1 = model.metric(pred_valid, label_valid, 'l1')
            valid_l1 += l1.item()

        train_loss /= len(dataset)
        valid_mse /= len(dataset)
        valid_l1 /= len(dataset)

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

    from plot_utils import plot_result
    obj = objectview(obj)
    plot_result(obj, log_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='new')
    parser.add_argument('--gnn_type', type=str, default='GNN')
    parser.add_argument('--model_types', type=str, default='EGCN_EGCN_EGCN')
    parser.add_argument('--node_dim', type=int, default=6)
    parser.add_argument('--edge_dim', type=int, default=6)
    parser.add_argument('--edge_mode', type=int, default=1) # 0: use it as weight 1: as input to mlp
    parser.add_argument('--predict_mode', type=int, default=0) # 0: use node embd, 1: use edge embd 
    parser.add_argument('--update_edge', type=int, default=1)  # 1: yes, 0: no
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
    parser.add_argument('--known', type=float, default=0.7)
    parser.add_argument('--fix_train_mask', type=int, default=1)  # 1: yes, 0: no
    parser.add_argument('--load_train_mask', type=int, default=1)  # 1: yes, 0: no
    parser.add_argument('--train_mask_dir', type=str, default='./Data/uci/len6336rate0.7seed0.npy')
    parser.add_argument('--remove_unknown_edge', type=int, default=1)  # 1: yes, 0: no
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='1')
    parser.add_argument('--data', type=str, default="uci")
    args = parser.parse_args()
    args.model_types = args.model_types.split('_')

    if args.mode == 'load':
        import joblib
        load_path = './Data/uci/'+args.log_dir+'/'
        result = joblib.load(load_path+'result.pkl')
        result = objectview(result)
        args = result.args
        args.mode = 'load'

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    from uci import get_dataset, UCIDataset, SimDataset
    if args.data == "uci":
        dataset = UCIDataset(root='/tmp/UCI')
        #print(dataset.is_undirected())
    elif args.data == "simulated":
        dataset = SimDataset(root='/tmp/SIM')
    # dataset = dataset.shuffle()

    log_path = './Data/uci/'+args.log_dir+'/'
    if args.mode == 'new':
        os.mkdir(log_path)
    train(dataset, args, log_path) 

if __name__ == '__main__':
    main()

