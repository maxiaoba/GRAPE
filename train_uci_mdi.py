import time
import argparse
import os

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

from utils import build_optimizer, objectview

def train(dataset, args, log_path):
    # build model
    from gnn_model import GNNStack
    model = GNNStack(dataset[0].num_node_features, args.node_dim,
                            args.edge_dim, args.edge_mode,
                            args.model_types, args.dropout)
    from prediction_model import MLPNet
    predict_model = MLPNet([args.node_dim, args.node_dim], 1, dropout=args.dropout)

    if args.mode == 'load':
        print('loading model param')
        model.load_state_dict(torch.load(log_path+'model.pt'))
        predict_model.load_state_dict(torch.load(log_path+'predict_model.pt'))
    # build optimizer
    scheduler, opt = build_optimizer(args, model.parameters())
    # build data loader
    # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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
    best_valid_l1 = np.inf
    best_epoch = None
    for epoch in range(args.epochs):
        train_loss = 0.
        valid_mse = 0.
        valid_l1 = 0.

        if scheduler is not None:
            scheduler.step(epoch)
        # for param_group in opt.param_groups:
        #     print('lr',param_group['lr'])
        for data in dataset: #loader:
            #print(data)
            model.train()
            predict_model.train()
            if (not mask_defined) or (args.fix_train_mask == 0):
                from utils import get_train_mask
                train_mask = \
                    get_train_mask(args.valid,(args.load_train_mask==1),log_path+'../',data)
            mask_defined = True
            
            known_mask = train_mask.clone().detach()
            known_mask[train_mask] = (torch.FloatTensor(torch.sum(train_mask).item()).uniform_() < args.known)
            # known mask is a mask that masks train mask

            # now concat all masks by it self
            double_train_mask = torch.cat((train_mask, train_mask),dim=0)
            double_known_mask = torch.cat((known_mask, known_mask),dim=0)

            x = data.x.clone().detach()
            edge_attr = data.edge_attr.clone().detach()
            edge_index = data.edge_index.clone().detach()
            from utils import mask_edge
            train_edge_index, train_edge_attr = mask_edge(edge_index,edge_attr,double_train_mask,(args.remove_unknown_edge == 1))
            known_edge_index, known_edge_attr = mask_edge(edge_index,edge_attr,double_known_mask,(args.remove_unknown_edge == 1))

            opt.zero_grad()
            x_embd = model(x, known_edge_attr, known_edge_index)
            predict_edge_index = edge_index[:,:int(edge_index.shape[1]/2)]
            pred = predict_model([x_embd[predict_edge_index[0],:],x_embd[predict_edge_index[1],:]])
            label = edge_attr[:int(edge_attr.shape[0]/2)]

            pred_train = pred[train_mask]
            label_train = label[train_mask]
            loss = F.mse_loss(pred_train, label_train)
            loss.backward()
            opt.step()
            train_loss += loss.item()

            model.eval()
            predict_model.eval()
            x_embd = model(x, train_edge_attr, train_edge_index)
            pred = predict_model([x_embd[predict_edge_index[0],:],x_embd[predict_edge_index[1],:]])

            pred_valid = pred[~train_mask]
            label_valid = label[~train_mask]
            mse = F.mse_loss(pred_valid, label_valid, 'mse')
            valid_mse += mse.item()
            l1 = F.l1_loss(pred_valid, label_valid, 'l1')
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

        if valid_l1 < best_valid_l1:
            if best_epoch is not None:
                os.remove(log_path+'best_ep'+str(best_epoch)+'l1'+f"{best_valid_l1:.5f}"+'.pt')
            best_epoch = epoch
            best_valid_l1 = valid_l1
            torch.save(model.state_dict(), log_path+'best_ep'+str(best_epoch)+'l1'+f"{best_valid_l1:.5f}"+'.pt')
        if args.save_gap != 0:
            if epoch % args.save_gap == 0:
                torch.save(model.state_dict(), log_path+'ep'+str(epoch)+'l1'+f"{valid_l1:.5f}"+'.pt')

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
    torch.save(predict_model.state_dict(), log_path+'predict_model.pt')

    from plot_utils import plot_result
    obj = objectview(obj)
    plot_result(obj, log_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uci_data', type=str, default='cancer') # 'pks', 'cancer', 'housing', 'wine'
    parser.add_argument('--mode', type=str, default='new')
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--edge_mode', type=int, default=1) # 0: use it as weight 1: as input to mlp
    parser.add_argument('--batch_size', type=int, default=32) # doesn't matter here
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--valid', type=float, default=0.3)
    parser.add_argument('--known', type=float, default=0.7)
    parser.add_argument('--fix_train_mask', type=int, default=1)  # 1: yes, 0: no
    parser.add_argument('--load_train_mask', type=int, default=1)  # 1: yes, 0: no
    parser.add_argument('--remove_unknown_edge', type=int, default=1)  # 1: yes, 0: no
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='0')
    parser.add_argument('--data', type=str, default="uci")
    parser.add_argument('--save_gap', type=int, default=0) # 0: not save by gap
    args = parser.parse_args()
    args.model_types = args.model_types.split('_')

    if args.mode == 'load':
        import joblib
        load_path = './Data/uci/'+args.uci_data+'/'+args.log_dir+'/'
        result = joblib.load(load_path+'result.pkl')
        result = objectview(result)
        args = result.args
        args.mode = 'load'

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    from uci import get_dataset
    dataset = get_dataset(args.uci_data)

    log_path = './Data/uci/'+args.uci_data+'/'+args.log_dir+'/'
    if args.mode == 'new':
        os.mkdir(log_path)
    train(dataset, args, log_path) 

if __name__ == '__main__':
    main()

