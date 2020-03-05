import time
import argparse
import os

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import joblib
import pickle

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

from utils import build_optimizer, objectview, get_known_mask, mask_edge
from uci import get_data

from gnn_model import GNNStack
from prediction_model import MLPNet
from plot_utils import plot_result

def train(data, args, log_path):
    # build model
    model = GNNStack(data.num_node_features, args.node_dim,
                            args.edge_dim, args.edge_mode,
                            args.model_types, args.dropout)
    predict_model = MLPNet([args.node_dim], 1,
                            hidden_layer_sizes=args.predict_hiddens, 
                            dropout=args.dropout)
    trainable_parameters = list(model.parameters())+list(predict_model.parameters())

    if args.mode == 1:
        load_file = './Data/uci/'+args.uci_data+'/'+args.load_dir+'/model.pt'
        print('loading model param from ',load_file)
        model.load_state_dict(torch.load(load_file))
        model.eval()
        trainable_parameters = predict_model.parameters()
    if args.mode == 3:
        edge_predict_model = MLPNet([args.node_dim, args.node_dim], 1, dropout=args.dropout)
    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)

    # train
    Train_loss = []
    Valid_mse = []
    Valid_l1 = []

    best_test_l1 = np.inf
    best_epoch = None

    x = data.x.clone().detach()
    y = data.y.clone().detach()
    train_edge_index = data.train_edge_index.clone().detach()
    train_edge_attr = data.train_edge_attr.clone().detach()
    train_y_mask = data.train_y_mask.clone().detach()
    test_y_mask = data.test_y_mask.clone().detach()
    for epoch in range(args.epochs):

        if scheduler is not None:
            scheduler.step(epoch)
        # for param_group in opt.param_groups:
        #     print('lr',param_group['lr'])

        predict_model.train()
        model.train()
        if args.mode == 1:
            model.eval()

        known_mask = get_known_mask(args.known,int(train_edge_attr.shape[0]/2))
        # now concat all masks by it self
        double_known_mask = torch.cat((known_mask, known_mask),dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index,train_edge_attr,double_known_mask,True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        pred = predict_model(x_embd)[:y.shape[0],0]
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]

        loss = F.mse_loss(pred_train, label_train)
        if args.mode == 3:
            edge_pred = edge_predict_model([x_embd[train_edge_index[0],:],x_embd[train_edge_index[1],:]])
            edge_pred_train = edge_pred[:int(train_edge_attr.shape[0]/2)]
            edge_label_train = train_edge_attr[:int(train_edge_attr.shape[0]/2)]
            edge_loss = F.mse_loss(edge_pred_train, edge_label_train)
            loss += edge_loss

        loss.backward()
        opt.step()
        train_loss = loss.item()

        model.eval()
        predict_model.eval()

        x_embd = model(x, train_edge_attr, train_edge_index)
        pred = predict_model(x_embd)[:y.shape[0],0]

        pred_test = pred[test_y_mask]
        label_test = y[test_y_mask]
        mse = F.mse_loss(pred_test, label_test, 'mse')
        test_mse = mse.item()
        l1 = F.l1_loss(pred_test, label_test, 'l1')
        test_l1 = l1.item()

        Train_loss.append(train_loss)
        Valid_mse.append(test_mse)
        Valid_l1.append(test_l1)
        print('epoch: ',epoch)
        print('loss: ',train_loss)
        if args.mode == 3:
            print("edge_loss: ",edge_loss.item())
        print('test mse: ',test_mse)
        print('test l1: ',test_l1)

        if test_l1 < best_test_l1:
            if best_epoch is not None:
                os.remove(log_path+'model_best_ep'+str(best_epoch)+'l1_'+f"{best_test_l1:.5f}"+'.pt')
                os.remove(log_path+'predict_model_best_ep'+str(best_epoch)+'l1_'+f"{best_test_l1:.5f}"+'.pt')
            best_epoch = epoch
            best_test_l1 = test_l1
            torch.save(model.state_dict(), log_path+'model_best_ep'+str(best_epoch)+'l1_'+f"{best_test_l1:.5f}"+'.pt')
            torch.save(predict_model.state_dict(), log_path+'predict_model_best_ep'+str(best_epoch)+'l1_'+f"{best_test_l1:.5f}"+'.pt')
        if args.save_gap != 0:
            if epoch % args.save_gap == 0:
                torch.save(model.state_dict(), log_path+'model_ep'+str(epoch)+'l1_'+f"{test_l1:.5f}"+'.pt')
                torch.save(predict_model.state_dict(), log_path+'predict_model_ep'+str(epoch)+'l1_'+f"{test_l1:.5f}"+'.pt')

    pred_train = pred_train.detach().numpy()
    label_train = label_train.detach().numpy()
    pred_test = pred_test.detach().numpy()
    label_test = label_test.detach().numpy()

    obj = dict()
    obj['args'] = args
    obj['train_loss'] = Train_loss
    obj['test_mse'] = Valid_mse
    obj['test_l1'] = Valid_l1
    obj['pred_train'] = pred_train
    obj['label_train'] = label_train
    obj['pred_test'] = pred_test
    obj['label_test'] = label_test
    pickle.dump(obj, open(log_path+'result.pkl', "wb" ))

    torch.save(model.state_dict(), log_path+'model.pt')
    torch.save(predict_model.state_dict(), log_path+'predict_model.pt')

    obj = objectview(obj)
    plot_result(obj, log_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uci_data', type=str, default='housing') # 'pks', 'cancer', 'housing', 'wine'
    parser.add_argument('--mode', type=int, default=0) 
    # 0: train gnn+mlp; 1: train mlp, load gnn; 3: train gnn+mlp, +mdi loss
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--edge_mode', type=int, default=1) # 0: use it as weight 1: as input to mlp
    parser.add_argument('--predict_hiddens', type=str, default='')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_edge', type=float, default=0.7)
    parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--known', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='y0')
    parser.add_argument('--load_dir', type=str, default='0')
    parser.add_argument('--save_gap', type=int, default=0) # 0: not save by gap
    args = parser.parse_args()
    args.model_types = args.model_types.split('_')
    if args.predict_hiddens == '':
        args.predict_hiddens = []
    else:
        args.predict_hiddens = list(map(int,args.predict_hiddens.split('_')))

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    df_X = pd.read_csv('./Data/uci/'+ args.uci_data +"/"+ args.uci_data +'.csv')
    df_y = pd.read_csv('./Data/uci/'+ args.uci_data +"/"+ args.uci_data +'_target.csv', header=None)
    data = get_data(df_X, df_y, args.train_edge, args.train_y, args.seed)

    log_path = './Data/uci/'+args.uci_data+'/'+args.log_dir+'/'
    os.mkdir(log_path)

    train(data, args, log_path) 

if __name__ == '__main__':
    main()

