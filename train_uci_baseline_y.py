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

from baseline import baseline_inpute

def train(data, args, log_path):
    # build model
    predict_model = MLPNet([data.df_X.shape[1]], 1, 
                            hidden_layer_sizes=args.predict_hiddens,
                            dropout=args.dropout)
    trainable_parameters = predict_model.parameters()

    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)

    # train
    Train_loss = []
    Valid_mse = []
    Valid_l1 = []

    best_test_l1 = np.inf
    best_epoch = None

    x_embd, impute_mae = baseline_inpute(data, args.method)
    x_embd = torch.tensor(x_embd).float()
    y = data.y.clone().detach()
    train_y_mask = data.train_y_mask.clone().detach()
    test_y_mask = data.test_y_mask.clone().detach()
    for epoch in range(args.epochs):

        if scheduler is not None:
            scheduler.step(epoch)

        predict_model.train()
        opt.zero_grad()

        pred = predict_model(x_embd)[:,0]
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]

        loss = F.mse_loss(pred_train, label_train)

        loss.backward()
        opt.step()
        train_loss = loss.item()

        predict_model.eval()

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
        print('test mse: ',test_mse)
        print('test l1: ',test_l1)

        if test_l1 < best_test_l1:
            if best_epoch is not None:
                os.remove(log_path+'predict_model_best_ep'+str(best_epoch)+'l1_'+f"{best_test_l1:.5f}"+'.pt')
            best_epoch = epoch
            best_test_l1 = test_l1
            torch.save(predict_model.state_dict(), log_path+'predict_model_best_ep'+str(best_epoch)+'l1_'+f"{best_test_l1:.5f}"+'.pt')
        if args.save_gap != 0:
            if epoch % args.save_gap == 0:
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

    torch.save(predict_model.state_dict(), log_path+'predict_model.pt')

    obj = objectview(obj)
    plot_result(obj, log_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uci_data', type=str, default='housing') # 'pks', 'cancer', 'housing', 'wine'
    parser.add_argument('--method', type=str, default='mean')
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='y0')
    parser.add_argument('--save_gap', type=int, default=0) # 0: not save by gap
    args = parser.parse_args()
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

    log_path = './Data/uci/'+args.uci_data+'/'+args.method+'_'+args.log_dir+'/'
    os.mkdir(log_path)

    train(data, args, log_path) 

if __name__ == '__main__':
    main()

