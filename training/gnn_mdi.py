import numpy as np
import torch
import torch.nn.functional as F
import pickle

import sys
sys.path.append("..")
from models.gnn_model import GNNStack
from models.prediction_model import MLPNet
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge

def train_gnn_mdi(data, args, log_path, device=torch.device('cpu')):
    model_types = args.model_types.split('_')
    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int,args.impute_hiddens.split('_')))

    # build model
    model = GNNStack(data.num_node_features, args.node_dim,
                        args.edge_dim, args.edge_mode,
                        model_types, args.dropout).to(device)
    impute_model = MLPNet([args.node_dim, args.node_dim], 1,
                            hidden_layer_sizes=impute_hiddens,
                            dropout=args.dropout).to(device)
    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters())

    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)

    # train
    Train_loss = []
    Test_mse = []
    Test_l1 = []

    x = data.x.clone().detach().to(device)
    all_train_edge_index = data.train_edge_index.clone().detach().to(device)
    all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    test_edge_index = data.test_edge_index.clone().detach().to(device)
    test_edge_attr = data.test_edge_attr.clone().detach().to(device)
    if args.valid > 0.:
        valid_mask = get_known_mask(args.valid, int(all_train_edge_attr.shape[0] / 2)).to(device)
        double_valid_mask = torch.cat((valid_mask, valid_mask), dim=0)
        valid_edge_index, valid_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, double_valid_mask, True)
        train_edge_index, train_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, ~double_valid_mask, True)
        print("train edge num is {}, valid edge num is {}, test edge num is {}"\
                .format(
                train_edge_attr.shape[0],valid_edge_attr.shape[0],
                test_edge_attr.shape[0]))
        Valid_mse = []
        Valid_l1 = []
        best_valid_l1 = np.inf
        best_epoch = 0
    else:
        train_edge_index, train_edge_attr = all_train_edge_index, all_train_edge_attr
        print("train edge num is {}, test edge num is {}"\
                .format(
                train_edge_attr.shape[0],test_edge_attr.shape[0]))

    for epoch in range(args.epochs):
        if scheduler is not None:
            scheduler.step(epoch)
        # for param_group in opt.param_groups:
        #     print('lr',param_group['lr'])

        model.train()
        impute_model.train()

        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
        pred_train = pred[:int(train_edge_attr.shape[0] / 2)]
        label_train = train_edge_attr[:int(train_edge_attr.shape[0] / 2)]

        loss = F.mse_loss(pred_train, label_train)
        loss.backward()
        opt.step()
        train_loss = loss.item()

        model.eval()
        impute_model.eval()
        if args.valid > 0.:
            x_embd = model(x, train_edge_attr, train_edge_index)
            pred = impute_model([x_embd[valid_edge_index[0], :], x_embd[valid_edge_index[1], :]])
            pred_valid = pred[:int(valid_edge_attr.shape[0] / 2)]
            label_valid = valid_edge_attr[:int(valid_edge_attr.shape[0] / 2)]
            mse = F.mse_loss(pred_valid, label_valid)
            valid_mse = mse.item()
            l1 = F.l1_loss(pred_valid, label_valid)
            valid_l1 = l1.item()
            if valid_l1 < best_valid_l1:
                best_valid_l1 = valid_l1
                best_epoch = epoch
                torch.save(model, log_path + 'model_best')
                torch.save(impute_model, log_path + 'impute_model_best')
            Valid_mse.append(valid_mse)
            Valid_l1.append(valid_l1)

        x_embd = model(x, all_train_edge_attr, all_train_edge_index)
        pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
        pred_test = pred[:int(test_edge_attr.shape[0] / 2)]
        label_test = test_edge_attr[:int(test_edge_attr.shape[0] / 2)]
        mse = F.mse_loss(pred_test, label_test)
        test_mse = mse.item()
        l1 = F.l1_loss(pred_test, label_test)
        test_l1 = l1.item()

        Train_loss.append(train_loss)
        Test_mse.append(test_mse)
        Test_l1.append(test_l1)
        print('epoch: ', epoch)
        print('loss: ', train_loss)
        if args.valid > 0.:
            print('valid mse: ', valid_mse)
            print('valid l1: ', valid_l1)
        print('test mse: ', test_mse)
        print('test l1: ', test_l1)

    pred_train = pred_train.detach().numpy()
    label_train = label_train.detach().numpy()
    pred_test = pred_test.detach().numpy()
    label_test = label_test.detach().numpy()

    obj = dict()
    obj['args'] = args
    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss
    if args.valid > 0.:
        obj['curves']['valid_mse'] = Valid_mse
        obj['curves']['valid_l1'] = Valid_l1
    obj['curves']['test_mse'] = Test_mse
    obj['curves']['test_l1'] = Test_l1
    obj['outputs'] = dict()
    obj['outputs']['pred_train'] = pred_train
    obj['outputs']['label_train'] = label_train
    obj['outputs']['pred_test'] = pred_test
    obj['outputs']['label_test'] = label_test
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    torch.save(model, log_path + 'model')
    torch.save(impute_model, log_path + 'impute_model')

    obj = objectview(obj)
    plot_curve(obj.curves, log_path+'curves.png',keys=None, 
                clip=True, label_min=True, label_end=True)
    plot_sample(obj.outputs, log_path+'outputs.png', 
                groups=[['pred_train','label_train'],
                        ['pred_test','label_test']
                        ], 
                num_points=20)
    if args.valid > 0.:
        print("best valid l1 is {:.3g} at epoch {}".format(best_valid_l1,best_epoch))
