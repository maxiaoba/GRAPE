import numpy as np
import torch
import torch.nn.functional as F
import pickle

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge

def train_gnn_mdi(data, args, log_path, device=torch.device('cpu')):
    model = get_gnn(data, args).to(device)
    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int,args.impute_hiddens.split('_')))
    if args.concat_states:
        input_dim = args.node_dim * len(model.convs) * 2
    else:
        input_dim = args.node_dim * 2
    if hasattr(args,'ce_loss') and args.ce_loss:
        output_dim = len(data.class_values)
    else:
        output_dim = 1
    impute_model = MLPNet(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)
    if args.transfer_dir: # this ensures the valid mask is consistant
        load_path = './{}/test/{}/{}/'.format(args.domain,args.data,args.transfer_dir)
        print("loading fron {} with {}".format(load_path,args.transfer_extra))
        model = torch.load(load_path+'model'+args.transfer_extra+'.pt',map_location=device)
        impute_model = torch.load(load_path+'impute_model'+args.transfer_extra+'.pt',map_location=device)

    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters())
    print("total trainable_parameters: ",len(trainable_parameters))
    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)

    # train
    Train_loss = []
    Test_rmse = []
    Test_l1 = []
    Lr = []

    x = data.x.clone().detach().to(device)
    if hasattr(args,'split_sample') and args.split_sample > 0.:
        if args.split_train:
            all_train_edge_index = data.lower_train_edge_index.clone().detach().to(device)
            all_train_edge_attr = data.lower_train_edge_attr.clone().detach().to(device)
            all_train_labels = data.lower_train_labels.clone().detach().to(device)
        else:
            all_train_edge_index = data.train_edge_index.clone().detach().to(device)
            all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
            all_train_labels = data.train_labels.clone().detach().to(device)
        if args.split_test:
            test_input_edge_index = data.higher_train_edge_index.clone().detach().to(device)
            test_input_edge_attr = data.higher_train_edge_attr.clone().detach().to(device)
        else:
            test_input_edge_index = data.train_edge_index.clone().detach().to(device)
            test_input_edge_attr = data.train_edge_attr.clone().detach().to(device)
        test_edge_index = data.higher_test_edge_index.clone().detach().to(device)
        test_edge_attr = data.higher_test_edge_attr.clone().detach().to(device)
        test_labels = data.higher_test_labels.clone().detach().to(device)
    else:
        all_train_edge_index = data.train_edge_index.clone().detach().to(device)
        all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
        all_train_labels = data.train_labels.clone().detach().to(device)
        test_input_edge_index = all_train_edge_index
        test_input_edge_attr = all_train_edge_attr
        test_edge_index = data.test_edge_index.clone().detach().to(device)
        test_edge_attr = data.test_edge_attr.clone().detach().to(device)
        test_labels = data.test_labels.clone().detach().to(device)
    if hasattr(data,'class_values'):
        class_values = data.class_values.clone().detach().to(device)
    if args.valid > 0.:
        valid_mask = get_known_mask(args.valid, int(all_train_edge_attr.shape[0] / 2)).to(device)
        print("valid mask sum: ",torch.sum(valid_mask))
        train_labels = all_train_labels[~valid_mask]
        valid_labels = all_train_labels[valid_mask]
        double_valid_mask = torch.cat((valid_mask, valid_mask), dim=0)
        valid_edge_index, valid_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, double_valid_mask, True)
        train_edge_index, train_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, ~double_valid_mask, True)
        print("train edge num is {}, valid edge num is {}, test edge num is input {} output {}"\
                .format(
                train_edge_attr.shape[0], valid_edge_attr.shape[0],
                test_input_edge_attr.shape[0], test_edge_attr.shape[0]))
        Valid_rmse = []
        Valid_l1 = []
        best_valid_rmse = np.inf
        best_valid_rmse_epoch = 0
        best_valid_l1 = np.inf
        best_valid_l1_epoch = 0
    else:
        train_edge_index, train_edge_attr, train_labels =\
             all_train_edge_index, all_train_edge_attr, all_train_labels
        print("train edge num is {}, test edge num is input {}, output {}"\
                .format(
                train_edge_attr.shape[0],
                test_input_edge_attr.shape[0], test_edge_attr.shape[0]))
    if args.auto_known:
        args.known = float(all_train_labels.shape[0])/float(all_train_labels.shape[0]+test_labels.shape[0])
        print("auto calculating known is {}/{} = {:.3g}".format(all_train_labels.shape[0],all_train_labels.shape[0]+test_labels.shape[0],args.known))
    obj = dict()
    obj['args'] = args
    obj['outputs'] = dict()
    for epoch in range(args.epochs):
        model.train()
        impute_model.train()
        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
        if hasattr(args,'ce_loss') and args.ce_loss:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2)]
        else:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2),0]
        if args.loss_mode == 1:
            pred_train[known_mask] = train_labels[known_mask]
        label_train = train_labels

        if hasattr(args,'ce_loss') and args.ce_loss:
            loss = F.cross_entropy(pred_train,train_labels)
        else:
            loss = F.mse_loss(pred_train, label_train)
        loss.backward()
        opt.step()
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])

        model.eval()
        impute_model.eval()
        with torch.no_grad():
            if args.valid > 0.:
                x_embd = model(x, train_edge_attr, train_edge_index)
                pred = impute_model([x_embd[valid_edge_index[0], :], x_embd[valid_edge_index[1], :]])
                if hasattr(args,'ce_loss') and args.ce_loss:
                    pred_valid = class_values[pred[:int(valid_edge_attr.shape[0] / 2)].max(1)[1]]
                    label_valid = class_values[valid_labels]
                elif hasattr(args,'norm_label') and args.norm_label:
                    pred_valid = pred[:int(valid_edge_attr.shape[0] / 2),0]
                    pred_valid = pred_valid * max(class_values)
                    label_valid = valid_labels
                    label_valid = label_valid * max(class_values)
                else:
                    pred_valid = pred[:int(valid_edge_attr.shape[0] / 2),0]
                    label_valid = valid_labels
                mse = F.mse_loss(pred_valid, label_valid)
                valid_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_valid, label_valid)
                valid_l1 = l1.item()
                if valid_l1 < best_valid_l1:
                    best_valid_l1 = valid_l1
                    best_valid_l1_epoch = epoch
                    if args.save_model:
                        torch.save(model, log_path + 'model_best_valid_l1.pt')
                        torch.save(impute_model, log_path + 'impute_model_best_valid_l1.pt')
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_valid_rmse_epoch = epoch
                    if args.save_model:
                        torch.save(model, log_path + 'model_best_valid_rmse.pt')
                        torch.save(impute_model, log_path + 'impute_model_best_valid_rmse.pt')
                Valid_rmse.append(valid_rmse)
                Valid_l1.append(valid_l1)

            x_embd = model(x, test_input_edge_attr, test_input_edge_index)
            pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
            if hasattr(args,'ce_loss') and args.ce_loss:
                pred_test = class_values[pred[:int(test_edge_attr.shape[0] / 2)].max(1)[1]]
                label_test = class_values[test_labels]
            elif hasattr(args,'norm_label') and args.norm_label:
                pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
                pred_test = pred_test * max(class_values)
                label_test = test_labels
                label_test = label_test * max(class_values)
            else:
                pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
                label_test = test_labels
            mse = F.mse_loss(pred_test, label_test)
            test_rmse = np.sqrt(mse.item())
            l1 = F.l1_loss(pred_test, label_test)
            test_l1 = l1.item()
            if args.save_prediction:
                if epoch == best_valid_rmse_epoch:
                    obj['outputs']['best_valid_rmse_pred_test'] = pred_test.detach().cpu().numpy()
                if epoch == best_valid_l1_epoch:
                    obj['outputs']['best_valid_l1_pred_test'] = pred_test.detach().cpu().numpy()

            if args.mode == 'debug':
                torch.save(model, log_path + 'model_{}.pt'.format(epoch))
                torch.save(impute_model, log_path + 'impute_model_{}.pt'.format(epoch))
            Train_loss.append(train_loss)
            Test_rmse.append(test_rmse)
            Test_l1.append(test_l1)
            print('epoch: ', epoch)
            print('loss: ', train_loss)
            if args.valid > 0.:
                print('valid rmse: ', valid_rmse)
                print('valid l1: ', valid_l1)
            print('test rmse: ', test_rmse)
            print('test l1: ', test_l1)

    pred_train = pred_train.detach().cpu().numpy()
    label_train = label_train.detach().cpu().numpy()
    pred_test = pred_test.detach().cpu().numpy()
    label_test = label_test.detach().cpu().numpy()

    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss
    if args.valid > 0.:
        obj['curves']['valid_rmse'] = Valid_rmse
        obj['curves']['valid_l1'] = Valid_l1
    obj['curves']['test_rmse'] = Test_rmse
    obj['curves']['test_l1'] = Test_l1
    obj['lr'] = Lr

    obj['outputs']['final_pred_train'] = pred_train
    obj['outputs']['label_train'] = label_train
    obj['outputs']['final_pred_test'] = pred_test
    obj['outputs']['label_test'] = label_test
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    if args.save_model:
        torch.save(model, log_path + 'model.pt')
        torch.save(impute_model, log_path + 'impute_model.pt')

    # obj = objectview(obj)
    plot_curve(obj['curves'], log_path+'curves.png',keys=None, 
                clip=True, label_min=True, label_end=True)
    plot_curve(obj, log_path+'lr.png',keys=['lr'], 
                clip=False, label_min=False, label_end=False)
    plot_sample(obj['outputs'], log_path+'outputs.png', 
                groups=[['final_pred_train','label_train'],
                        ['final_pred_test','label_test']
                        ], 
                num_points=20)
    if args.save_prediction and args.valid > 0.:
        plot_sample(obj['outputs'], log_path+'outputs_best_valid.png', 
                    groups=[['best_valid_rmse_pred_test','label_test'],
                            ['best_valid_l1_pred_test','label_test']
                            ], 
                    num_points=20)
    if args.valid > 0.:
        print("best valid rmse is {:.3g} at epoch {}".format(best_valid_rmse,best_valid_rmse_epoch))
        print("best valid l1 is {:.3g} at epoch {}".format(best_valid_l1,best_valid_l1_epoch))
