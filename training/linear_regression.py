from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import pickle

import sys
sys.path.append("..")
from training.baseline import baseline_inpute
from utils.utils import construct_missing_X

def linear_regression(data, args, log_path, load_path):
    x = data.x.clone().detach()
    train_edge_mask = data.train_edge_mask.numpy()
    train_edge_index = data.train_edge_index.clone().detach()
    train_edge_attr = data.train_edge_attr.clone().detach()
    test_edge_index = data.test_edge_index.clone().detach()
    test_edge_attr = data.test_edge_attr.clone().detach()

    y = data.y.detach().numpy()
    train_y_mask = data.train_y_mask.clone().detach()
    # print(torch.sum(train_y_mask))
    test_y_mask = data.test_y_mask.clone().detach()
    y_train = y[train_y_mask]
    y_test = y[test_y_mask]

    if args.method == 'gnn_mdi':
        model = torch.load(load_path+'model')
        model.eval()
        impute_model = torch.load(load_path+'impute_model')
        impute_model.eval()

        x_embd = model(x, train_edge_attr, train_edge_index)
        # X = x_embd.detach().numpy()[:y.shape[0],:]
        x_pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
        x_pred = x_pred[:int(test_edge_attr.shape[0] / 2)]
        X_true, X_incomplete = construct_missing_X(train_edge_mask, data.df_X)
        X = X_incomplete
        for i in range(int(test_edge_attr.shape[0] / 2)):
            assert X_true[test_edge_index[0, i], test_edge_index[1, i] - y.shape[0]] == test_edge_attr[i]
            X[test_edge_index[0, i], test_edge_index[1, i] - y.shape[0]] = x_pred[i]
    else:
        X = baseline_inpute(data, args.method)

    reg = LinearRegression().fit(X[train_y_mask, :], y_train)
    y_pred_test = reg.predict(X[test_y_mask, :])

    rmse = np.sqrt(np.mean((y_pred_test - y_test) ** 2))
    mae = np.mean(np.abs(y_pred_test - y_test))

    obj = dict()
    obj['args'] = args
    obj['load_path'] = load_path
    obj['rmse'] = rmse
    obj['mae'] = mae
    print('rmse: {:.3g}, mae: {:.3g}'.format(rmse,mae))
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))
