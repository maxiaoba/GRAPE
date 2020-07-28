from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import pickle
import joblib
import time
import os.path
from os import path

from training.baseline import baseline_inpute
from utils.utils import construct_missing_X_from_mask

def linear_regression(data, args, log_path, load_path):
    t0 = time.time()
    n_row, n_col = data.df_X.shape
    x = data.x.clone().detach()
    edge_index = data.edge_index.clone().detach()
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

    if args.method == 'gnn':
        model = torch.load(load_path+'model.pt',map_location=torch.device('cpu'))
        model.eval()
        impute_model = torch.load(load_path+'impute_model.pt',map_location=torch.device('cpu'))
        impute_model.eval()
        predict_model = torch.load(load_path+'predict_model.pt',map_location=torch.device('cpu'))
        predict_model.eval()
        t_load = time.time()

        with torch.no_grad():
            x_embd = model(x, train_edge_attr, train_edge_index)
            X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
            t_impute = time.time()

            X = torch.reshape(X, [n_row, n_col])
            y_pred = predict_model(X)[:, 0]
            y_pred_test = y_pred[test_y_mask].detach().numpy()
            t_reg = time.time()
    else:
        if args.method == 'gnn_mdi':
            model = torch.load(load_path+'model.pt',map_location=torch.device('cpu'))
            model.eval()
            impute_model = torch.load(load_path+'impute_model.pt',map_location=torch.device('cpu'))
            impute_model.eval()
            t_load = time.time()

            with torch.no_grad():
                x_embd = model(x, train_edge_attr, train_edge_index)
                x_pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
                t_impute = time.time()

                x_pred = x_pred[:int(test_edge_attr.shape[0] / 2)]
                X_true, X_incomplete = construct_missing_X_from_mask(train_edge_mask, data.df_X)
                X = X_incomplete
                for i in range(int(test_edge_attr.shape[0] / 2)):
                    assert X_true[test_edge_index[0, i], test_edge_index[1, i] - y.shape[0]] == test_edge_attr[i]
                    X[test_edge_index[0, i], test_edge_index[1, i] - y.shape[0]] = x_pred[i]
        else:
            X_true, X_incomplete = construct_missing_X_from_mask(train_edge_mask, data.df_X)
            t_load = time.time()

            X = baseline_inpute(X_incomplete, args.method, args.level)
            t_impute = time.time()

        reg = LinearRegression().fit(X[train_y_mask, :], y_train)
        y_pred_test = reg.predict(X[test_y_mask, :])
        t_reg = time.time()

    rmse = np.sqrt(np.mean((y_pred_test - y_test) ** 2))
    mae = np.mean(np.abs(y_pred_test - y_test))
    t_test = time.time()

    if path.exists(log_path + 'result.pkl'):
        obj = joblib.load(log_path + 'result.pkl')
        obj['args_linear_regression'] = args
    else:
        obj = dict()
        obj['args'] = args
    obj['load_path'] = load_path
    obj['rmse'] = rmse
    obj['mae'] = mae
    obj['load_time'] = t_load - t0
    obj['impute_time'] = t_impute - t_load
    obj['reg_time'] = t_reg - t_impute
    obj['test_time'] = t_test - t_reg
    print('{}: rmse: {:.3g}, mae: {:.3g}'.format(args.method,rmse,mae))
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))
