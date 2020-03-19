from fancyimpute import SimpleFill, KNN, IterativeImputer, IterativeSVD
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle

from utils.utils import construct_missing_X

def baseline_mdi(data, args, log_path):
    train_edge_mask = data.train_edge_mask.numpy()
    X, X_incomplete = construct_missing_X(train_edge_mask, data.df_X)

    X_filled = baseline_inpute(data, args.method)
    mask = train_edge_mask.reshape(X.shape)
    diff = X[~mask] - X_filled[~mask]
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    obj = dict()
    obj['args'] = args
    obj['rmse'] = rmse
    obj['mae'] = mae
    print('rmse: {:.3g}, mae: {:.3g}'.format(rmse,mae))
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

def baseline_inpute(data,method='mean'):
    train_mask = data.train_edge_mask.numpy()
    X, X_incomplete = construct_missing_X(train_mask, data.df_X)
    
    if method == 'mean':
        X_filled_mean = SimpleFill().fit_transform(X_incomplete)
        return X_filled_mean
    elif method == 'knn':
        X_filled_knn = KNN(k=3, verbose=False).fit_transform(X_incomplete)
        return X_filled_knn
    elif method == 'svd':
        X_filled_svd = IterativeSVD(rank=X_incomplete.shape[1]-1,verbose=False).fit_transform(X_incomplete)
        return X_filled_svd
    elif method == 'mice':
        X_filled_mice = IterativeImputer().fit_transform(X_incomplete)
        return X_filled_mice
    else:
        raise NotImplementedError