from fancyimpute import SimpleFill, KNN, IterativeImputer, IterativeSVD
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle

from utils.utils import construct_missing_X

def baseline_mdi(data, args, log_path):
    train_edge_mask = data.train_edge_mask.numpy()
    X, X_incomplete = construct_missing_X(train_edge_mask, data.df_X)

    X_filled = baseline_inpute(data, args.method,args.level)
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

def baseline_inpute(data,method='mean',level=0):
    train_mask = data.train_edge_mask.numpy()
    X, X_incomplete = construct_missing_X(train_mask, data.df_X)
    
    if method == 'mean':
        X_filled_mean = SimpleFill().fit_transform(X_incomplete)
        return X_filled_mean
    elif method == 'knn':
        k = [3,10,50][level]
        X_filled_knn = KNN(k=k, verbose=False).fit_transform(X_incomplete)
        return X_filled_knn
    elif method == 'svd':
        rank = [np.ceil((X_incomplete.shape[1]-1)/10),np.ceil((X_incomplete.shape[1]-1)/5),X_incomplete.shape[1]-1][level]
        X_filled_svd = IterativeSVD(rank=int(rank),verbose=False).fit_transform(X_incomplete)
        return X_filled_svd
    elif method == 'mice':
        max_iter = [3,10,50][level]
        X_filled_mice = IterativeImputer(max_iter=max_iter).fit_transform(X_incomplete)
        return X_filled_mice
    else:
        raise NotImplementedError