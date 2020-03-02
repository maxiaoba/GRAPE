from fancyimpute import SimpleFill, KNN, IterativeImputer, IterativeSVD
import numpy as np
import pandas as pd
from sklearn import preprocessing

def construct_missing_X(train_mask, df):
    nrow, ncol = df.shape
    data_incomplete = np.zeros((nrow, ncol))
    data_complete = np.zeros((nrow, ncol)) 
    train_mask = train_mask.reshape(nrow, ncol)
    for i in range(nrow):
        for j in range(ncol):
            data_complete[i,j] = df.iloc[i,j]
            if train_mask[i,j]:
                data_incomplete[i,j] = df.iloc[i,j]
            else:
                data_incomplete[i,j] = np.NaN
    return data_complete, data_incomplete

def baseline_inpute(data,method='mean'):
    train_mask = data.train_edge_mask.numpy()
    X, X_incomplete = construct_missing_X(train_mask, data.df_X)
    n_missing = len(train_mask) - sum(train_mask)
    
    if method == 'mean':
        X_filled_mean = SimpleFill().fit_transform(X_incomplete)
        MAE_mean = sum(sum(abs(X_filled_mean - X))) / n_missing
        return X_filled_mean, MAE_mean
    elif method == 'knn':
        X_filled_knn = KNN(k=3, verbose=False).fit_transform(X_incomplete)
        diff_knn = X - X_filled_knn
        MAE_knn = sum(sum(abs(diff_knn))) / n_missing
        return X_filled_knn, MAE_knn
    elif method == 'svd':
        X_filled_svd = IterativeSVD(rank=9,verbose=False).fit_transform(X_incomplete)
        MAE_svd = sum(sum(abs(X_filled_svd - X))) / n_missing
        return X_filled_svd, MAE_svd
    elif method == 'mice':
        X_filled_mice = IterativeImputer().fit_transform(X_incomplete)
        MAE_mice = sum(sum(abs(X_filled_mice - X))) / n_missing
        return X_filled_mice, MAE_mice
    else:
        raise NotImplementedError