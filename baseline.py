from fancyimpute import SimpleFill, KNN, IterativeImputer, IterativeSVD
import numpy as np
import pandas as pd
from sklearn import preprocessing
from utils import construct_missing_X

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
        X_filled_svd = IterativeSVD(rank=9,verbose=False).fit_transform(X_incomplete)
        return X_filled_svd
    elif method == 'mice':
        X_filled_mice = IterativeImputer().fit_transform(X_incomplete)
        return X_filled_mice
    else:
        raise NotImplementedError