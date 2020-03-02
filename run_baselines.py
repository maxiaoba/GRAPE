from fancyimpute import SimpleFill, KNN, IterativeImputer, IterativeSVD
import numpy as np
import pandas as pd
from sklearn import preprocessing

from uci import get_dataset

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

def impute_baselines(df, train_mask):
    X, X_incomplete = construct_missing_X(train_mask, df)
    n_missing = len(train_mask) - sum(train_mask)
    
    X_filled_mean = SimpleFill().fit_transform(X_incomplete)
    MAE_mean = sum(sum(abs(X_filled_mean - X))) / n_missing
    
    X_filled_knn = KNN(k=3, verbose=False).fit_transform(X_incomplete)
    diff_knn = X - X_filled_knn
    MAE_knn = sum(sum(abs(diff_knn))) / n_missing
    
    X_filled_svd = IterativeSVD(rank=9,verbose=False).fit_transform(X_incomplete)
    MAE_svd = sum(sum(abs(X_filled_svd - X))) / n_missing
    
    X_filled_mice = IterativeImputer().fit_transform(X_incomplete)
    MAE_mice = sum(sum(abs(X_filled_mice - X))) / n_missing
    
    print("Mean absolute error (MAE) for recosntructing X using baseline imputation methods")
    print("Mean: "+str(MAE_mean))
    print("KNN: "+str(MAE_knn))
    print("SVD: "+str(MAE_svd))
    print("MICE: "+str(MAE_mice)) 

    #return the imputed X matrices for downstream analysis
    return (X_filled_mean, X_filled_knn, X_filled_svd, X_filled_mice)

df_X = pd.read_csv('./Data/uci/housing/housing.csv')
df_y = pd.read_csv('./Data/uci/housing/housing_target.csv', header=None)
dataset = get_dataset(df_X, df_y, 0.7, 0.7, 0)
train_mask = dataset[0].train_edge_mask.numpy()

#X, X_incomplete = construct_missing_X(missing, df)
X_mean, X_knn, X_svd, X_mice = impute_baselines(df_X, train_mask)
print(X_mice)