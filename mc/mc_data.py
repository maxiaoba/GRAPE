import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing

import torch
import random
import numpy as np
import math

from utils.utils import get_known_mask, mask_edge, one_hot, soft_one_hot
from mc.preprocessing import *

def create_node(df, mode):
    if mode == 0: # onehot feature node, all 1 sample node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol))
        feature_node[np.arange(ncol), feature_ind] = 1
        sample_node = [[1]*ncol for i in range(nrow)]
        node = sample_node + feature_node.tolist()
    elif mode == 1: # onehot sample and feature node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol+1))
        feature_node[np.arange(ncol), feature_ind+1] = 1
        sample_node = np.zeros((nrow,ncol+1))
        sample_node[:,0] = 1
        node = sample_node.tolist() + feature_node.tolist()
    return node

def get_data(u_features, v_features, node_mode, adj_train,
    train_labels, train_u_indices, train_v_indices,
    val_labels, val_u_indices, val_v_indices, 
    test_labels, test_u_indices, test_v_indices, 
    class_values, one_hot_edge, soft_one_hot_edge, norm_label, ce_loss):
    train_indices = train_labels
    val_indices = val_labels
    test_indices = test_labels
    # indices are the index in class_values
    train_labels = torch.FloatTensor(class_values[train_labels])
    val_labels = torch.FloatTensor(class_values[val_labels])
    test_labels = torch.FloatTensor(class_values[test_labels])
    if norm_label:
        train_labels = train_labels/max(class_values)
        val_labels = val_labels/max(class_values)
        test_labels = test_labels/max(class_values)

    n_row, n_col = adj_train.shape
    if (u_features is None) or (v_features is None):
        x = torch.tensor(create_node(adj_train, node_mode), dtype=torch.float)
    else:
        print("using given feature")
        x = torch.zeros((u_features.shape[0]+v_features.shape[0],
                        np.maximum(u_features.shape[1],v_features.shape[1]))
                        ,dtype=torch.float)
        x[:u_features.shape[0],:u_features.shape[1]] = torch.tensor(u_features,dtype=torch.float)
        x[u_features.shape[0]:,:v_features.shape[1]] = torch.tensor(v_features,dtype=torch.float)

    train_v_indices = train_v_indices + n_row
    train_edge_index = torch.tensor([np.append(train_u_indices,train_v_indices),
                        np.append(train_v_indices,train_u_indices)],
                        dtype=int)
    if one_hot_edge:
        train_edge_attr = one_hot(train_indices, len(class_values))
        train_edge_attr = torch.cat((train_edge_attr, train_edge_attr),0)
    elif soft_one_hot_edge:
        train_edge_attr = soft_one_hot(train_indices, len(class_values))
        train_edge_attr = torch.cat((train_edge_attr, train_edge_attr),0)
    else:
        train_edge_attr = torch.tensor(np.append(train_labels,train_labels)[:,None],
                            dtype=torch.float)

    val_v_indices = val_v_indices + n_row
    val_edge_index = torch.tensor([np.append(val_u_indices,val_v_indices),
                        np.append(val_v_indices,val_u_indices)],
                        dtype=int)
    if one_hot_edge:
        val_edge_attr = one_hot(val_indices, len(class_values))
        val_edge_attr = torch.cat((val_edge_attr, val_edge_attr),0)
    elif soft_one_hot_edge:
        val_edge_attr = soft_one_hot(val_indices, len(class_values))
        val_edge_attr = torch.cat((val_edge_attr, val_edge_attr),0)
    else:
        val_edge_attr = torch.tensor(np.append(val_labels,val_labels)[:,None],
                            dtype=torch.float)

    test_v_indices = test_v_indices + n_row
    test_edge_index = torch.tensor([np.append(test_u_indices,test_v_indices),
                        np.append(test_v_indices,test_u_indices)],
                        dtype=int)
    if one_hot_edge:
        test_edge_attr = one_hot(test_indices, len(class_values))
        test_edge_attr = torch.cat((test_edge_attr, test_edge_attr),0)
    elif soft_one_hot_edge:
        test_edge_attr = soft_one_hot(test_indices, len(class_values))
        test_edge_attr = torch.cat((test_edge_attr, test_edge_attr),0)
    else:
        test_edge_attr = torch.tensor(np.append(test_labels,test_labels)[:,None],
                            dtype=torch.float)
        
    if ce_loss:
        train_labels = torch.tensor(train_indices,dtype=int)
        val_labels = torch.tensor(val_indices,dtype=int)
        test_labels = torch.tensor(test_indices,dtype=int)

    data = Data(x=x,
            train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,train_labels=train_labels,
            val_edge_index=val_edge_index,val_edge_attr=val_edge_attr,val_labels=val_labels,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr,test_labels=test_labels,
            edge_attr_dim=train_edge_attr.shape[-1], class_values=torch.FloatTensor(class_values),
            user_num=adj_train.shape[0]
            )
    return data

# def split_edge(edge_attr,edge_index,batch_size):
#     edge_num = int(edge_attr.shape[0]/2)
#     perm = np.random.permutation(edge_num)
#     edge_attrs, edge_indexes = [],[]
#     index = 0
#     while index + batch_size < edge_num:
#         edge_attr_i = torch.cat((edge_attr[index:index+batch_size,:],
#                                 edge_attr[index:index+batch_size,:]),dim=0)
#         edge_start_i = torch.cat([edge_index[0,index:index+batch_size],
#                                     edge_index[1,index:index+batch_size]])
#         edge_end_i = torch.cat([edge_index[1,index:index+batch_size],
#                                     edge_index[0,index:index+batch_size]])
#         edge_index_i = torch.tensor([edge_start_i.detach().numpy(),
#                                     edge_end_i.detach().numpy()],
#                                     dtype=int)
#         index += batch_size
#         edge_attrs.append(edge_attr_i)
#         edge_indexes.append(edge_index_i)
#     return edge_attrs, edge_indexes

def load_data(args):
    mc_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    rating_map, post_rating_map = None, None
    if args.standard_rating:
        if args.data in ['flixster', 'ml_10m']: # original 0.5, 1, ..., 5
            rating_map = {x: int(math.ceil(x)) for x in np.arange(0.5, 5.01, 0.5).tolist()}
        elif args.data == 'yahoo_music':  # original 1, 2, ..., 100
            rating_map = {x: (x-1)//20+1 for x in range(1, 101)}
        else:
            rating_map = None

    if args.data == 'ml_1m' or args.data == 'ml_10m':
        if args.use_features:
            datasplit_path = mc_path+'/raw_data/' + args.data + '/withfeatures_split_seed' + str(args.data_seed) + '.pickle'
        else:
            datasplit_path = mc_path+'/raw_data/' + args.data + '/split_seed' + str(args.data_seed) + '.pickle'
    elif args.use_features:
        datasplit_path = mc_path+'/raw_data/' + args.data + '/withfeatures.pickle'
    else:
        datasplit_path = mc_path+'/raw_data/' + args.data + '/nofeatures.pickle'

    if args.data == 'flixster' or args.data == 'douban' or args.data == 'yahoo_music':
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
            val_labels, val_u_indices, val_v_indices, test_labels, \
            test_u_indices, test_v_indices, class_values = load_data_monti(args.data, args.testing, rating_map, post_rating_map)
    elif args.data == 'ml_100k':
        print("Using official MovieLens dataset split u1.base/u1.test with 20% validation set size...")
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
            val_labels, val_u_indices, val_v_indices, test_labels, \
            test_u_indices, test_v_indices, class_values = load_official_trainvaltest_split(args.data, args.testing, rating_map, post_rating_map, args.ratio)
    else:
        print("Using random dataset split ...")
        u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
            val_labels, val_u_indices, val_v_indices, test_labels, \
            test_u_indices, test_v_indices, class_values = create_trainvaltest_split(args.data, args.data_seed, args.testing, datasplit_path, True, True, rating_map, post_rating_map, args.ratio)

    print('All ratings are:')
    print(class_values)
    '''
    Explanations of the above preprocessing:
        class_values are all the original continuous ratings, e.g. 0.5, 2...
        They are transformed to rating labels 0, 1, 2... acsendingly.
        Thus, to get the original rating from a rating label, apply: class_values[label]
        Note that train_labels etc. are all rating labels.
        But the numbers in adj_train are rating labels + 1, why? Because to accomodate neutral ratings 0! Thus, to get any edge label from adj_train, remember to substract 1.
        If testing=True, adj_train will include both train and val ratings, and all train data will be the combination of train and val.
    '''

    if args.use_features:
        u_features, v_features = u_features.toarray(), v_features.toarray()
        n_features = u_features.shape[1] + v_features.shape[1]
        print('Number of user features {}, item features {}, total features {}'.format(u_features.shape[1], v_features.shape[1], n_features))
    else:
        u_features, v_features = None, None
        n_features = 0

    if args.debug:  # use a small number of data to debug
        num_data = 1000
        train_u_indices, train_v_indices = train_u_indices[:num_data], train_v_indices[:num_data]
        train_labels = train_labels[:num_data]
        val_u_indices, val_v_indices = val_u_indices[:num_data], val_v_indices[:num_data]
        val_labels = val_labels[:num_data]
        test_u_indices, test_v_indices = test_u_indices[:num_data], test_v_indices[:num_data]
        test_labels = test_labels[:num_data]

    print('#train: %d, #val: %d, #test: %d' % (len(train_u_indices), len(val_u_indices), len(test_u_indices)))

    '''
        Transfer to torch geometric Data
    '''
    if not hasattr(args,'soft_one_hot_edge'):
        args.soft_one_hot_edge = False
    data = get_data(u_features, v_features, args.node_mode, adj_train,
        train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values,
        args.one_hot_edge, args.soft_one_hot_edge, args.norm_label, args.ce_loss)
    return data

