import torch.optim as optim
import numpy as np
import os.path as osp
import torch

def get_activateion(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation is None:
        return torch.nn. Identity()
    else:
        raise NotImplementedError

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def save_mask(length,true_rate,log_dir,seed):
    np.random.seed(seed)
    mask = np.random.rand(length) < true_rate
    np.save(osp.join(log_dir,'len'+str(length)+'rate'+str(true_rate)+'seed'+str(seed)),mask)
    return mask

def get_train_mask(valid,load,load_path,data):
    if load:
        print('loading train validation mask')
        train_rate = 1-valid
        train_mask_dir = load_path+'len'+str(int(data.edge_attr.shape[0]/2))+'rate'+f'{train_rate:.1f}'+'seed0.npy'
        if not osp.exists(train_mask_dir):
            from utils import save_mask
            save_mask(int(data.edge_attr.shape[0]/2),train_rate,log_path+'../',0)
        print(train_mask_dir)
        train_mask = np.load(train_mask_dir)
        train_mask = torch.BoolTensor(train_mask).view(-1)
    else:
        print('defining train validation mask')
        train_mask = (torch.FloatTensor(int(data.edge_attr.shape[0]/2), 1).uniform_() < (1-valid)).view(-1)
        #print(data.edge_attr.shape[0])
    #print(len(train_mask))

    return train_mask

def mask_edge(edge_index,edge_attr,mask,remove_edge):
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()
    if remove_edge:
        edge_index = edge_index[:,mask]
        edge_attr = edge_attr[mask]
    else:
        edge_attr[~mask] = 0.
    return edge_index, edge_attr