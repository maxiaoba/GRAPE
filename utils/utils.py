import torch.optim as optim
import numpy as np
import os.path as osp
import torch
import subprocess

def np_random(seed=None):
    rng = np.random.RandomState()
    rng.seed(seed)
    return rng

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return scheduler, optimizer

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def save_mask(length,true_rate,log_dir,seed):
    np.random.seed(seed)
    mask = np.random.rand(length) < true_rate
    np.save(osp.join(log_dir,'len'+str(length)+'rate'+str(true_rate)+'seed'+str(seed)),mask)
    return mask

def get_known_mask(known_prob, edge_num):
    known_mask = (torch.FloatTensor(edge_num, 1).uniform_() < known_prob).view(-1)
    return known_mask

def mask_edge(edge_index,edge_attr,mask,remove_edge):
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()
    if remove_edge:
        edge_index = edge_index[:,mask]
        edge_attr = edge_attr[mask]
    else:
        edge_attr[~mask] = 0.
    return edge_index, edge_attr

def one_hot(batch,depth):
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0,torch.tensor(batch,dtype=int))

def soft_one_hot(batch,depth):
    batch = torch.tensor(batch)
    encodings = torch.zeros((batch.shape[0],depth))
    for i,x in enumerate(batch):
        for r in range(depth):
            encodings[i,r] = torch.exp(-((x-float(r))/float(depth))**2)
        encodings[i,:] = encodings[i,:]/torch.sum(encodings[i,:])
    return encodings

def construct_missing_X_from_mask(train_mask, df):
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

def construct_missing_X_from_edge_index(train_edge_index, df):
    nrow, ncol = df.shape
    data_incomplete = np.zeros((nrow, ncol))
    data_complete = np.zeros((nrow, ncol)) 
    train_edge_list = torch.transpose(train_edge_index,1,0).numpy()
    train_edge_list = list(map(tuple,[*train_edge_list]))
    for i in range(nrow):
        for j in range(ncol):
            data_complete[i,j] = df.iloc[i,j]
            if (i,j) in train_edge_list:
                data_incomplete[i,j] = df.iloc[i,j]
            else:
                data_incomplete[i,j] = np.NaN
    return data_complete, data_incomplete

# get gpu usage
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

def auto_select_gpu(memory_threshold = 7000, smooth_ratio=200, strategy='greedy'):
    gpu_memory_raw = get_gpu_memory_map() + 10
    if strategy=='random':
        gpu_memory = gpu_memory_raw/smooth_ratio
        gpu_memory = gpu_memory.sum() / (gpu_memory+10)
        gpu_memory[gpu_memory_raw>memory_threshold] = 0
        gpu_prob = gpu_memory / gpu_memory.sum()
        cuda = str(np.random.choice(len(gpu_prob), p=gpu_prob))
        print('GPU select prob: {}, Select GPU {}'.format(gpu_prob, cuda))
    elif strategy == 'greedy':
        cuda = np.argmin(gpu_memory_raw)
        print('GPU mem: {}, Select GPU {}'.format(gpu_memory_raw[cuda], cuda))
    return cuda