import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from training.linear_regression import linear_regression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='uci') # 'uci'
    parser.add_argument('--data', type=str, default='housing')  # 'pks', 'cancer', 'housing', 'wine'
    parser.add_argument('--method', type=str, default='mean')
    parser.add_argument('--train_edge', type=float, default=0.7)
    parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--log_dir', type=str, default='lry0')
    parser.add_argument('--load_dir', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.domain == 'uci':
        from uci.uci_data import load_data
        data = load_data(args)

    log_path = './{}/test/{}/{}_{}/'.format(args.domain,args.data,args.method,args.log_dir)
    os.mkdir(log_path)

    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
        
    load_path = './{}/test/{}/{}/'.format(args.domain,args.data,args.load_dir)
    linear_regression(data, args, log_path, load_path)

if __name__ == '__main__':
    main()