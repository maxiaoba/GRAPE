import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from training.gnn_mdi import train_gnn_mdi
from mc.mc_subparser import add_mc_subparser
from uci.uci_subparser import add_uci_subparser
from training.baseline import baseline_mdi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='mean')
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='0')
    subparsers = parser.add_subparsers()
    add_uci_subparser(subparsers)
    add_mc_subparser(subparsers)
    args = parser.parse_args()
    print(args)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.domain == 'uci':
        from uci.uci_data import load_data
        data = load_data(args)
    elif args.domain == 'mc':
        from mc.mc_data import load_data
        data = load_data(args)

    log_path = './{}/test/{}/{}_{}/'.format(args.domain,args.data,args.method,args.log_dir)
    os.makedirs(log_path)

    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)

    baseline_mdi(data, args, log_path)


if __name__ == '__main__':
    main()