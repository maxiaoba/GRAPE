import argparse

def add_uci_subparser(subparsers):
    subparser = subparsers.add_parser('uci')
    # mc settings
    subparser.add_argument('--domain', type=str, default='uci')
    subparser.add_argument('--data', type=str, default='housing')
    subparser.add_argument('--train_edge', type=float, default=0.7)
    subparser.add_argument('--split_sample', type=float, default=0.)
    subparser.add_argument('--split_by', type=str, default='y') # 'y', 'random'
    subparser.add_argument('--split_train', action='store_true', default=False)
    subparser.add_argument('--split_test', action='store_true', default=False)
    subparser.add_argument('--train_y', type=float, default=0.7)
    subparser.add_argument('--node_mode', type=int, default=0)  # 0: feature onehot, sample all 1; 1: all onehot