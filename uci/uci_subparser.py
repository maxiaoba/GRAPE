import argparse

def add_uci_subparser(subparsers):
    subparser = subparsers.add_parser('uci')
    # mc settings
    subparser.add_argument('--domain', type=str, default='uci')
    subparser.add_argument('--data', type=str, default='housing')
    subparser.add_argument('--train_edge', type=float, default=0.7)
    subparser.add_argument('--train_y', type=float, default=0.7)