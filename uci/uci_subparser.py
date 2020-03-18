import argparse

def add_uci_subparser(subparsers):
    subparser = subparsers.add_parser('uci')
    # mc settings
    subparser.add_argument('--domain', type=str, default='uci')
    subparser.add_argument('--data', type=str, default='housing')