import argparse

def add_mc_subparser(subparsers):
    subparser = subparsers.add_parser('mc')
    # mc settings
    subparser.add_argument('--domain', type=str, default='mc')
    subparser.add_argument('--data', type=str, default='douban')
    subparser.add_argument('--testing', action='store_true', default=False,
                        help='if set, use testing mode which splits all ratings into train/test;\
                        otherwise, use validation model which splits all ratings into \
                        train/val/test and evaluate on val only')
    subparser.add_argument('--debug', action='store_true', default=False,
                        help='turn on debugging mode which uses a small number of data')
    subparser.add_argument('--data-seed', type=int, default=1234, metavar='S',
                        help='seed to shuffle data (1234,2341,3412,4123,1324 are used), \
                        valid only for ml_1m and ml_10m')
    subparser.add_argument('--use-features', action='store_true', default=False,
                        help='whether to use node features (side information)')
    subparser.add_argument('--standard-rating', action='store_true', default=False,
                        help='if True, maps all ratings to standard 1, 2, 3, 4, 5 before training')
    # sparsity experiment settings
    subparser.add_argument('--ratio', type=float, default=1.0,
                        help="For ml datasets, if ratio < 1, downsample training data to the\
                        target ratio")
    # node mode
    subparser.add_argument('--node_mode', type=int, default=0)  # 0: feature onehot, sample all 1; 1: all onehot
    # one hot edge attr
    subparser.add_argument('--one_hot_edge', action='store_true', default=False,
                    help="Make edge attr onehot vectors")
    # one hot edge attr
    subparser.add_argument('--soft_one_hot_edge', action='store_true', default=False,
                    help="Make edge attr soft onehot vectors")
    # normalize edge attr
    subparser.add_argument('--norm_label', action='store_true', default=False,
                    help="Normalize edge labels")
    # cross_entropy loss
    subparser.add_argument('--ce_loss', action='store_true', default=False,
                    help="Use cross entropy loss")