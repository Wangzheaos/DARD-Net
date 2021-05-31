# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.`

import re
import argparse
import logging

from . import architectures, datasets


LOG = logging.getLogger('main')

# __all__ = ['parse_cmd_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', metavar='DATASET', default='CNV',
                        choices=datasets.__all__)
    parser.add_argument('--train-subdir', type=str, default='train_6',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--eval-subdir', type=str, default='val_6',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--center-subdir', type=str, default='center_v1',
                        help='the subdirectory inside the data directory that contains the center_v data')
    parser.add_argument('--labels', default='./data-local/labels/CNV/3150_labels_train/00.txt', type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
    parser.add_argument('--exclude-unlabeled', default=False, type=str2bool, metavar='BOOL',
                        help='exclude unlabeled examples from the training set')
    parser.add_argument('--arch', '-a', metavar='ARCH', default=None,
                        choices=architectures.__all__,
                        help='model architecture: ' +
                            ' | '.join(architectures.__all__))
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled-batch-size', default=16, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=210, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency-weight-1', default=5, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-weight-2', default=5, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', default=10, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--checkpoint-epochs', default=1, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help="path to latest checkpoint (default: './results/main/2020-10-08_20 40 43/0/transient/best.ckpt')")
    parser.add_argument('-e', '--evaluate', default=False, type=str2bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--feature_dim',  default=64, type=int, metavar='feature_dim',
                        help='feature_dim')
    parser.add_argument('--relation_dim', default=8, type=int, metavar='relation_dim',
                        help='relation_dim')
    parser.add_argument('--beta', default=1.0, type=float, metavar='cutmix',
                        help='cutmix')

    parser.add_argument('--alpha', default=0.2, type=float, metavar='center_loss',
                        help='center_loss')
    parser.add_argument('--lamda', default=5, type=float, metavar='center_loss',
                        help='center_loss')


    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs


