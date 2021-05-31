import sys
import logging

import torch

import main
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext
import math
import os

LOG = logging.getLogger('runner')

acc_n = []
ema_acc_n = []


def parameters():
    defaults = {
        # Data
        'dataset': 'CNV',
        'train_subdir': 'train1',
        'eval_subdir': 'old_test1',

        # Data sampling
        'base_batch_size': 64,
        'base_labeled_batch_size': 16,
        # 'base_batch_size': 128,
        # 'base_labeled_batch_size': 32,


        # Costs
        'consistency_weight_1': 0,
        'consistency_weight_2': 5,

        # Optimization
        'base_lr': 0.1,
        'evaluate': True,

    }

    a = []
    best_num = []
    with open('./results/CNV_train_0_5/record_CNV_train_0_5.txt') as f:
        for line in f.read().splitlines():
            b = list(line.split(' '))
            if b[0] == 'num':
                continue
            elif b[0] == '----------------------------------------':
                break

            c = b[3] + ' ' + b[4] + ' ' + b[5]
            a.append(c)
            best_num.append(b[6])

    # 3150(25%) labels:
    for data_seed in range(10):
        r = './results/CNV_train_0_5/' + str(a[data_seed]) + '/3150_' + str(data_seed) + '/transient/best.ckpt'
        if not os.path.isfile(r+'.f'):
            r = './results/CNV_train_0_5/' + str(a[data_seed]) + '/3150_' + str(data_seed) + '/transient/checkpoint.' + best_num[data_seed] + '.ckpt'

        yield {
            **defaults,
            'title': '3150-label cnv',
            'n_labels': 3150,
            'data_seed': data_seed,
            'epochs': 200,
            'lr_rampdown_epochs': 210,
            'resume': r
        }


def run(title, base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, **kwargs):
    LOG.info('run title: %s, data seed: %d', title, data_seed)

    adapted_args = {
        'batch_size': base_batch_size,
        'labeled_batch_size': base_labeled_batch_size,
        'lr': base_lr,
        'labels': 'data-local/labels/CNV/{}_balanced_labels/{:02d}.txt'.format(n_labels, data_seed),
    }
    context = RunContext(__file__, "{}_{}".format(n_labels, data_seed))
    main.args = parse_dict_args(**adapted_args, **kwargs)

    # main.main(context)
    test(context)


def test(context):
    acc, ema_acc = main.main(context)
    acc_n.append(acc)
    ema_acc_n.append(ema_acc)


def calculate():
    n = len(acc_n)
    sum = 0.0
    ema_sum = 0.0
    for i in range(n):
        sum += acc_n[i]
        ema_sum += ema_acc_n[i]
    M = sum / n
    ema_M = ema_sum / n

    S = 0.0
    ema_S = 0.0
    for i in range(n):
        S += math.pow((acc_n[i] - M), 2)
        ema_S += math.pow((ema_acc_n[i] - ema_M), 2)
    S = S / n
    ema_S = ema_S / n
    print("acc:", acc_n)
    print("ema_acc:", ema_acc_n)
    print("total: {} runs\n均值:{} | {}\n方差:{} | {}".format(n, M, ema_M, S, ema_S))


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
    calculate()
