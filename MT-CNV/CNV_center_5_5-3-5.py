import sys
import logging

import torch

import main_class_center
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext
import math
import os

LOG = logging.getLogger('runner')

acc_n = []
ema_acc_n = []
auc_n = []
ema_auc_n = []
spe_n = []
ema_spe_n = []
sen_n = []
ema_sen_n = []


def parameters():
    defaults = {
        # Data
        'dataset': 'CNV',
        'train_subdir': 'train_6',
        # 'eval_subdir': 'val_6',
        'eval_subdir': 'old_test_3_80',
        'center_subdir': 'center_v1',

        # Data sampling
        'base_batch_size': 64,
        'base_labeled_batch_size': 3,
        # 'base_batch_size': 128,
        # 'base_labeled_batch_size': 32,


        # Costs
        'consistency_weight_1': 5,
        'consistency_weight_2': 5,
        'alpha': 0.5,
        'lamda': 0.005,


        # Optimization
        'base_lr': 0.1,
        'evaluate': True,

    }

    a = []
    with open('./results/CNV_center_5_5-3-5/record_CNV_center_5_5-3-5.txt') as f:
        for line in f.read().splitlines():
            b = list(line.split(' '))
            if b[0] == 'num':
                continue
            elif b[0] == '----------------------------------------':
                break
            c = b[3] + ' ' + b[4] + ' ' + b[5]
            a.append(c)

    # 630(5%) labels:
    for data_seed in range(0, 10):
        r = './results/CNV_center_5_5-3-5/' + str(a[data_seed]) + '/630_' + str(data_seed) + '/transient/best.ckpt'
        # if not os.path.isfile(r+'.f'):
        #     r = './results/CNV_center_5_5-3-5/' + str(a[data_seed]) + '/630_' + str(data_seed) + '/transient/checkpoint.185.ckpt'

        yield {
            **defaults,
            'title': '630-label cnv',
            'n_labels': 630,
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
        'labels': 'data-local/labels/CNV/{}_labels_train/{:02d}.txt'.format(n_labels, data_seed),
    }
    context = RunContext(__file__, "{}_{}".format(n_labels, data_seed))
    main_class_center.args = parse_dict_args(**adapted_args, **kwargs)

    # main_class_center.main(context)
    test(context)


def test(context):
    acc, auc, spe, sen, ema_acc, ema_auc, ema_spe, ema_sen = main_class_center.main(context)
    acc_n.append(acc)
    auc_n.append(auc)
    spe_n.append(spe)
    sen_n.append(sen)
    ema_acc_n.append(ema_acc)
    ema_auc_n.append(ema_auc)
    ema_spe_n.append(ema_spe)
    ema_sen_n.append(ema_sen)


def calculate():
    n = len(acc_n)
    M, ema_M, S, ema_S = junfang(acc_n, ema_acc_n)
    print("acc:", acc_n)
    print("ema_acc:", ema_acc_n)
    print("total: {} runs\n均值:{:.2f} | {:.2f}\n方差:{:.2f} | {:.2f}\n标准差:{:.2f} | {:.2f}\n".format(n, M, ema_M, S, ema_S,
                                                                                                 math.sqrt(S),
                                                                                                 math.sqrt(ema_S)))

    M, ema_M, S, ema_S = junfang(auc_n, ema_auc_n)
    print("auc:", auc_n)
    print("ema_auc:", ema_auc_n)
    print("total: {} runs\n均值:{:.2f} | {:.2f}\n方差:{:.2f} | {:.2f}\n标准差:{:.2f} | {:.2f}\n".format(n, M, ema_M, S, ema_S,
                                                                                                 math.sqrt(S),
                                                                                                 math.sqrt(ema_S)))
    M, ema_M, S, ema_S = junfang(spe_n, ema_spe_n)
    print("specificity:", spe_n)
    print("ema_specificity:", ema_spe_n)
    print("total: {} runs\n均值:{:.2f} | {:.2f}\n方差:{:.2f} | {:.2f}\n标准差:{:.2f} | {:.2f}\n".format(n, M, ema_M, S, ema_S,
                                                                                                 math.sqrt(S),
                                                                                                 math.sqrt(ema_S)))
    M, ema_M, S, ema_S = junfang(sen_n, ema_sen_n)
    print("sensitivity:", sen_n)
    print("ema_sensitivity:", ema_sen_n)
    print("total: {} runs\n均值:{:.2f} | {:.2f}\n方差:{:.2f} | {:.2f}\n标准差:{:.2f} | {:.2f}\n".format(n, M, ema_M, S, ema_S,
                                                                                                 math.sqrt(S),
                                                                                                 math.sqrt(ema_S)))


def junfang(a_n, ema_a_n):
    n = len(a_n)
    sum = 0.0
    ema_sum = 0.0
    for i in range(n):
        sum += a_n[i]
        ema_sum += ema_a_n[i]
    M = sum / n
    ema_M = ema_sum / n

    S = 0.0
    ema_S = 0.0
    for i in range(n):
        S += math.pow((a_n[i] - M), 2)
        ema_S += math.pow((ema_a_n[i] - ema_M), 2)
    S = S / n
    ema_S = ema_S / n
    return M, ema_M, S, ema_S


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
    calculate()
