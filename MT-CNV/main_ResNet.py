import os
import shutil
import time
import math
import logging

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

import torchvision.datasets
import torchvision.transforms as transforms

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
import time
# from apex import amp

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0
class_num = 0


def main(context):
    global global_step
    global best_prec1
    global class_num
    best_prec1 = 0
    global_step = 0
    class_num = 0



    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    dataset_config = datasets.__dict__[args.dataset]()
    class_num = dataset_config['num_classes']

    train_loader, eval_loader, center_loader = create_data_loaders(**dataset_config, args=args)

    feature_encoder, relation_network = create_model(class_num=class_num)

    ema_feature_encoder, ema_relation_network = create_model(ema=True, class_num=class_num)

    LOG.info(parameters_string(feature_encoder))
    LOG.info(parameters_string(relation_network))

    optimizer_f = torch.optim.SGD(feature_encoder.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)
    optimizer_r = torch.optim.SGD(relation_network.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)

    # feature_encoder, optimizer_f = amp.initialize(feature_encoder, optimizer_f, opt_level="O1")
    # relation_network, optimizer_r = amp.initialize(relation_network, optimizer_r, opt_level="O1")

    # optionally resume from a checkpoint
    if args.resume:
        # assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)

        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        resume_f = args.resume + '.f'
        resume_r = args.resume + '.r'
        checkpoint_f = torch.load(resume_f)
        checkpoint_r = torch.load(resume_r)
        args.start_epoch = checkpoint_f['epoch']
        global_step = checkpoint_f['global_step']
        best_prec1 = checkpoint_f['best_prec1']
        feature_encoder.load_state_dict(checkpoint_f['state_dict'])
        ema_feature_encoder.load_state_dict(checkpoint_f['ema_state_dict'])
        optimizer_f.load_state_dict(checkpoint_f['optimizer'])

        relation_network.load_state_dict(checkpoint_r['state_dict'])
        ema_relation_network.load_state_dict(checkpoint_r['ema_state_dict'])
        optimizer_r.load_state_dict(checkpoint_r['optimizer'])

        # amp.load_state_dict(checkpoint_f['amp'])

        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint_f['epoch']))

    cudnn.benchmark = True

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        acc = validate(eval_loader, center_loader, feature_encoder, relation_network, validation_log, global_step,
                       args.start_epoch)
        print("the primary model Acc:{}%".format(acc))
        LOG.info("Evaluating the EMA model:")
        ema_acc = validate(eval_loader, center_loader, ema_feature_encoder, ema_relation_network, ema_validation_log,
                           global_step, args.start_epoch)
        print("the EMA model Acc:{}%".format(ema_acc))
        return acc, ema_acc

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train(train_loader, center_loader, feature_encoder, relation_network, ema_feature_encoder, ema_relation_network,
              optimizer_f, optimizer_r, epoch, training_log)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(eval_loader, center_loader, feature_encoder, relation_network, validation_log, global_step,
                             epoch + 1)
            LOG.info("Evaluating the EMA model:")
            ema_prec1 = validate(eval_loader, center_loader, ema_feature_encoder, ema_relation_network,
                                 ema_validation_log, global_step, epoch + 1)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'state_dict': feature_encoder.state_dict(),
                'ema_state_dict': ema_feature_encoder.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer_f.state_dict(),
            }, '.f', is_best, checkpoint_path, epoch + 1)
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'state_dict': relation_network.state_dict(),
                'ema_state_dict': ema_relation_network.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer_r.state_dict(),
            }, '.r', is_best, checkpoint_path, epoch + 1)


def create_model(ema=False, class_num=3):
    LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
        pretrained='pre-trained ' if args.pretrained else '',
        ema='EMA ' if ema else '',
        arch=args.arch))

    model_factory = architectures.__dict__['resnet32']
    feature_encoder = model_factory(num_classes=class_num)

    model_factory = architectures.__dict__['relationnet']
    relation_network = model_factory(args.feature_dim, args.relation_dim)
    # feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder = nn.DataParallel(feature_encoder).cuda()
    relation_network = nn.DataParallel(relation_network).cuda()

    if ema:
        for param in feature_encoder.parameters():
            param.detach_()
        for param in relation_network.parameters():
            param.detach_()

    return feature_encoder, relation_network


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def create_data_loaders(train_transformation,
                        eval_transformation,
                        cent_transformation,
                        datadir,
                        num_classes,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)
    centerdir = os.path.join(datadir, args.center_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    def pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation, None)

    print(dataset.class_to_idx)

    with open(args.labels) as f:
        labels = dict(line.split(' ') for line in f.read().splitlines())
    labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

    batch_sampler = data.TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=False)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation, None),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=False,
        drop_last=True)

    center_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(centerdir, cent_transformation, None),
        batch_size=num_classes,
        pin_memory=True)

    return train_loader, eval_loader, center_loader


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, center_loader, feature_encoder, relation_network, ema_feature_encoder, ema_relation_network,
          optimizer_f, optimizer_r, epoch, log):
    global global_step
    global class_num

    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
    consistency_criterion_1 = losses.softmax_mse_loss1
    consistency_criterion_2 = losses.softmax_mse_loss2

    meters = AverageMeterSet()

    # switch to train mode
    feature_encoder.train()
    relation_network.train()
    ema_feature_encoder.train()
    ema_relation_network.train()

    cent_input, cent_target = center_loader.__iter__().next()

    end = time.time()

    """加快数据加载速度"""
    train_loader_p = data.data_prefetcher(train_loader)
    Input2, target = train_loader_p.next()
    i = 0
    target2 = torch.zeros(target.size(0))
    while Input2 is not None:
        input, ema_input = Input2
    # for i, ((input, ema_input), target) in enumerate(train_loader):
        # measure data loading time
        meters.update('data_time', time.time() - end)
        adjust_learning_rate(optimizer_f, epoch, i, len(train_loader))
        adjust_learning_rate(optimizer_r, epoch, i, len(train_loader))
        meters.update('lr_f', optimizer_f.param_groups[0]['lr'])
        meters.update('lr_r', optimizer_r.param_groups[0]['lr'])
        # print(target)

        for k in range(target.size(0)):
            if target[k] == -1:
                target2[k] = -1
            else:
                target2[k] = target[k] // 2
        target2_t = target2.long()

        input_var = torch.autograd.Variable(input)
        with torch.no_grad():
            ema_input_var = torch.autograd.Variable(ema_input)
            ema_cent_input_var = torch.autograd.Variable(cent_input)
        target_var = torch.autograd.Variable(target.cuda())

        target_var2 = torch.autograd.Variable(target2_t.cuda())

        cent_input_var = torch.autograd.Variable(cent_input)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().item()
        unlabeled_minibatch_size = minibatch_size - labeled_minibatch_size

        # calculate features
        center_features, _ = feature_encoder(cent_input_var.cuda())  # 5x64*5*5
        batch_features, batch_features2 = feature_encoder(input_var.cuda())  # 20x64*5*5

        ema_center_features, _ = ema_feature_encoder(ema_cent_input_var.cuda())
        ema_batch_features, ema_batch_features2 = ema_feature_encoder(ema_input_var.cuda())

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        center_features_ext = center_features.unsqueeze(0).repeat(minibatch_size, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(1 * class_num, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)  #交换两个维度
        # print(center_features_ext.size())
        print(batch_features.size())
        time.sleep(10)

        relation_pairs = torch.cat((center_features_ext, batch_features_ext), 2).view(-1, args.feature_dim * 2, 1, 1) #拼接按第二维
        relations = relation_network(relation_pairs).view(-1, class_num * 1)

        ema_center_features_ext = ema_center_features.unsqueeze(0).repeat(minibatch_size, 1, 1, 1, 1)
        ema_batch_features_ext = ema_batch_features.unsqueeze(0).repeat(1 * class_num, 1, 1, 1, 1)
        ema_batch_features_ext = torch.transpose(ema_batch_features_ext, 0, 1)
        ema_relation_pairs = torch.cat((ema_center_features_ext, ema_batch_features_ext), 2).view(-1,
                                                                                                  args.feature_dim * 2,
                                                                                                  1, 1)
        ema_relations = ema_relation_network(ema_relation_pairs).view(-1, class_num * 1)

        class_loss = class_criterion(batch_features2, target_var2) / minibatch_size
        meters.update('class_loss', class_loss.item())

        ema_class_loss = class_criterion(ema_batch_features2, target_var2) / minibatch_size
        meters.update('ema_class_loss', ema_class_loss.item())

        consistency_weight_1 = get_current_consistency_weight(args.consistency_weight_1, epoch)
        consistency_weight_2 = get_current_consistency_weight(args.consistency_weight_2, epoch)
        meters.update('cons_weight_1', consistency_weight_1)
        meters.update('cons_weight_2', consistency_weight_2)
        consistency_loss_1 = consistency_weight_1 * consistency_criterion_1(relations, ema_relations)
        consistency_loss_2 = consistency_weight_2 * consistency_criterion_2(relations, ema_relations)

        meters.update('cons_loss_1', consistency_loss_1.item())
        meters.update('cons_loss_2', consistency_loss_2.item())

        # print("cons_loss:", consistency_loss)

        loss = class_loss + consistency_loss_1 + consistency_loss_2
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())
        # print("class：",class_loss,"\tcons:",consistency_loss,"\tSia:",Siamese_loss)
        prec1 = accuracy(relations.data, target_var.data, topk=1)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)

        ema_prec1 = accuracy(ema_relations.data, target_var.data, topk=1)
        meters.update('ema_top1', ema_prec1[0], labeled_minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer_r.zero_grad()
        optimizer_f.zero_grad()

        # with amp.scale_loss(loss, optimizer_r) as scaled_loss:
        #     scaled_loss.backward()

        loss.backward()
        optimizer_r.step()
        optimizer_f.step()
        global_step += 1
        update_ema_variables(feature_encoder, ema_feature_encoder, args.ema_decay, global_step)
        update_ema_variables(relation_network, ema_relation_network, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                # 'Time {meters[batch_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Cons_1 {meters[cons_loss_1]:.4f}\t'
                'Cons_2 {meters[cons_loss_2]:.4f}\t'
                'Loos {meters[loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}'
                .format(epoch, i, len(train_loader), meters=meters))

            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })

        i += 1
        Input2, target = train_loader_p.next()


def validate(eval_loader, center_loader, feature_encoder, relation_network, log, global_step, epoch, accuracy_f):
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    feature_encoder.eval()
    relation_network.eval()

    cent_input, cent_target = center_loader.__iter__().next()

    end = time.time()

    """加快数据加载速度"""
    eval_loader_p = data.data_prefetcher(eval_loader)
    input, target = eval_loader_p.next()
    target2 = torch.zeros(target.size(0))
    i = 0
    while input is not None:
    # for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        for k in range(target.size(0)):
            if target[k] == -1:
                target2[k] = -1
            else:
                target2[k] = target[k] // 2
        target2_t = target2.long()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target.cuda())
            target_var2 = torch.autograd.Variable(target2_t.cuda())
            cent_input_var = torch.autograd.Variable(cent_input)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().item()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # calculate features
        center_features, _ = feature_encoder(cent_input_var.cuda())  # 5x64*5*5
        batch_features, batch_features2 = feature_encoder(input_var.cuda())  # 20x64*5*5

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        center_features_ext = center_features.unsqueeze(0).repeat(minibatch_size, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(1 * class_num, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

        relation_pairs = torch.cat((center_features_ext, batch_features_ext), 2).view(-1, args.feature_dim * 2, 1, 1)
        relations = relation_network(relation_pairs).view(-1, class_num * 1)
        # print("relations:", relations)
        # print("target_var:", target_var)
        class_loss = class_criterion(batch_features2, target_var2) / minibatch_size

        # measure accuracy and record loss
        prec1 = accuracy(relations.data, target_var.data, topk=1)
        meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                # 'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}'
                .format(i, len(eval_loader), meters=meters))

        i += 1
        input, target = eval_loader_p.next()

    LOG.info(' * Prec@1 {top1.avg:.3f}'.format(top1=meters['top1']))
    log.record(epoch, {
        'step': global_step,
        **meters.values(),
        **meters.averages(),
        **meters.sums()
    })

    return meters['top1'].avg


def save_checkpoint(state, name, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch) + name
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt' + name)
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        torch.save(state, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(consistency_weight, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency_weight * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, topk=1):
    maxk = 1
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum().item(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    total_num = target.ne(-2).sum().item()
    target_1 = target.view(1, -1).expand_as(pred)
    num = 0
    # print("total_num:", total_num, "  labeled_minibatch_size:", labeled_minibatch_size)
    for i in range(0, total_num):
        if pred[0][i] < 2 and -1 < target_1[0][i] < 2:
            num = num + 1
        elif 1 < pred[0][i] < 4 and 1 < target_1[0][i] < 4:
            num = num + 1
        elif 3 < pred[0][i] and 3 < target_1[0][i]:
            num = num + 1

    res = [num * 100.0 / labeled_minibatch_size]
    return res


def accuracy_test(output, target, topk=1):
    maxk = 1
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum().item(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    total_num = target.ne(-2).sum().item()
    target_1 = target.view(1, -1).expand_as(pred)
    num = 0
    # print("total_num:", total_num, "  labeled_minibatch_size:", labeled_minibatch_size)
    for i in range(0, total_num):
        if pred[0][i] < 2 and target_1[0][i] == 0:
            num = num + 1
        elif 1 < pred[0][i] < 4 and target_1[0][i] == 1 :
            num = num + 1
        elif 3 < pred[0][i] and target_1[0][i] == 2:
            num = num + 1

    res = [num * 100.0 / labeled_minibatch_size]

    auc, spe, sen = calcAUC_Spe_Sen(pred[0], target_1[0], total_num)

    return res, auc, spe, sen


def calcAUC_Spe_Sen(pred, target, total_num):
    AUC = 0.0
    Spe = 0.0
    Sen = 0.0
    y_pred = []
    y_target = []
    # 计算第一类
    for i in range(0, total_num):
        if pred[i] == 0 or pred[i] == 1:
            y_pred.append(0)
        else:
            y_pred.append(1)
        if target[i] != 0:
            y_target.append(1)
        else:
            y_target.append(0)
    C1 = confusion_matrix(y_target, y_pred, labels=list(set(y_target)))
    AUC += roc_auc_score(y_target, y_pred)
    Sen += C1[0][0] / (C1[0][0] + C1[0][1])
    Spe += C1[1][1] / (C1[1][0] + C1[1][1])

    y_pred.clear()
    y_target.clear()
    # 计算第二类
    for i in range(0, total_num):
        if pred[i] == 2 or pred[i] == 3:
            y_pred.append(0)
        else:
            y_pred.append(1)
        if target[i] == 1:
            y_target.append(0)
        else:
            y_target.append(1)
    C2 = confusion_matrix(y_target, y_pred, labels=list(set(y_target)))
    AUC += roc_auc_score(y_target, y_pred)
    Sen += C2[0][0] / (C2[0][0] + C2[0][1])
    Spe += C2[1][1] / (C2[1][0] + C2[1][1])

    y_pred.clear()
    y_target.clear()
    # 计算第三类
    for i in range(0, total_num):
        if pred[i] == 4 or pred[i] == 5:
            y_pred.append(0)
        else:
            y_pred.append(1)
        if target[i] == 2:
            y_target.append(0)
        else:
            y_target.append(1)
    C3 = confusion_matrix(y_target, y_pred, labels=list(set(y_target)))
    AUC += roc_auc_score(y_target, y_pred)
    Sen += C3[0][0] / (C3[0][0] + C3[0][1])
    Spe += C3[1][1] / (C3[1][0] + C3[1][1])

    AUC = AUC * 100 / 3
    Sen = Sen * 100 / 3
    Spe = Spe * 100 / 3

    return AUC, Spe, Sen


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
