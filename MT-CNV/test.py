import sys
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
from mean_teacher import architectures, datasets, data, losses, ramps, cli
from torch.utils.data import DataLoader
import torch.nn as nn
from mean_teacher.utils import *
from mean_teacher.run_context import RunContext
import math

test_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1989667, 0.1989667, 0.1989667],
                         std=[0.09176412, 0.09176412, 0.09176412])
])
center_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2405179, 0.2405179, 0.2405179],
                         std=[0.12401942, 0.12401942, 0.12401942])
])

classes = ['benign', 'malignant']
class_num = 4
feature_dim = 64
relation_dim = 8


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


def create_model(ema=False):
    model_factory = architectures.__dict__['cnnencoder']
    feature_encoder = model_factory()
    model_factory = architectures.__dict__['relationnet']
    relation_network = model_factory(feature_dim, relation_dim)
    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder = nn.DataParallel(feature_encoder).cuda()
    relation_network = nn.DataParallel(relation_network).cuda()

    if ema:
        for param in feature_encoder.parameters():
            param.detach_()
        for param in relation_network.parameters():
            param.detach_()

    return feature_encoder, relation_network


def prediect(model_path):
    ema_feature_encoder, ema_relation_network = create_model(ema=True)

    resume_f = model_path + '.f'
    resume_r = model_path + '.r'
    checkpoint_f = torch.load(resume_f)
    checkpoint_r = torch.load(resume_r)

    ema_feature_encoder.load_state_dict(checkpoint_f['ema_state_dict'])
    ema_relation_network.load_state_dict(checkpoint_r['ema_state_dict'])

    ema_feature_encoder.eval()
    ema_relation_network.eval()

    testdir = './data-local/images/ruxian/test/'
    centerdir = './data-local/images/ruxian/center_v/'
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(testdir, test_transform, None),
        batch_size=700,
        shuffle=True,
        num_workers=2 * 4,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    center_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(centerdir, center_transform, None),
        batch_size=class_num,
        pin_memory=True)

    cent_input, cent_target = center_loader.__iter__().next()
    num = 0
    total_num = 0
    for i, (input, target) in enumerate(test_loader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target.cuda())
            cent_input_var = torch.autograd.Variable(cent_input)

        minibatch_size = len(target_var)
        total_num = total_num + minibatch_size
        # calculate features
        center_features = ema_feature_encoder(cent_input_var.cuda())  # 5x64*5*5
        batch_features = ema_feature_encoder(input_var.cuda())  # 20x64*5*5

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        center_features_ext = center_features.unsqueeze(0).repeat(minibatch_size, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(1 * class_num, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((center_features_ext, batch_features_ext), 2).view(-1, feature_dim * 2, 18, 18)
        relations = ema_relation_network(relation_pairs).view(-1, class_num * 1)

        probability = torch.nn.functional.softmax(relations, dim=1)
        max_value, index = torch.max(probability, 1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
        # print(probability)
        # print('this breast tumor cell maybe :', classes[index])

        for j in range(0, minibatch_size):
            if index[j] < 2 and -1 < target[j] < 2:
                num = num + 1
            elif index[j] > 1 and target[j] > 1:
                num = num + 1
    acc = num * 100.0 / total_num
    print('correct num:', num, '\ntotal num:', total_num, '\nacc:', acc, '%')


# 命令行 第一个参数为测试图片路径  第二个参数为模型路径
if __name__ == '__main__':
    prediect('./results/main/2020-09-22_15 59 12/0/transient/best.ckpt')
