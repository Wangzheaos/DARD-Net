import numpy as np
import math
import torch
from torch.nn import functional as F
from torch.autograd import Variable


def softmax_mse_loss1(input_logits, target_logits):
    """细粒度的consistency_loss"""
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


def softmax_mse_loss2(input_logits, target_logits):
    """粗粒度的consistency_loss"""
    assert input_logits.size() == target_logits.size()
    num_classes = input_logits.size()[1]

    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    input_t = torch.zeros([input_logits.size()[0], 3]).cuda()
    for i in range(0, int(num_classes / 3)):
        input_t[:, 0] = input_t[:, 0] + input_softmax[:, i]
    for i in range(int(num_classes / 3), int(num_classes / 3) * 2):
        input_t[:, 1] = input_t[:, 1] + input_softmax[:, i]
    for i in range(int(num_classes / 3) * 2, num_classes):
        input_t[:, 2] = input_t[:, 2] + input_softmax[:, i]

    target_t = torch.zeros([target_logits.size()[0], 3]).cuda()
    for i in range(0, int(num_classes / 3)):
        target_t[:, 0] = target_t[:, 0] + target_softmax[:, i]
    for i in range(int(num_classes / 3), int(num_classes / 3) * 2):
        target_t[:, 1] = target_t[:, 1] + target_softmax[:, i]
    for i in range(int(num_classes / 3) * 2, num_classes):
        target_t[:, 2] = target_t[:, 2] + target_softmax[:, i]

    return F.mse_loss(input_t, target_t, reduction='sum') / num_classes


def compute_center_loss(features, centers, targets):
    featuresT = features.view(features.size(0), -1)
    target_centers = centers[targets]
    target_centersT = target_centers.view(target_centers.size(0), -1)
    loss_fn = torch.nn.MSELoss()
    # num = features.size()[0] + features.size()[1] + features.size()[2] + features.size()[3]
    return loss_fn(featuresT, target_centersT)


def get_center_delta(features, centers, targets, alpha):
    # implementation equation (4) in the center-loss paper
    device = torch.device("cuda")
    # features = features.view(features.size(0), -1)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)

    uni_targets = uni_targets.to(device)
    indices = indices.to(device)

    delta_centers = torch.zeros(
        uni_targets.size(0), delta_centers.size(1), delta_centers.size(2), delta_centers.size(3)
    ).to(device).index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)
    same_class_feature_count = same_class_feature_count.unsqueeze(2)
    same_class_feature_count = same_class_feature_count.unsqueeze(3)

    delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result


def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='sum')


def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2) ** 2) / num_classes
