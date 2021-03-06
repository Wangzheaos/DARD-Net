import re
import argparse
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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import torchvision.transforms as transforms

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *


from scipy import stats
from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import confusion_matrix, classification_report


# def pil_loader(path):
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('L')


# dataset = torchvision.datasets.ImageFolder('data-local/images/ruxian/train', data.TransformTwice(transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.Resize((32,32)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.4914],[0.2470])
#     ])), None, pil_loader)

# with open('data-local/labels/ruxian/800_balanced_labels/00.txt') as f:
#     labels = dict(line.split(' ') for line in f.read().splitlines())
# labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)
# batch_sampler = data.TwoStreamBatchSampler(
#     unlabeled_idxs, labeled_idxs, 64, 16)

# train_loader = torch.utils.data.DataLoader(dataset,
#                                                batch_sampler=batch_sampler,
#                                                num_workers=4,
#                                                pin_memory=True)
# minibatch_size = 64
# labeled_minibatch_size = 16
# for i, ((input,ema_input), target) in enumerate(train_loader):
#     # LPA
#     img_w = list(input.size())[2]
#     X = input.squeeze(1).view(minibatch_size, img_w * img_w).numpy()
#     y = target.numpy()
#     n_total_samples = minibatch_size
#     n_labeled_points = labeled_minibatch_size
#     indices = np.arange(n_total_samples)
#     # ???????????????
#     unlabeled_set = indices[:minibatch_size - n_labeled_points]
#     # ??????????????????
#     lp_model = LabelSpreading(kernel='knn', alpha=0.7)
#     lp_model.fit(X, y)
#     predicted_labels = lp_model.transduction_[unlabeled_set]
#     print("predicted_labels:", predicted_labels)







digits = datasets.load_digits()  # ????????????????????? ?????????
rng = np.random.RandomState(0)  # ??????????????????
indices = np.arange(len(digits.data))  # ??????
rng.shuffle(indices)

X = digits.data[indices[:330]]
y = digits.target[indices[:330]]
images = digits.images[indices[:330]]

print("X:\n",X)
print("y:\n",y)

n_total_samples = len(y)
n_labeled_points = 30

indices = np.arange(n_total_samples)

# ???????????????
unlabeled_set = indices[n_labeled_points:]

# ??? y_train ?????????????????????????????? -1??????????????????
y_train = np.copy(y)
y_train[unlabeled_set] = -1

# ??????????????????
lp_model = LabelSpreading(kernel='knn', alpha=0.7)
lp_model.fit(X, y_train)
predicted_labels = lp_model.transduction_[unlabeled_set]
true_labels = y[unlabeled_set]

cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)

print("Label Spreading model: %d labeled & %d unlabeled points (%d total)" % (
n_labeled_points, n_total_samples - n_labeled_points, n_total_samples))
report = classification_report(true_labels, predicted_labels)
print(report)
print("7:",report['accuracy'])
for key, value in report[7-1].items():
    print(f"{key:10s}:{value:10.2f}")
print("Confusion matrix")
print(cm)

# ???????????????????????????????????????
pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

# ?????????10?????????????????????
uncertainty_index = np.argsort(pred_entropies)[-10:]

# ??????
f = plt.figure(figsize=(7, 5))  # ??????????????????
for index, image_index in enumerate(uncertainty_index):
    image = images[image_index]

    sub = f.add_subplot(2, 5, index + 1)  # ????????????
    sub.imshow(image, cmap=plt.cm.gray_r)
    plt.xticks([])
    plt.yticks([])
    sub.set_title('predict: %i\ntrue: %i' % (lp_model.transduction_[image_index], y[image_index]))

f.suptitle('Learning with small amount of labeled data')
plt.show()  # ??????