import numpy as np
import time
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tfs
import random
# import mean_teacher.data as data
import scipy as sp
import scipy.stats
import os
import math
# img = Image.open('./data-local/images/ruxian/old_test/2/112_benign.bmp')
#
# img_aug = tfs.Compose([
#         # tfs.Resize((32, 32)),  # 随机放缩100
#         # data.RandomTranslateWithReflect(4),
#         # tfs.RandomHorizontalFlip(),  # 随机水平翻转
#         # tfs.RandomVerticalFlip(),  # 随机竖直翻转
#         # tfs.RandomRotation(6),  # 随机旋转在（-15， +15）
#         tfs.ColorJitter(brightness=(1.3, 1.3), contrast=(1.6, 1.6)),  # 修改图片的亮度、对比度、饱和度
#     ])
# new_img = img_aug(img)
# # new_img = tfs.ToPILImage()(new_img)
# new_img.save('./data-local/images/Test999/112_new_benign10.bmp')

x = torch.Tensor([[1, 1, 1], [1, 2, 3], [1, 1, 1]])
t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(x)
print(t)
x = t
print(x)
print(t)

# a = list(range(-2, 3))
# b = torch.tensor(a)
# b = b.cuda()
# c = torch.zeros(5)
# c = c.cuda()
# print(b)
# print(c)
# for i in range(b.size(0)):
#     if b[i] == -1:
#         c[i] = -1
#     else:
#         c[i] = b[i] // 2
# print(c.int())
