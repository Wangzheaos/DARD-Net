# coding:utf-8
import os
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import pickle

"""
    在训练前先运行该函数获得数据的均值和标准差
"""


class Dataloader():
    def __init__(self, isize, dataroot):
        # 训练，验证，测试数据集文件夹名
        self.isize = isize
        self.dataroot = dataroot
        self.dirs = ['train', 'test', 'val', 'center_v']

        self.means = [0, 0, 0]
        self.stdevs = [0, 0, 0]

        self.transform = transforms.Compose([transforms.Resize(self.isize),
                                             transforms.CenterCrop(self.isize),
                                             transforms.ToTensor(),  # 数据值从[0,255]范围转为[0,1]，相当于除以255操作
                                             # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
                                             ])

        # 因为这里使用的是ImageFolder，按文件夹给数据分类，一个文件夹为一类，label会自动标注好
        self.dataset = {x: ImageFolder(os.path.join(self.dataroot, x), self.transform) for x in self.dirs}

    def get_mean_std(self, type, mean_std_path):
        """
        计算数据集的均值和标准差
        :param type: 使用的是那个数据集的数据，有'train', 'test', 'testing'
        :param mean_std_path: 计算出来的均值和标准差存储的文件
        :return:
        """
        num_imgs = len(self.dataset[type])
        for data in self.dataset[type]:
            img = data[0]
            for i in range(3):
                # 一个通道的均值和标准差
                self.means[i] += img[i, :, :].mean()
                self.stdevs[i] += img[i, :, :].std()

        self.means = np.asarray(self.means) / num_imgs
        self.stdevs = np.asarray(self.stdevs) / num_imgs

        print("{} : normMean = {}".format(type, self.means))
        print("{} : normstdevs = {}".format(type, self.stdevs))

        # 将得到的均值和标准差写到文件中，之后就能够从中读取
        with open(mean_std_path, 'wb') as f:
            pickle.dump(self.means, f)
            pickle.dump(self.stdevs, f)
            print('pickle done')


if __name__ == '__main__':
    isize = 32
    dataroot = './data-local/images/ruxian/'
    dataloader = Dataloader(isize, dataroot)
    for x in dataloader.dirs:
        mean_std_path = 'mean_std_value_' + x + '.pkl'
        dataloader.get_mean_std(x, mean_std_path)