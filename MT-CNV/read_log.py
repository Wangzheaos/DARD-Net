# -*- coding: utf-8 -*-#
# -------------------------------------------------------------------------------
# @Name:         read_log.py
# @Description:  
# @Author:       WangRf
# @Date:         2020/5/1 0001
# -------------------------------------------------------------------------------
from mean_teacher.run_context import TrainLog
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


directory = "./results/CNV_center_5_5-3-LU/2021-04-16_09 31 59/384_0/"
Name = ['training', 'validation','ema_validation']
for name in Name:
    path = os.path.join(directory, name + '.html')
    print(path)
    df = pd.read_html(path)[0]

    plt.figure()
    axes_subplot = df.plot(subplots=True, x='step', figsize=(12, 30))

# axes_subplot2 = df['cons_loss_1'].plot(x='step', figsize=(12, 12), color='orange', label='con_loss_1')
# axes_subplot3 = df['cons_loss_2'].plot(x='step', figsize=(12, 12), color='blue', label='con_loss_2')
# plt.xlabel('step')
# plt.ylabel('loss value')
# plt.legend()

#
# data = np.array([[0.81, 0.50], [0.84, 0.62], [0.80, 0.61]])
# ddf = pd.DataFrame(data, index=['MT—ResNet', 'LGCRMT—ResNet', 'LGCRMT—AlexNet'], columns=['benign', 'malignant'])
# plt.figure()
# axes_subplot = ddf.plot(kind='bar', figsize=(8, 10), rot=90)
# plt.setp(axes_subplot.get_xticklabels(), rotation=0)
#
# plt.ylabel("F1-score")

    img_name = os.path.join(directory, name + '.png')
    plt.title(name)
    plt.savefig(img_name)
    # plt.show()
