import os
import sys
from PIL import Image
from torchvision import transforms as tfs
import mean_teacher.data as data

# 载入图片
# img = Image.open('./bad_data2/1_1.bmp')

img_aug = tfs.Compose([
    tfs.Resize((80, 80), Image.ANTIALIAS),  # 随机放缩84
    # data.RandomTranslateWithReflect(4),  # 随机位移
    # tfs.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    # tfs.RandomVerticalFlip(p=0.5),  # 随机竖直翻转
    # tfs.RandomRotation(6),  # 随机旋转在（-6， +6）
    # tfs.ColorJitter(brightness=(0.9, 1.3), contrast=(1, 1.6)),  # 修改图片的亮度、对比度、饱和度
])

# for j in range(1, 7):
#     # filepath = r'./old_unlabel/' + str(j)
#     filepath = r'./old_unlabel/CNV'
#     destpath = r'./train_6_L64+U/' + str(j)
#     pathDir = os.listdir(filepath)  # 列出文件路径中的所有路径或文件
#     i = len([files for files in os.listdir(destpath)])
#     for n in range(1000):
#         if i == 64:
#             break
#         for allDir in pathDir:
#             old = os.path.join(filepath, allDir)
#             img = Image.open(old)
#             new_img = img_aug(img)
#             new = os.path.join(destpath, str(i) + '_' + str(j) + ".bmp")
#             new_img.save(new)
#             i = i + 1
#             if i == 64:
#                 break
#     print(destpath)
#     print(i)

# for i in range(1, 7):
#     for j in range(0, 64):
#         print(str(j) + '_' + str(i) + ".bmp " + str(i))


filepath = r'./old_unlabel/CNV'
pathDir = os.listdir(filepath)
print(pathDir)


# filepath = r'./old_unlabel/CNV'
# destpath = r'./train_6_L64+U/'
# pathDir = os.listdir(filepath)  # 列出文件路径中的所有路径或文件
# j = 1
# i = len([files for files in os.listdir(destpath + str(j))])
# for allDir in pathDir:
#     old = os.path.join(filepath, allDir)
#     img = Image.open(old)
#     new_img = img_aug(img)
#     new = os.path.join(destpath + str(j), str(i) + '_' + str(j) + ".bmp")
#     new_img.save(new)
#     i = i + 1
#     if i == 6000:
#         j = j + 1
#         i = len([files for files in os.listdir(destpath + str(j))])
#     if j == 7:
#         break


# for j in range(3, 4):
#     filepath = r'./old_train_6/' + str(j)
#     destpath = r'./haha/'
#     pathDir = os.listdir(filepath)  # 列出文件路径中的所有路径或文件
#     # print(pathDir)
#     i = len([files for files in os.listdir(destpath)])
#     for allDir in pathDir:
#
#         if allDir != '33_1.jpg':
#             continue
#         old = os.path.join(filepath, allDir)
#         img = Image.open(old)
#         new_img = img_aug(img)
#         new = os.path.join(destpath, str(i) + '_' + str(j) + ".bmp")
#         new_img.save(new)
#         i = i + 1
#     print(destpath)
#     print(i)
