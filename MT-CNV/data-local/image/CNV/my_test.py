import os

fileList = os.listdir(r"./2")
fileList.sort(key=lambda x: int(x[:9]))
# 输出此文件夹中包含的文件名称
# print("修改前：" + str(fileList)[1])
# 得到进程当前工作目录
currentpath = os.getcwd()
# 将当前工作目录修改为待修改文件夹的位置
os.chdir(r"./2")

i = 1
# 遍历文件夹中所有文件
for fileName in fileList:
    file = os.path.splitext(fileName)  # 将文件名与后缀分割开
    # 文件重新命名
    os.rename(fileName, str(i)+'_2'+file[1])
    i = i+1
# 改回程序运行前的工作目录
os.chdir(currentpath)
