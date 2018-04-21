# -*- coding: utf-8 -*-
# @Time    : 2017/7/27 14:05
# @Author  : zhoujun
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_roma = FontProperties(fname=r"C:\Windows\Fonts\times.ttf")

matplotlib.rcParams["font.family"] = "Times New Roman"
# matplotlib.rcParams["font.weight"] = "light"

def get_steo_acc_lose(log_path):
    iter_step = []
    acc = []
    loss = []
    with open(log_path, mode='r', encoding='utf-8') as file:
        for line in file.readlines():
            if line.__contains__('acc'):
                continue
            line = line.strip()
            param = line.split(',')
            iter_step.append(param[0])
            acc.append(param[1])
            loss.append(param[2])
            # print('step: {0},acc = {1},loss = {2}'.format(param[0],param[1],param[2]))
    return iter_step, acc, loss


if __name__ == '__main__':
    root_path = r'Z:\zhoujun\tf-crnn\result'

    labels = ['AlexNet', 'GoogLeNet', 'VGG', 'Proposed']
    file_list = ['alexnet', 'googlenetv1', 'vgg', 'resnet']
    line_type_list = ['-', '-', '-', '-', '-']
    # line_w_list = [2, 2, 2, 2 , 2]
    file_list = [os.path.join(root_path, i + '_acc_loss.csv') for i in file_list]
    colors = [(0, 1, 0), (237 / 255, 125 / 255, 49 / 255), (0, 0, 1), (1, 0, 0)]
    fig = plt.figure(num=1, figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)

    for i in range(len(file_list)):
        if not os.path.exists(file_list[i]):
            continue
        iter_step, acc, loss = get_steo_acc_lose(file_list[i])

        line1 = plt.plot(iter_step, acc, linestyle=line_type_list[i], label=labels[i], color=colors[i])

    plt.legend(loc='lower right')
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    # plt.xlim(0, 120000)
    # plt.ylim(0, 1)
    plt.subplots_adjust(left=0.12, wspace=0.20, hspace=0.20, bottom=0.15, top=0.88)
    plt.savefig(root_path + '/acc.jpg', dpi=600)
    plt.show()
