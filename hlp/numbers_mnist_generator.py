#!/usr/bin/env python
__author__ = 'solivr'

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import csv
import scipy.misc
from tqdm import tqdm
import random
import argparse


def generate_random_image_numbers(mnist_dir, dataset_type, output_dir, csv_filename, n_numbers):
    '''
    用于根据mnist数据集随机生成长度不定的图片
    :param mnist_dir: mnist数据集存放地址
    :param dataset:生成数据集的类型，train(default),val,test
    :param output_dir:数据集保存地址
    :param csv_filename:保存图片地址和标签的文件路径
    :param n_numbers:生成多少张图片，默认1000张
    :return: None
    '''
    # 读取mnist数据集
    mnist = input_data.read_data_sets(mnist_dir, one_hot=False)
    # 构造输出目录
    img_output_dir = os.path.join(output_dir, dataset_type)
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)
    # 设定数据集类型
    if dataset_type == 'train':
        dataset = mnist.train
    elif dataset_type == 'val':
        dataset = mnist.validation
    elif dataset_type == 'test':
        dataset = mnist.test

    list_paths = list()
    list_labels = list()

    for i in tqdm(range(n_numbers), total=n_numbers):
        # 在3,8之间随机生成一个数
        n_digits = random.randint(3, 8)
        # 获取随机数这么多张图片
        digits, labels = dataset.next_batch(n_digits)
        # 图片reshape为 28x28
        square_digits = np.reshape(digits, [-1, 28, 28])
        # 图像像素变为0-255
        square_digits = -(square_digits - 1) * 255
        # 图像前后都去掉四个像素的宽度，然后水平堆叠在一起
        stacked_number = np.hstack(square_digits[:, :, 4:-4])
        stacked_label = ''.join(map(str, labels))
        # chans3 = np.dstack([stacked_number]*3)

        # 保存图片
        img_filename = '{:09}_{}.jpg'.format(i, stacked_label)
        img_path = os.path.join(img_output_dir, img_filename)
        scipy.misc.imsave(img_path, stacked_number)

        # Add to list of paths and list of labels
        list_paths.append(img_filename)
        list_labels.append(stacked_label)

    root = '/data1/zhoujun/tf-crnn/data/' + dataset_type
    csv_path = os.path.join(output_dir, csv_filename)
    with open(csv_path, 'w') as csvfile:
        for i in tqdm(range(len(list_paths)), total=len(list_paths)):
            csvwriter = csv.writer(csvfile, delimiter=' ')
            csvwriter.writerow([os.path.join(root, list_paths[i]), list_labels[i]])


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-md', '--mnist_dir', type=str, help='Directory for MNIST data', default='./MNIST_data')
    # parser.add_argument('-d', '--dataset', type=str, help='Dataset wanted (train, test, validation)', default='train')
    # parser.add_argument('-csv', '--csv_filename', type=str, help='CSV filename to output paths and labels')
    # parser.add_argument('-od', '--output_dir', type=str, help='Directory to output images and csv files',
    #                     default='./output_numbers')
    # parser.add_argument('-n', '--n_samples', type=int, help='Desired numbers of generated samples', default=1000)

    # args = parser.parse_args()
    # generate_random_image_numbers(args.mnist_dir, args.dataset, args.output_dir, args.csv_filename, args.n_samples)
    generate_random_image_numbers(mnist_dir='./MNIST_data',dataset_type='test',output_dir='./data',csv_filename='test.csv',n_numbers=10)
