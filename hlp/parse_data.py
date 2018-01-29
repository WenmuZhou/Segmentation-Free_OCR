# -*- coding: utf-8 -*-
# @Time    : 2017/12/4 15:04
# @Author  : zhoujun

import os
import csv


def parse_car_data(data_path):
    train_file = data_path + '/gtest.txt'
    train_file_w = open(data_path + '/test.csv', encoding='utf8', mode='w')
    csvwriter = csv.writer(train_file_w, delimiter=' ')
    with open(train_file) as train_file_read:
        for line in train_file_read.readlines():
            line = line.strip().split(' ')
            file_path = data_path + '/test/oas_' + line[0] + '.jpg'
            label = line[1]
            if check_file(file_path):
                csvwriter.writerow([file_path, label])
            else:
                print(file_path, '不存在')
    train_file_w.close()


def parse_cvl_data(data_path):
    files = os.listdir(data_path)
    train_file_w = open(data_path + '/test.csv', encoding='utf8', mode='w')
    csvwriter = csv.writer(train_file_w, delimiter=' ')
    for file in files:
        file_path = data_path + '/' + file
        # label = file.
        (shotname, _) = os.path.splitext(file)
        label = shotname.split('_')[-1]
        if check_file(file_path):
            csvwriter.writerow([file_path, label])
        else:
            print(file_path, '不存在')
    train_file_w.close()


def check_file(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)


def remove_resize(path):
    files = os.listdir(path)
    for file in files:
        if file.endswith('r120.jpg'):
            os.remove(os.path.join(path, file))


if __name__ == '__main__':
    data_path = '/data/zhoujun/data/CVL/test'
    parse_cvl_data(data_path)
    # remove_resize(data_path)
