# -*- coding: utf-8 -*-
# @Time    : 2017/11/27 16:00
# @Author  : zhoujun
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def getFileName(path):
    return path.split('/')[-1]


def readLines(file_path):
    with open(file_path, 'r') as T:
        lines = T.readlines()
    return lines


def split_lines(src):
    lines = src
    label_record = {}
    for line in lines:
        name = line.split(' ')[0]
        label = line.split(' ')[1]
        label = label.split('\n')[0]
        label_record[name] = label
    return label_record


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def recordsCreater(label_file, dst_records):
    writer = tf.python_io.TFRecordWriter(dst_records)

    lines = readLines(label_file)
    label_record = split_lines(lines)
    index = 0

    pbar = tqdm(total=len(lines))
    for file_path, label in label_record.items():
        index = index + 1
        img = Image.open(file_path)
        image_raw = img.tobytes()

        cols = img.size[0]
        rows = img.size[1]
        depth = 3 if img.mode is 'RGB' else 1

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _bytes_feature(bytes(label, encoding = "utf8")  ),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
        writer.flush()
        pbar.update(1)
    print("done!")
    writer.close()
    pbar.close()

# 读取二进制数据
def recordsReader(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    image = tf.reshape(image, [height,width,depth])
    label = tf.cast(features['label'], tf.string)
    return image, label

def test_reader(recordsFile):
    image, label = recordsReader(recordsFile)
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1):
            image, label = sess.run([image, label])  # 在会话中取出image和label
            print(label)
            # img = Image.fromarray(example, 'RGB')  # 如果img是RGB图像
            # img = Image.fromarray(example)
            #
            # img.save('./' + '_'+'Label_' + str(l) + '.jpg')  # 存下图片
            Image._show(Image.fromarray(image))
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    test_label_file, test_dst_records = "E:\\val1.csv", "E:\\val1.tfrecords"
    # train_label_file, train_dst_records = "../MNIST_data/mnist_train/train.txt", "../MNIST_data/mnist_train.tfrecords"
    # recordsCreater(test_label_file, test_dst_records)
    # recordsCreater(test_label_file, test_dst_records)
    test_reader(test_dst_records)