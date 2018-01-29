# -*- coding: utf-8 -*-
# @Time    : 2017/12/21 10:47
# @Author  : zhoujun

# !/usr/bin/env python
import tensorflow as tf
from .resnet import batch_norm_relu

def inception_v1(inputs,is_training, conv11_size, conv33_11_size, conv33_size,
                 conv55_11_size, conv55_size, pool11_size):
    with tf.variable_scope("conv_1x1"):
        conv11 = tf.layers.conv2d(inputs=inputs, filters=conv11_size, kernel_size=[1, 1], strides=1, padding="SAME")
        conv11 = batch_norm_relu(conv11,is_training)

    with tf.variable_scope("conv_3x3"):
        conv33_11 = tf.layers.conv2d(inputs=inputs, filters=conv33_11_size, kernel_size=[1, 1], strides=1,
                                     padding="SAME")
        conv33_11 = batch_norm_relu(conv33_11,is_training)

        conv33 = tf.layers.conv2d(inputs=conv33_11, filters=conv33_size, kernel_size=[3, 3], strides=1, padding="SAME")
        conv33 = batch_norm_relu(conv33,is_training)

    with tf.variable_scope("conv_5x5"):
        conv55_11 = tf.layers.conv2d(inputs=inputs, filters=conv55_11_size, kernel_size=[1, 1], strides=1,
                                     padding="SAME")
        conv55_11 = batch_norm_relu(conv55_11,is_training)

        conv55 = tf.layers.conv2d(inputs=conv55_11, filters=conv55_size, kernel_size=[5, 5], strides=1, padding="SAME")
        conv55 = batch_norm_relu(conv55,is_training)

    with tf.variable_scope("pool_proj"):
        pool_proj = tf.layers.max_pooling2d(inputs=inputs, pool_size=[3, 3], strides=1, padding="SAME")
        pool11 = tf.layers.conv2d(inputs=pool_proj, filters=pool11_size, kernel_size=[1, 1], strides=1, padding="SAME")
        pool11 = batch_norm_relu(pool11,is_training)

    return tf.concat([conv11, conv33, conv55, pool11], 3)


def inception_v2(inputs, is_training,conv11_size, conv33_11_size, conv33_size, conv55_11_size, conv55_size1, conv55_size,
                 pool11_size):
    with tf.variable_scope("conv_1x1"):
        conv11 = tf.layers.conv2d(inputs=inputs, filters=conv11_size, kernel_size=[1, 1], strides=1, padding="SAME")
        conv11 = batch_norm_relu(conv11,is_training)
    with tf.variable_scope("conv_3x3"):
        conv33_11 = tf.layers.conv2d(inputs=inputs, filters=conv33_11_size, kernel_size=[1, 1], strides=1,
                                     padding="SAME")
        conv33_11 = batch_norm_relu(conv33_11,is_training)
        conv33 = tf.layers.conv2d(inputs=conv33_11, filters=conv33_size, kernel_size=[3, 3], strides=1, padding="SAME")
        conv33 = batch_norm_relu(conv33,is_training)
    with tf.variable_scope("conv_3x3x2"):
        conv55_11 = tf.layers.conv2d(inputs=inputs, filters=conv55_11_size, kernel_size=[1, 1], strides=1,
                                     padding="SAME")
        conv55_11 = batch_norm_relu(conv55_11,is_training)

        conv55 = tf.layers.conv2d(inputs=conv55_11, filters=conv55_size1, kernel_size=[3, 3], strides=1, padding="SAME")
        conv55 = batch_norm_relu(conv55,is_training)
        conv55 = tf.layers.conv2d(inputs=conv55, filters=conv55_size, kernel_size=[3, 3], strides=1, padding="SAME")
        conv55 = batch_norm_relu(conv55,is_training)

    with tf.variable_scope("pool_proj"):
        pool_proj = tf.layers.max_pooling2d(inputs=inputs, pool_size=[3, 3], strides=1, padding="SAME")
        pool11 = tf.layers.conv2d(inputs=pool_proj, filters=pool11_size, kernel_size=[1, 1], strides=1, padding="SAME")
        pool11 = batch_norm_relu(pool11,is_training)

    return tf.concat([conv11, conv33, conv55, pool11], 3)


def googlenet(input_imgs: tf.Tensor, is_training: bool, summaries: bool = True) -> tf.Tensor:
    '''
    Implementation of https://arxiv.org/pdf/1409.4842.pdf
    '''
    inputs = input_imgs
    if inputs.shape[-1] not in [1, 3]:
        raise NotImplementedError

    with tf.variable_scope("googlenet"):
        # (128, 32, 304, 3)->(128, 8, 76, 64)
        with tf.variable_scope("inputs"):
            inputs = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=3, strides=2, padding="SAME",
                                      name='conv0')
            inputs = batch_norm_relu(inputs,is_training)
            print(inputs.get_shape)
            inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=2, padding="SAME", name='pool0')
            print(inputs.get_shape)
        # (128, 8, 76, 64)->(128, 8, 76, 128)
        with tf.variable_scope("inception_3a"):
            inputs = inception_v2(inputs,is_training, 32, 48, 64, 8, 12, 16, 16)
            print(inputs.get_shape)
        # (128, 8, 76, 64)->(128, 8, 76, 256)
        with tf.variable_scope("inception_3b"):
            inputs = inception_v2(inputs,is_training, 64, 64, 128, 16, 24, 32, 32)
            print(inputs.get_shape)
        # (128, 8, 76, 256)->(128, 8, 76, 480)
        with tf.variable_scope("inception_3c"):
            inputs = inception_v2(inputs,is_training, 128, 128, 192, 32, 48, 96, 64)
            print(inputs.get_shape)

        # (128, 8, 76, 480)->(128, 4, 76, 480)
        inputs = tf.layers.max_pooling2d(inputs, pool_size=3, strides=[2, 1], padding="SAME", name='pool2')
        print(inputs.get_shape)
        # (128, 4, 76, 480)->(128, 4, 76, 512)
        with tf.variable_scope("inception_3d"):
            inputs = inception_v2(inputs,is_training, 128, 128, 256, 32, 48, 64, 64)
            print(inputs.get_shape)
        # (128, 4, 76, 480)->(128, 2, 76, 512)
        inputs = tf.layers.max_pooling2d(inputs, pool_size=3, strides=[2, 1], padding="SAME", name='pool3')
        print(inputs.get_shape)

        inputs = tf.layers.conv2d(inputs=inputs, filters=512, kernel_size=[2, 2], strides=1, padding='VALID',
                                  use_bias=False, name='last_conv7')
        inputs = batch_norm_relu(inputs, is_training)
        print(inputs.get_shape)
        # reshape 特征图
        with tf.variable_scope('Reshaping_cnn'):
            shape = inputs.get_shape().as_list()  # [batch, height, width, features]
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3],
                                  name='transposed')  # [batch, width, height, features] 128*1*75*512 -> 128*75*1*512
            inputs = tf.reshape(inputs, [shape[0], -1, shape[1] * shape[3]],
                                name='reshaped')  # [batch, width, height x features] 128*75*1*512 -> 128*75*512
    return inputs


def googlenet1(input_imgs: tf.Tensor, is_training: bool, summaries: bool = True) -> tf.Tensor:
    '''
    Implementation of https://arxiv.org/pdf/1409.4842.pdf
    '''
    inputs = input_imgs
    if inputs.shape[-1] not in [1, 3]:
        raise NotImplementedError

    with tf.variable_scope("googlenet"):
        with tf.variable_scope("inputs"):
            inputs = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[7, 7], strides=2, padding="SAME",
                                      name='conv0')
            inputs = batch_norm_relu(inputs, is_training)
            print(inputs.get_shape)
            inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[3, 3], strides=[2, 1], padding="SAME",
                                             name='pool0')
            print(inputs.get_shape)
            inputs = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[1, 1], strides=1, padding="SAME",
                                      name='conv1_a')
            inputs = batch_norm_relu(inputs, is_training)
            print(inputs.get_shape)
            inputs = tf.layers.conv2d(inputs=inputs, filters=192, kernel_size=[3, 3], strides=1, padding="SAME",
                                      name='conv1_b')
            inputs = batch_norm_relu(inputs, is_training)
            print(inputs.get_shape)
            inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[3, 3], strides=[2, 1], padding="SAME",
                                             name='pool1')
            print(inputs.get_shape)

        with tf.variable_scope("inception_3a"):
            inputs = inception_v1(inputs,is_training, 64, 96, 128, 16, 32, 32)
            print(inputs.get_shape)
        with tf.variable_scope("inception_3b"):
            inputs = inception_v1(inputs,is_training, 128, 128, 192, 32, 96, 64)
            print(inputs.get_shape)

        inputs = tf.layers.max_pooling2d(inputs, pool_size=[3, 3], strides=[2, 1], padding="SAME", name='pool2')
        print(inputs.get_shape)
        with tf.variable_scope("inception_4a"):
            inputs = inception_v1(inputs,is_training, 192, 96, 208, 16, 48, 64)
            print(inputs.get_shape)

        with tf.variable_scope("inception_4b"):
            inputs = inception_v1(inputs,is_training, 160, 112, 224, 24, 64, 64)
            print(inputs.get_shape)

        with tf.variable_scope("inception_4c"):
            inputs = inception_v1(inputs,is_training, 128, 128, 256, 24, 64, 64)
            print(inputs.get_shape)

        with tf.variable_scope("inception_4d"):
            inputs = inception_v1(inputs,is_training, 112, 144, 288, 32, 64, 64)
            print(inputs.get_shape)

        with tf.variable_scope("inception_4e"):
            inputs = inception_v1(inputs,is_training, 256, 160, 320, 32, 128, 128)
            print(inputs.get_shape)
        inputs = tf.layers.max_pooling2d(inputs, pool_size=[3, 3], strides=[1, 2], padding="SAME", name='pool3')
        print(inputs.get_shape)
        with tf.variable_scope("inception_5a"):
            inputs = inception_v1(inputs,is_training, 256, 160, 320, 32, 128, 128)
            print(inputs.get_shape)
        with tf.variable_scope("inception_5b"):
            inputs = inception_v1(inputs,is_training, 384, 192, 384, 48, 128,
                                  128)
            print(inputs.get_shape)

        inputs = tf.layers.conv2d(inputs=inputs, filters=512, kernel_size=[2, 2], strides=1, padding='VALID',
                                  use_bias=False, name='last_conv7')
        inputs = batch_norm_relu(inputs, is_training)
        print(inputs.get_shape)
        # reshape 特征图
        with tf.variable_scope('Reshaping_cnn'):
            shape = inputs.get_shape().as_list()  # [batch, height, width, features]
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3],
                                  name='transposed')  # [batch, width, height, features] 128*1*75*512 -> 128*75*1*512
            inputs = tf.reshape(inputs, [shape[0], -1, shape[1] * shape[3]],
                                name='reshaped')  # [batch, width, height x features] 128*75*1*512 -> 128*75*512
    return inputs
