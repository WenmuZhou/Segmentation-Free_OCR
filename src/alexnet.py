# -*- coding: utf-8 -*-
# @Time    : 2017/12/13 19:31
# @Author  : zhoujun

import tensorflow as tf
from .resnet import batch_norm_relu

def alexnet(input_imgs: tf.Tensor, is_training: bool, summaries: bool = True) -> tf.Tensor:
    inputs = input_imgs
    if inputs.shape[-1] not in [1, 3]:
        raise NotImplementedError

    with tf.variable_scope('AlexNet'):
        # conv1 128*32*304*3 -> 128*6*74*64
        inputs = tf.layers.conv2d(inputs=inputs, filters=96, padding='SAME', kernel_size=11, strides=4,
                                  activation=tf.nn.relu, name='conv1')
        inputs = batch_norm_relu(inputs, is_training)
        print(inputs.get_shape)

        inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=[2,1], padding='SAME', name='pool1')
        print(inputs.get_shape)

        inputs = tf.layers.conv2d(inputs=inputs, filters=256, padding='SAME', kernel_size=5, strides=1,
                                  activation=tf.nn.relu, name='conv2')
        inputs = batch_norm_relu(inputs, is_training)
        print(inputs.get_shape)

        inputs = tf.layers.max_pooling2d(inputs, pool_size=3, strides=[2,1], padding='SAME', name='pool2')
        print(inputs.get_shape)

        inputs = tf.layers.conv2d(inputs=inputs, filters=384, padding='SAME', kernel_size=3, strides=1,
                                  activation=tf.nn.relu, name='conv3')
        inputs = batch_norm_relu(inputs, is_training)
        print(inputs.get_shape)

        inputs = tf.layers.conv2d(inputs=inputs, filters=384, padding='SAME', kernel_size=3, strides=1,
                                  activation=tf.nn.relu, name='conv4')
        inputs = batch_norm_relu(inputs, is_training)
        print(inputs.get_shape)

        inputs = tf.layers.conv2d(inputs=inputs, filters=256, padding='SAME', kernel_size=3, strides=1,
                                  activation=tf.nn.relu, name='conv5')
        inputs = batch_norm_relu(inputs, is_training)
        print(inputs.get_shape)

        inputs = tf.layers.max_pooling2d(inputs, pool_size=3,  padding='SAME', strides=[2,1], name='pool3')
        print(inputs.get_shape)

        inputs = tf.layers.conv2d(inputs=inputs, filters=512, kernel_size=[2, 2], strides=1, padding='VALID',
                                  use_bias=False, kernel_initializer=tf.variance_scaling_initializer(), name='conv7')
        inputs = batch_norm_relu(inputs, is_training)
        print(inputs.get_shape)

        with tf.variable_scope('Reshaping_cnn'):
            shape = inputs.get_shape().as_list()  # [batch, height, width, features]
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3],
                                  name='transposed')  # [batch, width, height, features] 128*1*75*512 -> 128*75*1*512
            inputs = tf.reshape(inputs, [shape[0], -1, shape[1] * shape[3]],
                                name='reshaped')  # [batch, width, height x features] 128*75*1*512 -> 128*75*512
            print(inputs.get_shape)
        return inputs
