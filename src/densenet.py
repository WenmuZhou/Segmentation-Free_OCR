# -*- coding: utf-8 -*-
# @Time    : 2017/12/3 13:01
# @Author  : zhoujun

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

dropout_rate = 0.2

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network


def Batch_Normalization(x, training, scope):
    inputs = tf.layers.batch_normalization(inputs=x, training=training)
    # return tf.cond(training,
    #                    lambda : batch_norm(inputs=x, is_training=training, reuse=None),
    #                    lambda : batch_norm(inputs=x, is_training=training, reuse=True))
    return inputs

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)


class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def transition_layer_x(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=[2,1],padding='SAME')

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))
            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            return x

    def Dense_net(self, input_x):
        with tf.name_scope('DenseNet'):
            # (128, 32, 304, 3)  -> (128, 16, 152, 48)
            x = conv_layer(input_x, filter=2 * self.filters, kernel=3, stride=2, layer_name='conv0')
            print(x.get_shape)

            # (128, 16, 152, 48)  -> (128, 16, 152, 24)
            x = self.dense_block(input_x=x, nb_layers=2, layer_name='dense_1')
            print(x.get_shape)
            # (128, 16, 152, 24) -> (128, 8, 76, 24)
            x = self.transition_layer(x, scope='trans_1')
            print(x.get_shape)

            # (128, 8, 76, 24)  -> (128, 8, 76, 24)
            x = self.dense_block(input_x=x, nb_layers=2, layer_name='dense_2')
            print(x.get_shape)
            # (128, 8, 76, 24)  -> (128, 4, 76, 24)
            x = self.transition_layer_x(x, scope='trans_2')
            print(x.get_shape)

            # (128, 4, 76, 24)  -> (128, 4, 76, 24)
            x = self.dense_block(input_x=x, nb_layers=2, layer_name='dense_3')
            print(x.get_shape)
            # (128, 4, 76, 24)  -> (128, 2, 76, 24)
            x = self.transition_layer_x(x, scope='trans_3')
            print(x.get_shape)

            # (128, 2, 76, 24)  -> (128, 2, 76, 24)
            x = self.dense_block(input_x=x, nb_layers=2, layer_name='dense_4')
            print(x.get_shape)
            # (128, 1, 76, 24)  -> (128, 1, 76, 24)
            x = self.transition_layer_x(x, scope='trans_4')
            print(x.get_shape)

            # (128, 4, 76, 24)  -> (128, 2, 76, 24)
            x = self.dense_block(input_x=x, nb_layers=2, layer_name='dense_final')
            print(x.get_shape)

            # reshape 特征图
            with tf.variable_scope('Reshaping_cnn'):
                shape = x.get_shape().as_list()  # [batch, height, width, features]
                x = tf.transpose(x, perm=[0, 2, 1, 3],
                                      name='transposed')  # [batch, width, height, features] 128*1*75*512 -> 128*75*1*512
                x = tf.reshape(x, [shape[0], -1, shape[1] * shape[3]],
                                    name='reshaped')  # [batch, width, height x features] 128*75*1*512 -> 128*75*512
            print(x.get_shape)
        return x

def dense_net(input_imgs: tf.Tensor, is_training: bool, summaries: bool = True) -> tf.Tensor:
    inputs = DenseNet(x=input_imgs, nb_blocks=2, filters=48, training=is_training).model
    return inputs