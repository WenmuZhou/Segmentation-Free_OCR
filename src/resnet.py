from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training):
    '''
    后面跟着一个relu的batch normalization
    :param inputs: 输入
    :param is_training: 是否是训练状态
    :return:
    '''
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=-1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size):
    '''
    对输入在空间维度上进行pads
    :param inputs: 输入
    :param kernel_size: 在卷积或池化中使用的kernel大小
    :return:
    '''

    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, name):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(), name=name)


def building_block(inputs, filters, is_training, projection_shortcut, strides):
    '''
    卷积后带有bn的残差模块
    :param inputs:
    :param filters:
    :param is_training:
    :param projection_shortcut:
    :param strides:
    :return:
    '''
    inputs = batch_norm_relu(inputs, is_training)
    shortcut = inputs

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs, name= 'shortcut')

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides, name= 'conv1')

    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1, name= 'conv2')

    return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut, strides, name):
    '''
    卷积后面带有bn的残差模块
    :param inputs:
    :param filters:
    :param is_training:
    :param projection_shortcut:
    :param strides:
    :return:
    '''
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1, name=name + '/conv1')

    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides, name=name + '/conv2')

    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, name=name + '/conv3')

    return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name, has_short_cut=True):
    with tf.variable_scope(name):
        # Bottleneck blocks end with 4x the number of filters as they start with
        filters_out = 4 * filters if block_fn is bottleneck_block else filters

        def projection_shortcut(inputs,name):
            return conv2d_fixed_padding(
                inputs=inputs, filters=filters_out, kernel_size=1, strides=strides, name=name)

        # 只有第一个block使用projection_shortcut 和 strides
        inputs = block_fn(inputs, filters, is_training, projection_shortcut if has_short_cut else None, strides)

        for _ in range(1, blocks):
            inputs = block_fn(inputs, filters, is_training, None, 1)

    return inputs

def resnet(input_imgs: tf.Tensor, is_training: bool, summaries: bool = True) -> tf.Tensor:
    '''
    对输入图像进行卷积操作
    :param input_imgs: 输入如下 tensor类型 128*32*304*3
    :param is_training: 是否是训练阶段
    :param summaries: 是否显示在tensorboard中
    :return:
    '''
    # 取出输入图像通道
    inputs = input_imgs
    if inputs.shape[-1] not in [1, 3]:
        raise NotImplementedError

    with tf.variable_scope('deep_resnet'):
        # conv1 128*32*304*3 -> 128*16*152*64
        inputs = batch_norm_relu(inputs, is_training)
        inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=3, strides=2, name='conv1')
        print(inputs.get_shape)

        # conv1 128*16*125*64 -> 128*16*152*64
        # inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=1, padding='SAME', name='pool1_3x3')
        # print(inputs.get_shape)

        # block_layer1 128*16*152*64 -> 128*16*152*64
        inputs = block_layer(inputs=inputs, filters=64, block_fn=building_block, blocks=1,
                             strides=1, is_training=is_training, name='block_layer1', has_short_cut=False)
        print(inputs.get_shape)

        # block_layer2 128*16*152*64 -> 128*16*152*128
        inputs = block_layer(inputs=inputs, filters=128, block_fn=building_block, blocks=1,
                             strides=1, is_training=is_training, name='block_layer2')
        print(inputs.get_shape)

        # block_layer3 128*16*152*128 -> 128*8*76*256
        inputs = block_layer(inputs=inputs, filters=256, block_fn=building_block, blocks=1,
                             strides=2, is_training=is_training, name='block_layer3')
        print(inputs.get_shape)

        # pool2_1x2  128*8*76*128 -> 128*4*38*256
        inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=2, padding='SAME', name='pool1')
        print(inputs.get_shape)

        # block_layer4 128*4*38*256 -> 128*4*38*512
        inputs = block_layer(inputs=inputs, filters=512, block_fn=building_block, blocks=1,
                             strides=1, is_training=is_training, name='block_layer4')
        print(inputs.get_shape)

        # pool3_1x2  128*4*38*512 -> 128*2*19*512
        inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=2, padding='SAME', name='pool3_1x2')
        print(inputs.get_shape)

        # pool3_1x2  128*2*19*512 -> 128*1*18*512
        inputs = tf.layers.conv2d(inputs=inputs, filters=512, kernel_size=[2, 2], strides=1, padding='VALID',
                                  use_bias=False, kernel_initializer=tf.variance_scaling_initializer(), name='conv2')
        print(inputs.get_shape)

        inputs = batch_norm_relu(inputs, is_training)

        # reshape 特征图
        with tf.variable_scope('Reshaping_cnn'):
            shape = inputs.get_shape().as_list()  # [batch, height, width, features]
            inputs = tf.transpose(inputs, perm=[0, 2, 1, 3],
                                  name='transposed')  # [batch, width, height, features] 128*1*75*512 -> 128*75*1*512
            inputs = tf.reshape(inputs, [shape[0], -1, shape[1] * shape[3]],
                                name='reshaped')  # [batch, width, height x features] 128*75*1*512 -> 128*75*512
        return inputs