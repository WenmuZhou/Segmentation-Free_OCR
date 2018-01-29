# -*- coding: utf-8 -*-
# @Time    : 2017/11/29 8:43
# @Author  : zhoujun
from src.data_handler import data_loader, input_fn
from src.config import Params, Alphabet
import tensorflow as tf
import time

if __name__ == '__main__':
    parameters = Params(eval_batch_size=128,
                        input_shape=(32, 304),
                        digits_only=False,
                        alphabet=Alphabet.CHINESECHAR_LETTERS_DIGITS_EXTENDED,
                        alphabet_decoding='same',
                        image_channels=1,
                        csv_delimiter=' ',
                        )
    # next_batch = input_fn(filename='/media/zhoujun/文件/val1.tfrecords', is_training=False, params=parameters,
    #                       batch_size=2)

    featureBatch, labelBatch = input_fn(csv_filename='E:/val1.csv', params=parameters,
                                        batch_size=parameters.eval_batch_size,
                                        num_epochs=1)

    global_init = tf.global_variables_initializer()
    loacl_init = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(global_init)
        sess.run(loacl_init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        start = time.time()
        example, label = sess.run([featureBatch, labelBatch])
        print('time: ',time.time()-start)
        print(len(label))
        coord.request_stop()
        coord.join(threads)
