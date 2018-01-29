#!/usr/bin/env python
__author__ = 'solivr'

import argparse
import os
import csv
import sys
import numpy as np

try:
    import better_exceptions
except ImportError:
    pass
import tensorflow as tf
from src.model_info import crnn_fn
from src.data_handler import data_loader, preprocess_image_for_prediction
from src.input_utils import input_fn
from src.config import Params, Alphabet, import_params_from_json


def main(unused_argv):
    # 输出路径不存在就创建
    if not os.path.exists(FLAGS.output_model_dir):
        os.makedirs(FLAGS.output_model_dir)

    if FLAGS.params_file:
        dict_params = import_params_from_json(json_filename=FLAGS.params_file)
        parameters = Params(**dict_params)
    else:
        parameters = Params(train_batch_size=128,
                            eval_batch_size=59,
                            learning_rate=0.001,  # 1e-3 recommended
                            learning_decay_rate=0.5,
                            learning_decay_steps=23438 * 2,
                            evaluate_every_epoch=1,
                            save_interval=5e3,
                            input_shape=(32, 304),
                            image_channels=3,
                            optimizer='adam',
                            digits_only=False,
                            alphabet=Alphabet.CHINESECHAR_LETTERS_DIGITS_EXTENDED,
                            alphabet_decoding='same',
                            csv_delimiter=' ',
                            csv_files_train=FLAGS.csv_files_train,
                            csv_files_eval=FLAGS.csv_files_eval,
                            output_model_dir=FLAGS.output_model_dir,
                            n_epochs=FLAGS.nb_epochs,
                            gpu=FLAGS.gpu
                            )

    model_params = {
        'Params': parameters,
    }
    # 保存配置
    parameters.export_experiment_params()

    os.environ['CUDA_VISIBLE_DEVICES'] = parameters.gpu
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.4

    # Count number of image filenames in csv
    n_samples = 0
    for file in parameters.csv_files_train:
        with open(file, mode='r', encoding='utf8') as csvfile:
            n_samples += len(csvfile.readlines())

    save_checkpoints_steps = int(np.ceil(n_samples / parameters.train_batch_size))
    keep_checkpoint_max = parameters.n_epochs
    print(n_samples, 'save_checkpoints_steps', save_checkpoints_steps, ' keep_checkpoint_max', keep_checkpoint_max)
    # Config estimator

    est_config = tf.estimator.RunConfig()
    est_config = est_config.replace(keep_checkpoint_max=keep_checkpoint_max,
                                    save_checkpoints_steps=save_checkpoints_steps,
                                    session_config=config_sess,
                                    save_summary_steps=100,
                                    model_dir=parameters.output_model_dir)

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,
                                       params=model_params,
                                       model_dir=parameters.output_model_dir,
                                       config=est_config
                                       )
    try:
        tensors_to_log = {'train_accuracy': 'train_accuracy'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
        for e in range(0, parameters.n_epochs, parameters.evaluate_every_epoch):
            estimator.train(input_fn=data_loader(csv_filename=parameters.csv_files_train,
                                                 params=parameters,
                                                 batch_size=parameters.train_batch_size,
                                                 num_epochs=parameters.evaluate_every_epoch,
                                                 data_augmentation=True,
                                                 image_summaries=True),
                            hooks=[logging_hook])
            eval_results = estimator.evaluate(input_fn=data_loader(csv_filename=parameters.csv_files_eval,
                                                                   params=parameters,
                                                                   batch_size=parameters.eval_batch_size,
                                                                   num_epochs=1),
                                              steps=np.floor(n_samples / parameters.eval_batch_size),
                                              )
            print('Evaluation results: %s' % (str(eval_results)))
        # for tensorflow1.4
        # estimator.train(input_fn=input_fn(filename=parameters.csv_files_train,
        #                                   is_training=True
        #                                   params=parameters,
        #                                   batch_size=parameters.train_batch_size,
        #                                   num_epochs=parameters.n_epochs),
        #                 hooks=[logging_hook])
    except KeyboardInterrupt:
        print('Interrupted')
        estimator.export_savedmodel(os.path.join(parameters.output_model_dir, 'export'),
                                    preprocess_image_for_prediction(min_width=10))
        print('Exported model to {}'.format(os.path.join(parameters.output_model_dir, 'export')))

    estimator.export_savedmodel(os.path.join(parameters.output_model_dir, 'export'),
                                preprocess_image_for_prediction(min_width=10))
    print('Exported model to {}'.format(os.path.join(parameters.output_model_dir, 'export')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ft', '--csv_files_train', required=True, type=str, help='CSV filename for training',
                        nargs='*', default=['E:/val1.csv'])
    parser.add_argument('-fe', '--csv_files_eval', type=str, help='CSV filename for evaluation',
                        nargs='*', default=None)
    parser.add_argument('-o', '--output_model_dir', required=True, type=str,
                        help='Directory for output', default='./estimator')
    parser.add_argument('-n', '--nb_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('-g', '--gpu', type=str, help="GPU 0,1 or '' ", default='0')
    parser.add_argument('-p', '--params-file', type=str, help='Parameters filename', default=None)

    tf.logging.set_verbosity(tf.logging.DEBUG)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
