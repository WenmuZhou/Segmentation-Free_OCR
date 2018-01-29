#!/usr/bin/env python
__author__ = 'zj'

import argparse
import os
import sys
import numpy as np
import time
# try:
#     import better_exceptions
# except ImportError:
#     pass
import tensorflow as tf
from src.model_info import crnn_fn
from src.data_handler import data_loader, input_fn
from src.config import Params, Alphabet


# from src.input_utils import input_fn

def main(unused_argv):
    models_path = FLAGS.input_model_dir
    if not os.path.exists(models_path):
        assert FileNotFoundError

    models_list = [os.path.join(models_path, x[:-5]) for x in os.listdir(models_path) if x.endswith('.meta')]

    # 输出路径不存在就创建
    if not os.path.exists(FLAGS.output_model_dir):
        os.makedirs(FLAGS.output_model_dir)

    parameters = Params(eval_batch_size=128,
                        input_shape=(45, 304),
                        digits_only=False,
                        alphabet=Alphabet.CHINESECHAR_LETTERS_DIGITS_EXTENDED,
                        alphabet_decoding='same',
                        image_channels=3,
                        csv_delimiter=' ',
                        csv_files_eval=FLAGS.csv_files_eval,
                        output_model_dir=FLAGS.output_model_dir,
                        gpu=FLAGS.gpu
                        )

    model_params = {
        'Params': parameters,
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = parameters.gpu
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.27

    # Config estimator
    est_config = tf.estimator.RunConfig()
    est_config = est_config.replace(session_config=config_sess,
                                    save_summary_steps=100,
                                    model_dir=parameters.output_model_dir)

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,
                                       params=model_params,
                                       config=est_config,
                                       model_dir=parameters.output_model_dir,
                                       )
    # Count number of image filenames in csv
    n_samples = 0
    for file in parameters.csv_files_eval:
        with open(file, mode='r', encoding='utf8') as csvfile:
            n_samples += len(csvfile.readlines())
    print(n_samples, np.floor(n_samples / parameters.eval_batch_size), parameters.eval_batch_size)
    try:
        with open(FLAGS.output_file, encoding='utf-8', mode='w') as save_file:
            for model in models_list:
                start = time.time()

                eval_results = estimator.evaluate(input_fn=lambda: input_fn(csv_filename=parameters.csv_files_eval,
                                                                            params=parameters,
                                                                            batch_size=parameters.eval_batch_size,
                                                                            num_epochs=1),
                                                  steps=np.floor(n_samples / parameters.eval_batch_size),
                                                  checkpoint_path=model)

                # eval_results = estimator.evaluate(input_fn=lambda: input_fn(filename=parameters.csv_files_eval,
                #                                                             is_training=False,
                #                                                             params=parameters,
                #                                                             batch_size=parameters.eval_batch_size,
                #                                                             num_epochs=1),
                #                                   steps=3,
                #                                   checkpoint_path=model)
                print(time.time() - start)
                print('model: %s Evaluation results: %s' % (model, str(eval_results)))
                save_file.write(model + ' ' + str(eval_results) + '\n')

    except KeyboardInterrupt:
        print('Interrupted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fe', '--csv_files_eval', required=False, type=str, help='CSV filename for evaluation',
                        nargs='*', default=['E:/val1.csv'])
    parser.add_argument('-o', '--output_model_dir', required=False, type=str,
                        help='Directory for output', default='models_vgg_100K_no_eval')
    parser.add_argument('-m', '--input_model_dir', required=False, type=str,
                        help='Directory for output', default='model_test')
    parser.add_argument('-g', '--gpu', type=str, help="GPU 0,1 or '' ", default='0')
    parser.add_argument('-of', '--output_file', required=False, type=str, default='123.txt', help="the log output file")

    print(tf.__version__)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
