# -*- coding: utf-8 -*-
# @Time    : 2017/11/21 19:52
# @Author  : zhoujun

import os

try:
    import better_exceptions
except ImportError:
    pass
import tensorflow as tf
from src.model_info import crnn_fn
from src.config import Params, Alphabet
from src.data_handler import preprocess_image_for_prediction


def convert_cpkt(checkpoint_path, model_output):
    '''
    cpkt文件转换为pb文件
    :param checkpoint_path: cpkt路径
    :param model_output: pb文件保存位置
    :return: None
    '''
    # 输入路径不存在就报错
    if not os.path.exists(checkpoint_path):
        assert FileNotFoundError

    # 输出路径不存在就创建
    if not os.path.exists(model_output):
        print(model_output,'not exist')
        os.mkdir(model_output)

    parameters = Params(digits_only=False,
                        alphabet=Alphabet.CHINESECHAR_LETTERS_DIGITS_EXTENDED,
                        output_model_dir=model_output,
                        image_channels=3,
                        )

    model_params = {'Params': parameters, }

    # Config estimator
    est_config = tf.estimator.RunConfig()

    est_config = est_config.replace(model_dir=parameters.output_model_dir)

    estimator = tf.estimator.Estimator(model_fn=crnn_fn,
                                       params=model_params,
                                       config=est_config,
                                       model_dir=parameters.output_model_dir,
                                       )
    try:
        estimator.export_savedmodel(os.path.join(model_output, 'export'),
                                    preprocess_image_for_prediction(min_width=10,image_channels = parameters.image_channels),
                                    checkpoint_path=checkpoint_path)
        print('Exported model to {}'.format(os.path.join(model_output, 'export')))

    except KeyboardInterrupt:
        print('Interrupted')


if __name__ == '__main__':
    model_path = r'Z:\zhoujun\tf-crnn\result\mutil_font\resnet_300k_e\model.ckpt-281245'
    model_output = r'Z:\zhoujun\tf-crnn\result\mutil_font\resnet_300k_e'
    convert_cpkt(model_path, model_output)
