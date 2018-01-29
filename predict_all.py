# -*- coding: utf-8 -*-
# @Time    : 2017/10/26 14:09
# @Author  : zhoujun

import tensorflow as tf
# from src.loader import PredictionModel
from scipy.misc import imread
import time
import os
import cv2
from tqdm import tqdm
import numpy as np


class PredictionModel:

    def __init__(self, model_dir, gpu_id = None):
        if gpu_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            config_sess = tf.ConfigProto()
            config_sess.gpu_options.per_process_gpu_memory_fraction = 0.2
        else:
            config_sess = None

        self.session = tf.Session(config=config_sess)
        # if session != None:
        #     self.session = session
        # else:
        #     self.session = tf.get_default_session()
        self.model = tf.saved_model.loader.load(self.session, ['serve'], model_dir)
        # 得到predict  op
        self._input_dict, self._output_dict = self.__signature_def_to_tensors(self.model.signature_def['predictions'])

    def predict_one(self, image):
        output = self._output_dict
        # 运行predict  op
        result = self.session.run(output, feed_dict={self._input_dict['images']: image})
        return result

    def predict_all(self, image_path, img_channel=3):
        if isinstance(image_path, list):
            image_list = image_path
        elif os.path.isfile(image_path):
            image_list = [image_path]
        else:
            image_list = [file for file in os.listdir(image_path) if file.endswith('.jpg')]
            image_list.sort(key=lambda x: int(x[:-4]))
            image_list = [os.path.join(image_path, f) for f in image_list]

        result_list = []

        for img in image_list:
            if img_channel == 1:
                image = imread(img, mode='L')
                image = image.reshape([image.shape[0], image.shape[1], 1])
            else:
                image = imread(img, mode='RGB')
            predictions = self.predict_one(image)
            transcription = predictions['words']
            print(transcription[0].decode())
            result_list.append(transcription[0].decode())
        return result_list

    def __signature_def_to_tensors(self, signature_def):  # from SeguinBe
        g = tf.get_default_graph()
        return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
               {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}


def write_file(filename, result):
    s = '\n'.join(result)
    with open(filename, mode='w', encoding='utf8') as w:
        w.writelines(s)


def cal_acc(data_path, result_list):
    gt = []
    with open(data_path + '/zhengque.txt', encoding='utf8', mode='r') as g_f:
        g = g_f.readlines()
        for g_i in g:
            gt.append(g_i.strip())
    count = 0
    for i in range(len(result_list)):
        if gt[i] == result_list[i]:
            count += 1
    print(count / len(result_list))
    return gt


def cal_acc1(gt, result_list):
    count_a = 0
    for i in range(len(result_list)):
        count = 0
        all_num = 0
        all_num += len(result_list[i])

        a = gt[i]
        b = result_list[i]

        a_i = 0
        b_i = 0
        while (a_i < len(a) and b_i < len(b)):
            if a[a_i] == b[b_i]:
                a_i += 1
                b_i += 1
                count += 1
            elif a_i + 1 < len(a) and a[a_i + 1] == b[b_i]:
                a_i += 1
            else:
                b_i += 1
        print(count / all_num)
        count_a += count / all_num
    print('avg', count_a / len(result_list))


if __name__ == '__main__':
    root_path = r'E:\tf-crnn'
    model_path = root_path + r'\result\mutil_font\models'
    image_path = root_path + r'\data\images'
    result_path = root_path + r'\data\images\zhengque'

    model_list = os.listdir(model_path)
    image_list = [x for x in os.listdir(image_path) if x.__contains__('single')]
    pbar = tqdm(total=len(model_list))


    for model_i in model_list:
        model = os.path.join(model_path, model_i)
        if not model.__contains__('resnet'):
            continue
        print(model)
        tf_model = PredictionModel(model)
        if model.__contains__('resnet'):
            img_chanel = 1
        else:
            img_chanel = 3
        for image_i in image_list:
            img = os.path.join(image_path, image_i)
            img_list = [os.path.join(image_path, image_i, x) for x in os.listdir(img)]
            if not os.path.exists(os.path.join(result_path, image_i)):
                os.makedirs(os.path.join(result_path, image_i))
            save_path = os.path.join(result_path, image_i, model_i + '_predict.txt')
            # if os.path.exists(save_path):
            #     print(save_path, ' exist')
            #     continue

            print(save_path)
            result_list = tf_model.predict_all(img_list, img_channel=img_chanel)
            write_file(save_path, result_list)
        pbar.update(1)
    pbar.close()
    # result_list, gt = predict(model_dir=model_dir, image_path=data_path, gpu_id=None)
