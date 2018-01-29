# -*- coding: utf-8 -*-
# @Time    : 2017/10/26 14:09
# @Author  : zhoujun

import tensorflow as tf
from scipy.misc import imread
import os
import natsort

class PredictionModel:

    def __init__(self, model_dir,img_channel=3, gpu_id = None):
        if gpu_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            config_sess = tf.ConfigProto()
            config_sess.gpu_options.per_process_gpu_memory_fraction = 0.2
        else:
            config_sess = None

        self.session = tf.Session(config=config_sess)
        self.img_channel=img_channel
        # if session != None:
        #     self.session = session
        # else:
        #     self.session = tf.get_default_session()
        self.model = tf.saved_model.loader.load(self.session, ['serve'], model_dir)
        # 得到predict  op
        self._input_dict, self._output_dict = self.__signature_def_to_tensors(self.model.signature_def['predictions'])

    def predict_one(self, image):
        if self.img_channel == 1:
            image = imread(image, mode='L')
            image = image.reshape([image.shape[0], image.shape[1], 1])
        else:
            image = imread(image, mode='RGB')
        output = self._output_dict
        # 运行predict  op
        predictions = self.session.run(output, feed_dict={self._input_dict['images']: image})
        return predictions['words'],predictions['score']

    def predict_all(self, image_path):
        if isinstance(image_path, list):
            image_list = image_path
        elif os.path.isfile(image_path):
            image_list = [image_path]
        else:
            image_list = [os.path.join(image_path, file) for file in os.listdir(image_path) if file.endswith('.jpg')]
            image_list = natsort.natsorted(image_list)

        result_list = []

        for img in image_list:
            predictions,_ = self.predict_one(img)
            print(predictions[0].decode())
            result_list.append(predictions[0].decode())
        return result_list

    def __signature_def_to_tensors(self, signature_def):  # from SeguinBe
        g = tf.get_default_graph()
        return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
               {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}

if __name__ == '__main__':
    model_dir = r'Z:\zhoujun\tf-crnn\result\mutil_font\resnet_300k_e\export\1515589663'
    image_path = 'data/images/single1'
    tf_model = PredictionModel(model_dir,img_channel=3)
    result_list = tf_model.predict_all(image_path)
