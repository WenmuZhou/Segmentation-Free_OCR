# -*- coding: utf-8 -*-
# @Time    : 2017/11/27 19:12
# @Author  : zhoujun
import tensorflow as tf
from src.data_handler import padding_inputs_width, augment_data
from src.config import CONST, Params, Alphabet


def input_fn(filename, is_training, params, batch_size=1, num_epochs=1):
    """A simple input_fn using the tf.data input pipeline."""

    def example_parser(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.string),
                'image_raw': tf.FixedLenFeature([], tf.string)
            }
        )
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        depth = tf.cast(features['depth'], tf.int32)
        image = tf.reshape(image, [height, width, depth])
        label = tf.cast(features['label'], tf.string)

        # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
        # image = tf.cast(image, tf.float32) / 255 - 0.5
        # Data augmentation
        if depth != params.image_channels:
            if params.image_channels == 1:
                image = tf.image.rgb_to_grayscale(image)
            elif params.image_channels == 3:
                image = tf.image.grayscale_to_rgb(image)

        if is_training:
            image = augment_data(image)

        image, width = padding_inputs_width(image, params.input_shape, increment=CONST.DIMENSION_REDUCTION_W_POOLING)

        return {'images': image, 'images_widths': width, 'labels': label}, label

    dataset = tf.data.TFRecordDataset([filename])

    # Apply dataset transformations
    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance. Because MNIST is
        # a small dataset, we can easily shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.repeat(num_epochs)

    # Map example_parser over dataset, and batch results by up to batch_size
    dataset = dataset.map(example_parser).prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels


if __name__ == '__main__':
    parameters = Params(eval_batch_size=128,
                        input_shape=(32, 304),
                        digits_only=False,
                        alphabet=Alphabet.CHINESECHAR_LETTERS_DIGITS_EXTENDED,
                        alphabet_decoding='same',
                        image_channels=3
                        )

    next_batch = input_fn(filename='/media/zhoujun/文件/val1.tfrecords', is_training=False, params=parameters,
                          batch_size=2)

    # Now let's try it out, retrieving and printing one batch of data.
    # Although this code looks strange, you don't need to understand
    # the details.
    with tf.Session() as sess:
        first_batch = sess.run(next_batch)
    print(first_batch['images'])
