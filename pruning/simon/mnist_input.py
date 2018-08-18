###############################################################################
##############################################################################
# Author:               Imanol Schlag (more info on ischlag.github.io)
# Description:  MNIST input pipeline
# Date:                 11.2016
#
# Note: Only uses one queue,

import tensorflow as tf
import sys
sys.settrace
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

class MnistData:
    """
    Downloads the MNIST dataset and creates an input pipeline ready to be fed
    into a model.

    - converts [0 1] to [-1 1]
    - shuffles the input
    - builds batches
    """
    NUM_THREADS = 8
    NUMBER_OF_CLASSES = 10
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    NUM_OF_CHANNELS = 1
    MNIST_PATH = '../../data/mnist'

    def __init__(self, batch_size, mnist_path = MNIST_PATH):
        """ Downloads the mnist data if necessary. """
        self.batch_size = batch_size
        MNIST_PATH = mnist_path
        self.mnist = input_data.read_data_sets(mnist_path)

        self.TRAIN_SET_SIZE = self.mnist.train.images.shape[0]
        self.TEST_SET_SIZE = self.mnist.test.images.shape[0]
        self.VALIDATION_SET_SIZE = self.mnist.validation.images.shape[0]

    def build_train_data_tensor(self, shuffle=False, augmentation=False):
        return self.__build_generic_data_tensor(
            self.mnist.train.images, self.mnist.train.labels, shuffle,
            augmentation)

    def build_test_data_tensor(self, shuffle=False, augmentation=False):
        return self.__build_generic_data_tensor(
            self.mnist.test.images, self.mnist.test.labels, shuffle,
            augmentation)

    def build_validation_data_tensor(self, shuffle=False, augmentation=False):
        return self.__build_generic_data_tensor(
            self.mnist.validation.images, self.mnist.validation.labels,
            shuffle, augmentation)

    def __build_generic_data_tensor(self, raw_images, raw_targets, shuffle,
            augmentation):
        """ Creates the input pipeline and performs some preprocessing. """

        images = ops.convert_to_tensor(raw_images)
        targets = ops.convert_to_tensor(raw_targets)

        set_size = raw_images.shape[0]
# turn off reshaping because I do that later in mnist_inference.py
#        images = tf.reshape(images, [set_size, 28, 28, 1])
        image, label = tf.train.slice_input_producer([images, targets],
                                                     shuffle=shuffle)

        # Data Augmentation
        #if augmentation:
        # image = tf.image.resize_image_with_crop_or_pad(image, self.IMAGE_HEIGHT+4, self.IMAGE_WIDTH+4)
        # image = tf.random_crop(image, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.NUM_OF_CHANNELS])
        # image = tf.image.random_flip_left_right(image)
    # also getting rid of this for my stuff for now
    #       image = tf.image.per_image_standardization(image)

        images_batch, labels_batch = tf.train.batch(
            [image, label], batch_size=self.batch_size,
            num_threads=self.NUM_THREADS)

        return images_batch, labels_batch


if __name__ == '__main__':
    batch_size = 100
    data = MnistData(batch_size)
    image_batch_tensor, target_batch_tensor = data.build_train_data_tensor()

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for i in range(10):
        image_batch, target_batch = sess.run(
            [image_batch_tensor, target_batch_tensor])
        print(image_batch.shape)
        print(target_batch.shape)

    coord.request_stop()
    coord.join(threads)
