# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:46:37 2018

@author: xiang
"""

import sys
sys.path.insert(0, './data_loader/')
import tensorflow as tf

from Data_Load import DataLoaderNumpy_Big, DataLoaderNumpy_Small,DataLoaderTFRecord

class Config:
    project = "cifar-100-python"
    feature_type = tf.float32
    target_type = tf.int64
    image_height = 32
    image_width = 32
    channel = 3
    batch_size = 8
    shuffle_size = 1000
    reshuffle = True
    num_class = 5


def test_Numpybig(config):
    tf.reset_default_graph()

    sess = tf.Session()

    data_loader = DataLoaderNumpy_Big(config)

    data_loader.initialize(sess, is_train = True)

    x,y = data_loader.get_value() ##x,y are pointer to the next iterator

    batch_x, batch_y = sess.run([x,y])

    print(batch_x.shape, batch_x.dtype)
    print(batch_y.shape, batch_y.dtype)

    data_loader.initialize(sess, is_train = False)

    batch_x, batch_y = sess.run([x,y])

    print(batch_x.shape, batch_x.dtype)
    print(batch_y.shape , batch_y.dtype)

test_Numpybig(config)


def test_NumpySmall(config):
    tf.reset_default_graph()

    sess = tf.Session()

    data_loader = DataLoaderNumpy_Small(config)

    data_loader.initialize(sess, is_train = True)

    x,y = data_loader.get_value() ##x,y are pointer to the next iterator

    batch_x, batch_y = sess.run([x,y])

    print(batch_x.shape, batch_x.dtype)
    print(batch_y.shape, batch_y.dtype)

    data_loader.initialize(sess, is_train = False)

    batch_x, batch_y = sess.run([x,y])

    print(batch_x.shape, batch_x.dtype)
    print(batch_y.shape , batch_y.dtype)

test_NumpySmall(config)



def test_TFRecord(config):
    tf.reset_default_graph()

    sess = tf.Session()

    data_loader = DataLoaderTFRecord(config)

    data_loader.initialize(sess, is_train = True)

    x,y = data_loader.get_value() ##x,y are pointer to the next iterator

    batch_x, batch_y = sess.run([x,y])

    print(batch_x.shape, batch_x.dtype)
    print(batch_y.shape, batch_y.dtype)

    data_loader.initialize(sess, is_train = False)

    batch_x, batch_y = sess.run([x,y])

    print(batch_x.shape, batch_x.dtype)
    print(batch_y.shape , batch_y.dtype)

test_TFRecord(config)