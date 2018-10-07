# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 21:38:39 2018

@author: xiang
"""

import tensorflow as tf
import numpy as np
import pprint as pp
from data_loader.Data_Load import DataLoaderNumpy_Big
from models.Cifar100_Model import LeNet5


def main():

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
        ## Model Parameter
        num_class = 5
        learning_rate = 0.1
        max_to_keep = 5

    tf.reset_default_graph()

    sess = tf.Session()

    data_loader = DataLoaderNumpy_Big(Config)

    data_loader.initialize(sess, is_train = True)

    model = LeNet5(Config, data_loader)

    ## check model input: batch_size * input_shape
    x_batch, y_batch = sess.run([model.x, model.y])
    print(x_batch.shape, x_batch.dtype )
    print(y_batch.shape, y_batch.dtype )

    ## check counter
    epoch_step1, global_step1 = sess.run([model.epoch_step, model.global_step])
    epoch_step1, global_step1 = sess.run([model.epoch_step, model.global_step])

    ## check Tensor collection
    graph = tf.get_default_graph()
    collection_keys = graph.get_all_collection_keys()
    for key in collection_keys:
        pp.pprint("\n {} All collections in key {} {} \n". format("=" * 20, key, "=" * 20))
        if len(tf.get_collection(key)) !=0 :
            pp.pprint(tf.get_collection(key))

    tf.Session.close()
