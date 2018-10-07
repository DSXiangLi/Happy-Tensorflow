# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 22:13:53 2018

@author: xiang
"""

import tensorflow as tf
from base.base_model import BaseModel

class BaseModel_IMG(BaseModel):
    '''
    Define Classic component used for Image NN
    1. conv + batchnormalization + relu
    2. conv + relu
    3. dense + relu + dropout
    4. dense + relu + batchnormalization + dropout
    Note: all normalization/augumentation operation is for train only
    '''

    def __init__(self, config):
        super(BaseModel_IMG, self).__init__(config)

    def conv_bn_relu(name, x_t, filters, kernel_size, padding, is_train):
        with tf.variable_scope(name):
            out = tf.layers.conv2d(x_t, filters, kernel_size, padding, name = 'conv')
            out = tf.layers.batch_normalization(out, training = is_train, name = 'bn')
            out = tf.nn.relu(out)
        return out

    def conv_relu(name, x_t, filters, kernel_size, padding, is_train):
        with tf.variable_scope(name):
            out = tf.layers.conv2d(x_t, filters, kernel_size, padding, name = 'conv')
            out = tf.nn.relu(out)
        return out

    def dense_relu_dropout(name, x_t, units, dropout_rate, is_train):
        with tf.variable_scope(name):
            out = tf.layer.dense(x_t, units, name = 'dense')
            out = tf.nn.relu(out)
            out = tf.layers.dropout(out, dropout_rate, training= is_train, name ='dropout')
        return out

    def dense_relu_bn_dropout(name, x_t, units, dropout_rate, is_train):
        with tf.variable_scope(name):
            out = tf.layer.dense(x_t, units, name = 'dense')
            out = tf.nn.relu(out)
            out = tf.layers.batch_normalization(out, training = is_train, name = 'bn')
            out = tf.layers.dropout(out, dropout_rate, training = is_train, name ='dropout')
        return out

