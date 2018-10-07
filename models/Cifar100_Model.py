# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 11:44:12 2018

@author: xiang
"""

import tensorflow as tf
from base.model_collection import BaseModel_IMG

class LeNet5(BaseModel_IMG):
    def __init__(self, config, data_loader):
        super(LeNet5, self).__init__(config)
        self.data_loader = data_loader
        ## Input
        self.is_train = None
        self.x = None
        self.y = None
        ## Output
        self.out = None
        self.out_argmax = None
        ## train
        self.loss = None
        self.accu = None
        self.optimizer = None
        self.train_step = None
        ## Initialize function
        self.build_model()
        self.init_save()

    def build_model(self):

        with tf.variable_scope('inputs'):
            self.x, self.y = self.data_loader.get_value()
            self.is_train = tf.placeholder(tf.bool, name = 'training_flag')

        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_train)

        with tf.variable_scope('network'):
            ## layer1 input 32*32*3
            with tf.variable_scope('convolution'):
                conv1 = tf.layers.conv2d(self.x, filters = 6, kernel_size = (5,5), strides =(1,1),
                                         padding = 'valid', activation = tf.nn.tanh, name = 'conv1') # 28*28*6
                pool1 = tf.layers.average_pooling2d(conv1, pool_size = (2,2),
                                                strides = (2,2), name = 'pool1') # 14*14*6

                conv2 = tf.layers.conv2d(pool1, filters = 16, kernel_size = (5,5), strides = (1,1),
                                         padding = 'valid', activation = tf.nn.tanh, name = 'conv2' ) # 10*10*16
                pool2 = tf.layers.average_pooling2d(conv2, pool_size = (2,2),
                                                strides = (2,2), name = 'pool2') # 5*5*16

            with tf.variable_scope('dense'):
                flatten = tf.layers.flatten(pool2, name ='flatten')

                dense1 = tf.layers.dense(flatten, units = 120, activation = tf.nn.tanh,
                                         kernel_initializer = tf.initializers.truncated_normal, name = 'dense1')

                dense2 = tf.layers.dense(dense1, units = 10, activation = tf.nn.tanh,
                                         kernel_initializer = tf.initializers.truncated_normal, name = 'dense2')

            with tf.variable_scope('out'):
                self.out = tf.layers.dense(dense2, units = self.config.num_class,
                                           kernel_initializer = tf.initializers.truncated_normal, name = 'dense')
                tf.add_to_collection('output',self.out)

            with tf.variable_scope('out_argmax'):
                self.out_argmax = tf.argmax(self.out, axis = -1, output_type = self.config.target_type, name = 'out_argmax')

            with tf.variable_scope('loss_accu'):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y,
                                                                           logits = self.out,
                                                                           name = 'loss')
                self.accu = tf.reduce_mean(tf.cast(tf.equal(self.out_argmax, self.y), tf.float32)) ## accu in eah batch?

            with tf.variable_scope('train'):
                self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_step = self.optimizer.minimize(self.loss,
                                                              global_step = self.global_step)
            tf.add_to_collection('train', self.loss)
            tf.add_to_collection('train', self.accu)
            tf.add_to_collection('train', self.train_step)

    def init_save(self):
        self.saver = tf.train.Saver(max_to_keep = self.config.max_to_keep,
                                    save_relative_paths = True)

