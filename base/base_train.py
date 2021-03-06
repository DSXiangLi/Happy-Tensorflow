# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:11:55 2018

@author: xiang
"""

import tensorflow as tf

class BaseTrain:
    '''
    Combine config, data, logger and model. And iterate across train
    '''
    def __init__(self, sess, config, data_loader = None, logger, model):
        self.config = config
        self.sess = sess
        self.logger = logger
        self.model = model

        if data_loader is not None:
            self.data_loader = data_loader

        self.train_init = tf.global_variable_initializer()
        self.sess.run(self.train_init)

    def train(self):
        pass 
    def train_epoch(self):
        '''
        Iterate across all training record.
        '''
        pass 

    def train_step(self):
        '''
        trigger computation of the graph
        train against batch sample, depending on batch size
        '''
        pass
