# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:24:34 2018

@author: xiang
"""

import tensorflow as tf

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.epoch_step = None
        self.global_step = None
        self.epoch_step_increment = None
        self.global_step_increment = None

        self.saver = None

        self.init_epoch_step()
        self.init_global_step()

    def init_epoch_step(self):
        with tf.variable_scope("cur_epoch"):
            self.epoch_step = tf.Variable(0, trainable = False, name = 'epoch_step')
            self.epoch_step_increment = tf.assign(self.epoch_step, self.epoch_step + 1)

    def init_global_step(self):
        with tf.variable_scope("global"):
            self.global_step = tf.Variable(0, trainable = False, name = 'global_step')
            self.global_step_increment = tf.assign(self.global_step, self.global_step + 1)

    def save(self, sess):
        print('Saving Model')
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step)
        print('Model Saved')

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print('Loading model checkpoint {} ...\n'.format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print('Model loaded')

    def init_save(self, sess):
        ### Initialize saver object for later model saving 
        pass 

    def build_model(self):
        raise NotImplementedError


