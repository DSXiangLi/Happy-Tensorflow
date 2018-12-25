# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:56:47 2018

@author: xiang
"""

import os
import tensorflow as tf
import pickle
import numpy as np

class DataLoaderTFRecord:
    def __init__(self, config):
        self.config = config
        self.train= tf.data.TFRecordDataset(os.path.join('data', config.project, 'train.tfrecord'))
        self.test = tf.data.TFRecordDataset(os.path.join('data', config.project, 'test.tfrecord'))

        self.dataset = None
        self.iterator = None
        self.init_op = None
        self.next_batch = None


    def dataset_api(self, is_train):
        print("Build Dataset API for training :", is_train)
        if is_train:
            self.dataset = self.train.map(self.parse_image)
            self.dataset = self.dataset.shuffle(self.config.shuffle_size)
        else:
            self.dataset = self.test.map(self.parse_image)

        self.dataset = self.dataset.batch(self.config.batch_size)
        print(self.dataset.output_shapes, self.dataset.output_types)

        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types,
                                                        self.dataset.output_shapes)
        self.init_op = self.iterator.make_initializer(self.dataset)

        self.next_batch = self.iterator.get_next()
        print("x_batch: ", self.next_batch[0].shape)
        print("y_batch: ", self.next_batch[1].shape)

    def parse_image(self, record):
        features = {'label': tf.FixedLenFeature((), tf.int64),
                   'img': tf.FixedLenFeature((), tf.string)}
        parsed_record= tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed_record['img'], tf.uint8) ## unint8 is binary encoding
        image = tf.reshape(image, [self.config.image_width, self.config.image_height, self.config.channel])
        return tf.cast(parsed_record['label'], self.config.target_type), tf.cast(image, self.config.feature_type)

    def initialize(self, sess, is_train):
        if is_train:
            print("Initialize Iterator for training")
        else:
            print("Initialize Iterator for testing")
        self.dataset_api(is_train)
        sess.run(self.init_op)

    def get_value(self):
        return self.next_batch

class DataLoaderNumpy_Small:
    '''
    1. Create tensor directly from fileanme and label - tf.constant
    2. apply parser to read in file name
    '''
    def __init__(self, config):
        self.config = config

        self.train_files = []
        self.test_files = []
        ## loader should load both test and train. And treate them differently in iteration
        with open(os.path.join('data',self.config.project,'x_train_filenames.pkl'), 'rb') as f:
            self.train_files = pickle.load(f)

        with open(os.path.join('data',self.config.project,'x_test_filenames.pkl'), 'rb') as f:
            self.test_files = pickle.load(f)

        self.y_train = np.load(os.path.join('data',self.config.project,'y_train.npy'))
        self.y_test = np.load(os.path.join('data',self.config.project,'y_test.npy'))

        self.dataset = None
        self.iterator = None
        self.init_op = None
        self.next_batch = None

    def dataset_api(self, is_train):
        print("Dataset API for training :", is_train)
        ## create Dataset directly from input data - tf.constant
        if is_train:
            self.dataset = tf.data.Dataset.from_tensor_slices((self.train_files, self.y_train))
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices((self.test_files, self.y_test))

        ## can apply other fucntion given different input
        self.dataset = self.dataset.map(self.parse_image)
        self.dataset = self.dataset.batch(self.config.batch_size)

        ## shuffle data if train
        if is_train:
            self.dataset = self.dataset.shuffle(self.config.shuffle_size)

        ## can be more general if config has hull dimension of input file
        self.iterator = tf.data.Iterator.from_structure((self.config.feature_type, self.config.target_type),
                                                        ([None,
                                                          self.config.image_height,
                                                          self.config.image_width,
                                                          self.config.channel],[None,]))
        self.init_op = self.iterator.make_initializer(self.dataset)

        self.next_batch = self.iterator.get_next()
        print("x_batch: ", self.next_batch[0].shape)
        print("y_batch: ", self.next_batch[1].shape)

    def parse_image(self, file, target):
        img = tf.read_file('data//'+ self.config.project + '//'+ file)
        img = tf.image.decode_png(img, channels = self.config.channel)
        img = tf.image.resize_images(img, [self.config.image_width, self.config.image_height] )

        return tf.cast(img, self.config.feature_type), tf.cast(target, self.config.target_type)

    def initialize(self, sess, is_train):
        if is_train:
            print("Initialize Iterator for training")
        else:
            print("Initialize Iterator for testing")
        self.dataset_api(is_train) ## Build dataset and iterator given training flag
        sess.run(self.init_op)

    def get_value(self):
        return self.next_batch

class DataLoaderNumpy_Big:
    '''
    1. Read Pickle file that contains train&test
    2. Create dataset with place holder
    3. Create reinitizlizatble iterator agaisnt dataset
    4. Create initialization opertaion with feed_dict
    5. Def method to get batch data
    '''
    def __init__(self, config):
        self.config = config

        with open(os.path.join('data',self.config.project,'data_numpy.pkl'), 'rb') as f:
            self.data = pickle.load(f)

        self.x_train = self.data['x_train']
        self.y_train = self.data['y_train']
        self.x_test = self.data['x_test']
        self.y_test = self.data['y_test']

        print("x_train", self.x_train.shape, self.x_train.dtype)
        print("y_train", self.y_train.shape, self.y_train.dtype)
        print("x_test", self.x_test.shape, self.x_test.dtype)
        print("y_test", self.y_test.shape, self.y_test.dtype)

        self.train_n_sample = self.x_train.shape[0]
        self.test_n_sample = self.x_test.shape[0]

        self.train_iteration = np.floor( (self.train_n_sample-1)/self.config.batch_size) +1
        self.test_iteration = np.floor( (self.test_n_sample-1)/self.config.batch_size) +1

        ##initialize class attributes
        self.feature_placeholder = None
        self.target_placehholder = None
        self.dataset = None
        self.iterator = None
        self.init_op = None
        self.next_batch = None

        self.dataset_api() ##Create dataset, iterator, and Initialization op

    def dataset_api(self):
        with tf.device('/cpu:0'):
        ## set first dimension of placeholder to None for batching
        ## placeholder is used for bigger dataset
            self.feature_placeholder =  tf.placeholder(self.config.feature_type,
                                                       [None] + list(self.x_train.shape[1:]))
            self.target_placeholder = tf.placeholder(self.config.target_type,
                                                     [None] + list(self.y_train.shape[1:]))
            ## create dataset from tensor[]
            self.dataset = tf.data.Dataset.from_tensor_slices((self.feature_placeholder,
                                                              self.target_placeholder))
            ## config Dataset: batch, shuffle
            self.dataset = self.dataset.batch(self.config.batch_size)
            ## Define a reinitializable Iterator by (type, shape)
				 ## Compare with initializalbe Iterator, it can be initialized by different dataset with same structure
            self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types,
                                                       self.dataset.output_shapes)
            ## Create initialize operation for Iterator
            self.init_op = self.iterator.make_initializer(self.dataset)
            ## return a nested structure of tensor
            self.next_batch = self.iterator.get_next()

            print("x_batch: ", self.next_batch[0].shape)
            print("y_batch: ", self.next_batch[1].shape)

    def initialize(self, sess, is_train):
        if is_train:
            print("Initialize Iterator for training")
            self.dataset  = self.dataset.shuffle(self.config.shuffle_size)
#            rows = np.random.choice(self.train_n_sample, self.train_n_sample, replace = False)
#            self.x_train = self.x_train[rows]
#            self.y_train = self.y_train[rows]

            sess.run(self.init_op, feed_dict = {self.feature_placeholder: self.x_train,
                                             self.target_placeholder: self.y_train})
        else:
            print("Initialize Iterator for testing/valdiation")
            sess.run(self.init_op, feed_dict = {self.feature_placeholder: self.x_test,
                                                self.target_placeholder: self.y_test})

    def get_value(self):
        return self.next_batch

