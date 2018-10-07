# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 13:03:45 2018

@author: xiang
"""
from tqdm import tqdm
import imageio
import tensorflow as tf
import numpy as np
import os


def save_imgs_to_disk(path, array, file_names):
    for i,img in tqdm(enumerate(array)):
        imageio.imwrite(os.path.join(path, file_names[i]), img, 'PNG-PIL')

def save_numpy_to_disk(path, array):
    np.save(path, array)

def Bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def Int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def Float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

def save_img_to_tfrecord(path, array_x, array_y):
    with tf.python_io.TFRecordWriter(path) as writer:
        for i in tqdm(range(array_x.shape[0])):
            img_raw = array_x[i].tostring()
            features = tf.train.Features(
                    feature = {'label': Int64_feature(array_y[i]),
                               'img': Bytes_feature(img_raw)
                               })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
