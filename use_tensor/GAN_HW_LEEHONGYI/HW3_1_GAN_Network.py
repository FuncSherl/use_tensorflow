#coding:utf-8
'''
Created on 2018年12月10日

@author: sherl
'''
from xml.dom import minidom
import cv2,os,random
from datetime import datetime
import os.path as op
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from use_tensor import GAN_HW_LEEHONGYI
import use_tensor.GAN_HW_LEEHONGYI.HW3_1_gen_anime_tfrecord as anime_data


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

batchsize=32
noise_size=100
img_size=96

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


class GAN_Net:
    def __init__(self, sess):
        self.sess=sess
        self.tf_inimg=anime_data.read_tfrecord_batch( batchsize=batchsize)  #tensor的img输入
        self.training_G=tf.placeholder(tf.bool)
        self.training_D=tf.placeholder(tf.bool)
        
        #两个placeholder， img和noise
        self.noise_pla=tf.placeholder(tf.float32, [batchsize, noise_size], name='noise_in')
        self.imgs_pla = tf.placeholder(tf.float32, [batchsize, img_size, img_size, 3], name='imgs_in')
        
    def get_noise(self):
        return np.random.random([batchsize, noise_size])
        
    
    def Generator_net(self, noise):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))#除去batch维度，剩下的乘积，用于flatten原来的二维featuremap
            
            
            
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]
        
        #dropout1
        self.fc1=tf.cond(self.training, lambda: tf.nn.dropout(self.fc1, self.dropout), lambda: self.fc1)
        pass
    
    def Discriminator_net(self, imgs):
        pass
        




if __name__ == '__main__':
    pass











