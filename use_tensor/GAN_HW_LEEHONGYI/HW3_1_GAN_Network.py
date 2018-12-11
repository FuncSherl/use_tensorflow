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
        
        self.G_para=[]
        self.D_para=[]       
        self.dropout=0.5 
        self.leakyrelurate=0.2
        self.stddev=0.1
        self.bias_init=0
        
        #3个placeholder， img和noise
        self.noise_pla=tf.placeholder(tf.float32, [batchsize, noise_size], name='noise_in')
        self.imgs_pla = tf.placeholder(tf.float32, [batchsize, img_size, img_size, 3], name='imgs_in')
        self.training=tf.placeholder(tf.bool, name='training_in')
        
    def get_noise(self):
        return np.random.random([batchsize, noise_size])
        
    
    def Generator_net(self, noise):
        # fc1
        with tf.variable_scope('G_fc1',  reuse=tf.AUTO_REUSE) as scope:                    
            G_fc1w = tf.get_variable('weights', [noise_size, 128*16*16], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=self.stddev))
            G_fc1b = tf.get_variable('bias', [128*16*16], dtype=tf.float32, initializer=tf.constant_initializer(self.bias_init))
        
            G_fc1l = tf.nn.bias_add(tf.matmul(noise, G_fc1w), G_fc1b)
            
            self.G_fc1 = tf.nn.leaky_relu(G_fc1l, self.leakyrelurate)
            self.G_para += [G_fc1w, G_fc1b]
        
        #dropout1
        #self.G_fc1=tf.cond(self.training, lambda: tf.nn.dropout(self.G_fc1, self.dropout), lambda: self.G_fc1)
        
        #reshape
        self.G_fc1=tf.reshape(self.G_fc1, [-1, 16,16,128])
        
        #deconv1
        with tf.variable_scope('G_deconv1',  reuse=tf.AUTO_REUSE) as scope:  
            kernel=tf.get_variable('weights', [3,3, 128, 128], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias=tf.get_variable('bias', [128], dtype=tf.float32, initializer=tf.constant_initializer(self.bias_init))
            #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
            #而tf.nn.conv2d_transpose中的filter参数，是[filter_height, filter_width, out_channels，in_channels]的形式
            deconv=tf.nn.conv2d_transpose(self.G_fc1, kernel, output_shape=[batchsize, 32, 32, 128], stride=[1,2,2,1], padding="SAME")
            self.G_deconv1=tf.nn.bias_add(deconv, bias)
            
            self.G_para += [kernel, bias]
            #self.G_deconv1=tf.nn.leaky_relu(self.G_deconv1, self.leakyrelurate)
            
        #conv1
        with tf.variable_scope('G_conv1',  reuse=tf.AUTO_REUSE) as scope: 
            kernel=tf.get_variable('weights', [4,4, 128, 128], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias=tf.get_variable('bias', [128], dtype=tf.float32, initializer=tf.constant_initializer(self.bias_init))
            
            conv=tf.nn.conv2d(self.G_deconv1, kernel, stride=[1,1,1,1], padding='SAME')
            self.G_conv1=tf.nn.bias_add(conv, bias)
            
            self.G_para += [kernel, bias]
            self.G_conv1=tf.nn.leaky_relu(self.G_conv1, self.leakyrelurate)
            
        #deconv2
        with tf.variable_scope('G_deconv2',  reuse=tf.AUTO_REUSE) as scope:  
            kernel=tf.get_variable('weights', [3,3, 64, 128], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias=tf.get_variable('bias', [64], dtype=tf.float32, initializer=tf.constant_initializer(self.bias_init))
            #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
            #而tf.nn.conv2d_transpose中的filter参数，是[filter_height, filter_width, out_channels，in_channels]的形式
            deconv=tf.nn.conv2d_transpose(self.G_conv1, kernel, output_shape=[batchsize, 64, 64, 64], stride=[1,2,2,1], padding="SAME")
            self.G_deconv2=tf.nn.bias_add(deconv, bias)
            
            self.G_para += [kernel, bias]
            #self.G_deconv2=tf.nn.leaky_relu(self.G_deconv2, self.leakyrelurate)
            
        #conv2
        with tf.variable_scope('G_conv2',  reuse=tf.AUTO_REUSE) as scope: 
            kernel=tf.get_variable('weights', [4,4, 64, 64], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias=tf.get_variable('bias', [64], dtype=tf.float32, initializer=tf.constant_initializer(self.bias_init))
            
            conv=tf.nn.conv2d(self.G_deconv2, kernel, stride=[1,1,1,1], padding='SAME')
            self.G_conv2=tf.nn.bias_add(conv, bias)
            
            self.G_para += [kernel, bias]
            self.G_conv2=tf.nn.leaky_relu(self.G_conv2, self.leakyrelurate)
            
        #conv3
        with tf.variable_scope('G_conv3',  reuse=tf.AUTO_REUSE) as scope: 
            kernel=tf.get_variable('weights', [4,4, 64, 3], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias=tf.get_variable('bias', [3], dtype=tf.float32, initializer=tf.constant_initializer(self.bias_init))
            
            conv=tf.nn.conv2d(self.G_conv2, kernel, stride=[1,1,1,1], padding='SAME')
            self.G_conv3=tf.nn.bias_add(conv, bias)
            
            self.G_para += [kernel, bias]
            #self.G_conv3=tf.nn.leaky_relu(self.G_conv3, self.leakyrelurate)
            
        #tanh
        G_tanh= tf.nn.tanh(self.G_conv3, name='G_tanh')
        
        return G_tanh
            
        
            
    
    def Discriminator_net(self, imgs):
        #conv1
        with tf.variable_scope('D_conv1',  reuse=tf.AUTO_REUSE) as scope: 
            kernel=tf.get_variable('weights', [4,4, 3, 32], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias=tf.get_variable('bias', [32], dtype=tf.float32, initializer=tf.constant_initializer(self.bias_init))
            #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
            #而tf.nn.conv2d_transpose中的filter参数，是[filter_height, filter_width, out_channels，in_channels]的形式
            conv=tf.nn.conv2d(imgs, kernel, stride=[1,2,2,1], padding='SAME')
            self.D_conv1=tf.nn.bias_add(conv, bias)
            
            self.D_para += [kernel, bias]
            self.D_conv1=tf.nn.leaky_relu(self.D_conv1, self.leakyrelurate)
            
        #conv2
        with tf.variable_scope('D_conv2',  reuse=tf.AUTO_REUSE) as scope: 
            kernel=tf.get_variable('weights', [4,4, 32, 64], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias=tf.get_variable('bias', [64], dtype=tf.float32, initializer=tf.constant_initializer(self.bias_init))
            #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
            #而tf.nn.conv2d_transpose中的filter参数，是[filter_height, filter_width, out_channels，in_channels]的形式
            conv=tf.nn.conv2d(self.D_conv1, kernel, stride=[1,2,2,1], padding='SAME')
            self.D_conv2=tf.nn.bias_add(conv, bias)
            
            self.D_para += [kernel, bias]
            self.D_conv2=tf.nn.leaky_relu(self.D_conv2, self.leakyrelurate)
            
        #conv3
        with tf.variable_scope('D_conv3',  reuse=tf.AUTO_REUSE) as scope: 
            kernel=tf.get_variable('weights', [4,4, 64, 128], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias=tf.get_variable('bias', [128], dtype=tf.float32, initializer=tf.constant_initializer(self.bias_init))
            #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
            #而tf.nn.conv2d_transpose中的filter参数，是[filter_height, filter_width, out_channels，in_channels]的形式
            conv=tf.nn.conv2d(self.D_conv2, kernel, stride=[1,2,2,1], padding='SAME')
            self.D_conv3=tf.nn.bias_add(conv, bias)
            
            self.D_para += [kernel, bias]
            self.D_conv3=tf.nn.leaky_relu(self.D_conv3, self.leakyrelurate)
            
        #conv4
        with tf.variable_scope('D_conv4',  reuse=tf.AUTO_REUSE) as scope: 
            kernel=tf.get_variable('weights', [4,4, 128, 256], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=self.stddev))
            bias=tf.get_variable('bias', [256], dtype=tf.float32, initializer=tf.constant_initializer(self.bias_init))
            #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
            #而tf.nn.conv2d_transpose中的filter参数，是[filter_height, filter_width, out_channels，in_channels]的形式
            conv=tf.nn.conv2d(self.D_conv3, kernel, stride=[1,2,2,1], padding='SAME')
            self.D_conv4=tf.nn.bias_add(conv, bias)
            
            self.D_para += [kernel, bias]
            self.D_conv4=tf.nn.leaky_relu(self.D_conv4, self.leakyrelurate)
            
        #flatten
        self.flatten=tf.reshape(self.D_conv4, [batchsize, -1])
        
        # fc1
        with tf.variable_scope('D_fc1',  reuse=tf.AUTO_REUSE) as scope:                    
            D_fc1w = tf.get_variable('weights', [self.flatten.get_shape()[-1], 1], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=self.stddev))
            D_fc1b = tf.get_variable('bias', [1], dtype=tf.float32, initializer=tf.constant_initializer(self.bias_init))
        
            D_fc1l = tf.nn.bias_add(tf.matmul(self.flatten, D_fc1w), D_fc1b)
            
            self.D_fc1 = tf.nn.leaky_relu(D_fc1l, self.leakyrelurate)
            self.D_para += [D_fc1w, D_fc1b]
            
        self.D_sigmoid=tf.nn.sigmoid(self.D_fc1, name='D_sigmoid')
        
        return self.D_sigmoid
        
        




if __name__ == '__main__':
    pass











