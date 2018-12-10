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
        
        #两个placeholder， img和noise
        self.noise_pla=tf.placeholder(tf.float32, [batchsize, noise_size], name='noise_in')
        self.imgs_pla = tf.placeholder(tf.float32, [batchsize, img_size, img_size, 3], name='imgs_in')
        
    def get_noise(self):
        
        
    
    def Generator_net(self, noise):
        pass
    
    def Discriminator_net(self, imgs):
        pass
        




if __name__ == '__main__':
    pass











