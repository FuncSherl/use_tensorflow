#coding:utf-8
'''
Created on 2018年10月29日

@author:China
'''

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from datetime import datetime
import time,cv2
import os.path as op

import matplotlib.pyplot as plt

#当前路径下的文件
from  use_tensor.locate_feature.vgg16_ori import *



modelpath='./logs/VOC_2018-10-12_15-51-44_base_lr-0.001000_batchsize-30_maxstep-30000'


class bp_model:
    def __init__(self, sess, modelpath=modelpath):
        self.sess=sess
        self.graph = tf.get_default_graph() 
        self.load_model(modelpath)   
        
        
        self.prob=self.graph.get_tensor_by_name('Softmax:0') 
        self.dat_place=self.graph.get_tensor_by_name('Placeholder:0') 
        self.label_place=self.graph.get_tensor_by_name('Placeholder_1:0') 
        self.training=self.graph.get_tensor_by_name('Placeholder_2:0') 
        
        print(tf.trainable_variables())
        print (self.graph.get_all_collection_keys())
        
        
         
    
    def load_model(self,modelpath):
        saver = tf.train.import_meta_graph(op.join(modelpath,'model_keep-29999.meta'))
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        print ('restore weights done!')
        
    


if __name__ == '__main__':
    with tf.Session() as sess:
        tep=bp_model(sess)
        




