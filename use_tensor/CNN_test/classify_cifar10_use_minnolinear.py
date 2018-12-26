#coding:utf-8
'''
Created on 2018年9月9日

@author: sherl
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import cv2,time,os
import os.path as op
from datetime import datetime
import math


TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#-----------------------------------------------------------------------------------------
stdev_init=0.1
lr=0.001
batchsize=100
maxiter=10000
inputshape=[batchsize,1]

logdir="./logs/test_tailor_"+TIMESTAMP+('_base_lr-%f_batchsize-%d_maxstep-%d'%(lr,batchsize, maxiter))
if not op.exists(logdir): os.makedirs(logdir)
######-------------------------------------------------------------------------------------
class test_fit_tailor:
    def __init__(self, sess):
        self.sess=sess
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        
        self.x_pla=tf.placeholder(tf.float32, inputshape, name='x_in')
        self.lab_pla=tf.placeholder(tf.float32, inputshape, name='lab_in')
        
        self.net=self.netstructure(self.x_pla)
        self.loss_all=self.loss()
        self.train_all=self.trainonce()
        
        self.summary_all=tf.summary.merge_all()
        init = tf.global_variables_initializer()#初始化tf.Variable,虽然后面会有初始化权重过程，但是最后一层是要根据任务来的,无法finetune，其参数需要随机初始化
        self.sess.run(init)
        
    def getdata(self):
        tep=np.random.rand(*inputshape)*8-4
        #print (tep)
        lab=np.exp(tep)
        return tep,lab
        
    
    def one_layer(self, cnt, x):
        with tf.variable_scope('layer'+str(cnt), reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable(  'div',[1] , initializer=tf.truncated_normal_initializer(stddev=stdev_init))
            other=tf.pow(x, cnt)/math.factorial(cnt)
            return w*other
            
    
    
    def netstructure(self, x, cnt=10):
        tep=0
        for i in range(cnt):
            tep+=self.one_layer(i, x)
        return tep
    
    def loss(self):
        return tf.abs(tf.reduce_mean(self.lab_pla)-tf.reduce_mean(self.net))
    
    def trainonce(self,decay_steps=100, decay_rate=0.99, beta1=0.5):
        self.lr_rate = tf.train.exponential_decay(lr,  global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate)
        
        #print ('AdamOptimizer to maxmize %d vars..'%(len(self.D_para)))
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op= tf.train.AdamOptimizer(self.lr_rate, beta1=beta1).minimize(self.loss_all,global_step=self.global_step)   #
        return train_op
    
    
        
    
    
    ###################################no tensor---------------------
    def start(self):
        for i in range(maxiter):
            dat,lab=self.getdata()
            _,loss=self.sess.run([self.train_all,self.loss_all], feed_dict={  self.x_pla: dat , self.lab_pla:lab})
            print (i,'/',maxiter,' train once,loss:',loss)
            
            
if __name__=='__main__':
    with tf.Session() as sess: 
        gan=test_fit_tailor(sess)
        gan.start()
            
            
        
        
        
        
    