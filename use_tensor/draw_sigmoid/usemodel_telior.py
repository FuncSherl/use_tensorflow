# coding:utf-8
'''
Created on 2018��7��30��

@author: sherl
'''
import tensorflow as tf
import numpy as np
import math,random,time
import matplotlib.pyplot as plt
from datetime import datetime
import os.path as op
from use_tensor.draw_sigmoid.test_draw_sigmoid import *

TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

modelpath='./logs/2018-12-01_13-32-33'

class cal_tailor:
    def __init__(self,sess, modelpath=modelpath):
        self.sess=sess
        self.graph = tf.get_default_graph() 
        self.load_model(modelpath)  
        
        self.logit=self.graph.get_tensor_by_name('softmax_linear/add:0') 
        self.dat_place=self.graph.get_tensor_by_name('input_img:0') 
        self.label_place=self.graph.get_tensor_by_name('input_lab:0') 
        #self.training=self.graph.get_tensor_by_name('Placeholder_2:0') 
        
        for i in tf.trainable_variables():
            print(i)
        print (self.graph.get_all_collection_keys())
        
        
    def eval_model(self):
        cnt_true=0
        cnt_all=0
    
        for i in range(100):
            dat,lab=get_batch_data()
            l=self.sess.run(self.logit, feed_dict={self.dat_place:dat})
            
            cnt_all+=batchsize
            tep=np.sum(np.argmax(l,axis=1)==lab)
            cnt_true+=tep
            print ('eval one batch:',tep,'/',batchsize,'-->',tep/batchsize)
            
        print ('eval once, accu:',cnt_true/cnt_all,'\n')
        
    
    def load_model(self,modelpath=modelpath):
        saver = tf.train.import_meta_graph(op.join(modelpath,'model_keep-49999.meta'))
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        print ('restore weights done!')
        
        
        
        
        
if __name__ == '__main__':
    with tf.Session() as sess:
        tep=cal_tailor(sess)
        tep.eval_model()
    
    
    