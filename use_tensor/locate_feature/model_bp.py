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



modelpath='./logs/VOC_2018-11-02_20-45-58_base_lr-0.001000_batchsize-30_maxstep-30000'


class bp_model:
    def __init__(self, sess, modelpath=modelpath):
        self.sess=sess
        self.graph = tf.get_default_graph() 
        self.load_model(modelpath)   
        
        
        self.prob=self.graph.get_tensor_by_name('Softmax:0') 
        self.dat_place=self.graph.get_tensor_by_name('Placeholder:0') 
        self.label_place=self.graph.get_tensor_by_name('Placeholder_1:0') 
        self.training=self.graph.get_tensor_by_name('Placeholder_2:0') 
        
        for i in tf.trainable_variables():
            print(i)
        print (self.graph.get_all_collection_keys())
        
        self.fc3_w=self.graph.get_tensor_by_name('fc3/weights:0')
        self.fc3_b=self.graph.get_tensor_by_name('fc3/biases:0')
        
        w3,b3=self.sess.run([self.fc3_w, self.fc3_b])
        print (w3,'\n',b3)
        
        self.show_weight(w3)
         
    
    def load_model(self,modelpath=modelpath):
        saver = tf.train.import_meta_graph(op.join(modelpath,'model_keep-29999.meta'))
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        print ('restore weights done!')
        
    def show_weight(self,w3):
        cValue = ['r','y','g','b','c','k','m']
        print (type(w3))
        w3=np.mat(w3)
        for ind,i in enumerate(w3.T):    
            tep=i.getA()[0]
            print (tep.shape)
            
            dif=[]
            for j in range(len(tep)-1):
                dif.append(tep[j]-tep[j+1])
                
            plt.scatter(range(len(dif)), dif, c=cValue[ind%len(cValue)],s=1,marker='.')
            plt.scatter(range(len(tep)), tep, c=cValue[(ind+1)%len(cValue)],s=1,marker='.')
            plt.show()
        
    


if __name__ == '__main__':
    with tf.Session() as sess:
        tep=bp_model(sess)
        




