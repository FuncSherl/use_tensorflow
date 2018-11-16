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
        
        '''
        out: Tensor("conv5_3/BiasAdd:0", shape=(30, 14, 14, 512), dtype=float32)
        self.conv5_3 Tensor("conv5_3:0", shape=(30, 14, 14, 512), dtype=float32)
        '''
        #/fc3////////////////////////////////////////////////////////
        '''
        self.fc3_w=self.graph.get_tensor_by_name('fc3/weights:0')
        self.fc3_b=self.graph.get_tensor_by_name('fc3/biases:0')
        
        self.fc3_out=self.graph.get_tensor_by_name('fc3/BiasAdd:0')
        #self.fc3_relu
        
        #/fc2//////////////////////////////////////////////////////
        self.fc2_w=self.graph.get_tensor_by_name('fc2/weights:0')
        self.fc2_b=self.graph.get_tensor_by_name('fc2/biases:0')
        self.fc2_out=self.graph.get_tensor_by_name('fc2/BiasAdd:0')
        self.fc2_relu=self.graph.get_tensor_by_name('fc2/Relu:0')
        '''
        self.get_onelayer_tensors('fc3')
        self.get_onelayer_tensors('fc2')
        
        #待测试图片
        imgst,labst=self.sess.run([test_imgs, test_labs])
        feed_tep={self.dat_place: imgst,self.label_place: labst, self.training:False}#
        
        
        #///////////////////////////////////////////////////////////////////////////
        w3,b3, out3,relu2=self.sess.run([self.fc3_w, self.fc3_b, self.fc3_out, self.fc2_relu], feed_dict=feed_tep)
        
        print (type(w3))
        batch_select=0
        
        tep=out3[batch_select]
        mx=np.argmax(tep)
        
       
        #print (w3,'\n',b3)
        print (w3.T.shape)
        self.show_weight(w3)
        
        
    def get_layerandlastlayer_max(self,layername,lastlayername, feed_tep):
        now=self.get_onelayer_tensors(layername)#
        last=self.get_onelayer_tensors(lastlayername)#
        
        relu3, w3,b3, relu2=self.sess.run([*now, last[-1]], feed_dict=feed_tep)
        
        return [relu3, w3,b3, relu2]
        
        
    def get_onelayer_tensors(self,name):
        '''
        get one namescope's tensors through name
        '''
        ret=[]
        classname = self.__dict__
        classname[name+'_w']=self.graph.get_tensor_by_name(name+'/weights:0')
        ret.append(classname[name+'_w'])
        
        classname[name+'_b']=self.graph.get_tensor_by_name(name+'/biases:0')
        ret.append(classname[name+'_b'])
        
        classname[name+'_out']=self.graph.get_tensor_by_name(name+'/BiasAdd:0')
        ret.append(classname[name+'_out'])
        
        if name!='fc3':
            classname[name+'_relu']=self.graph.get_tensor_by_name(name+'/Relu:0')
            ret.append(classname[name+'_relu'])
        
        return ret
        
         
    
    def load_model(self,modelpath=modelpath):
        saver = tf.train.import_meta_graph(op.join(modelpath,'model_keep-29999.meta'))
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        print ('restore weights done!')
        
    def show_feature_oneimg(self,fea):
        
        pass
        
    def show_weight(self,w3):
        '''
        w3 shape:[last lauer's shape, out shape] like [4096,20(class num)]
        '''
        cValue = ['r','y','g','b','c','k','m']
        print (type(w3))
        w3=np.mat(w3)
        for ind,i in enumerate(w3.T):    
            tep=i.getA()[0]
            print (tep.shape)
            
            dif=[]
            for j in range(len(tep)-1):
                dif.append(tep[j]-tep[j+1])
            tep=np.sort(tep)
            #plt.scatter(range(len(dif)), dif, c=cValue[ind%len(cValue)],s=1,marker='.')
            plt.scatter(range(len(tep)), tep, c=cValue[(ind+1)%len(cValue)],s=1,marker='.')
            plt.show()
        
    


if __name__ == '__main__':
    with tf.Session() as sess:
        tep=bp_model(sess)
        




