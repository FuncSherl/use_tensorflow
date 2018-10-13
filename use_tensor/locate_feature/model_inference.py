#coding:utf-8
'''
Created on 2018年10月13日

@author:China
'''
import tensorflow as tf
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from datetime import datetime
import time,cv2
import os.path as op

#当前路径下的文件
from  use_tensor.locate_feature.vgg16_ori import *



modelpath='./logs/VOC_2018-10-12_15-51-44_base_lr-0.001000_batchsize-30_maxstep-30000'

class test_model:
    def __init__(self, sess, modelpath=modelpath):
        self.sess=sess
        self.graph = tf.get_default_graph() 
        self.load_model(modelpath)   
        
        self.prob=self.graph.get_tensor_by_name('Softmax:0') 
        self.dat_place=self.graph.get_tensor_by_name('Placeholder:0') 
        self.label_place=self.graph.get_tensor_by_name('Placeholder_1:0') 
        self.training=self.graph.get_tensor_by_name('Placeholder_2:0') 
         
    
    def load_model(self,modelpath):
        saver = tf.train.import_meta_graph(op.join(modelpath,'model_keep-29999.meta'))
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        print ('restore weights done!')
        
            
    
    def testoneimg(self,stride=(1,1), kernel_rate=0.5):
        #stride(h,w)
        imgst,labst=self.sess.run([test_imgs, test_labs])
        
        batchsize=imgst.shape[0]
        
        kep_img=imgst[0].copy()
        
        shape=kep_img.shape
        kernel_h=int(kernel_rate*shape[0])
        kernel_w=int(kernel_rate*shape[1])
        
        cnt_h=int((shape[0]-kernel_h)/stride[0]+1)
        cnt_w=int((shape[1]-kernel_w)/stride[1]+1)
        
        ret_prob=[]
        
        print ('label[0]:',labst[0])
        
        for i in range(cnt_h):
            for j in range(cnt_w):                
                cnt=i*cnt_w+j
                ind=(cnt+1)%batchsize#这里加上1是为了将第一个原图也测试一下，即第一次inference的时候第一张为原图
                #print('\ni=',i,'  j=',j,'  cnt=',cnt)
                
                imgst[ind]=kep_img.copy()
                imgst[ind][i:i+kernel_h, j:j+kernel_w]=[0,0,0]
                
                if ind==(batchsize-1) or (i==(cnt_h-1) and j==(cnt_w-1)):        
                    prob=self.sess.run([self.prob], feed_dict={self.dat_place: imgst,self.label_place: labst, self.training:False})[0]
                    
                    
                    ret_prob.extend(prob[:ind+1])
                    print (cnt,'/',cnt_h*cnt_w,'  ret shape:',np.array(ret_prob).shape)
                    '''
                    for k in imgst:
                        cv2.imshow('test', cv2.cvtColor(k, cv2.COLOR_RGB2BGR))
                        cv2.waitKey()
                    '''
        oriprob,minedprob=np.array(ret_prob[0]) ,np.array(ret_prob[1:])
        
        print ('cnt_w:',cnt_w, '  cnt_h:',cnt_h)
        print('orishape:',oriprob.shape,'  mined:',minedprob.shape)
        
        minedprob=minedprob.reshape([cnt_h,cnt_w,minedprob.shape[-1]])
        
        return minedprob,oriprob,labst[0]
    
    

if __name__ == '__main__':
    with tf.Session() as sess:
        tep=test_model(sess)
        tep.testoneimg()
        
    
    
    
    
    
    
    
    
    