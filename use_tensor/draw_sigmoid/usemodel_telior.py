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

div_step=0.00001
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
        
    def get_onedimval(self, point,dim=0, num=1000, step=div_step):
        ret=[]
        point=np.array(point)
        start=point[dim]-num*step/2
        dat=np.zeros([batchsize, len(point)])
        print('point:',point)
        print('dim:',dim,'  from:',start, '  to  ',start+num*step)
        for i in range(num):
            dat[i%batchsize]=point.copy()
            dat[i%batchsize][dim]=start+i*step
            if (i+1)%batchsize==0 or i==(num-1):
                lot=self.sess.run(self.logit, feed_dict={self.dat_place:dat})
                ret.extend(lot[0:i%batchsize+1])
                
        ret=np.array(ret)
        print (ret.shape)
        return ret
    
    def cal_derivative(self, l, cnt=10, step=div_step):
        '''
        :算一个list的数的导数
        l:list of data
        cnt: how many data to cal one derivative
        '''
        ret=[]
        tep=cnt//2
        for i in range(0,len(l)):
            st=max(0, i-tep)
            ed=min(len(l)-1, i+tep)
            cnt_all=0
            for j in range(st,ed):
                cnt_all+=(l[j+1]-l[j])/step
            ret.append(cnt_all/(ed-st))
            
        return np.array(ret)
    
    def test_derivative(self):
        '''
        :对上面的求导函数测试
        '''
        tep=[]
        r=1000
        step=2.0*np.pi/r
        for i in range(r):
            tep.append(math.sin(i*step))
        ret=self.cal_derivative(tep,10, step)
        print(len(tep),len(ret))
        
        plt.grid(True, color = "b")
        plt.scatter(list(range(r)),tep, color="orange",s=1,marker='.')
        plt.scatter(list(range(r)),ret, color="red",s=1,marker='.')
        plt.show()
        
        
    def get_values(self, point, num=1000, step=div_step):
        '''
        :以point为基本点进行泰勒拟合，num是取多少点，step是数据间隔，注意有可能因为计算机精确度的原因加上step后结果并不改变
        '''
        #kep_val=np.zeros([num]*len(point))
        tep=self.get_onedimval(point, 1, num=num, step=0.003)
    
        print (tep)
        #fig = plt.figure() 
        plt.scatter(list(range(len(tep))),tep[:,0],s=1,marker='.')
        plt.scatter(list(range(len(tep))),tep[:,1], color="orange",s=1,marker='.')
        plt.scatter(list(range(len(tep))),tep[:,1]+tep[:,0], color="red",s=1,marker='.')
        plt.show()
        
        
        
    def eval_model(self):
        cnt_true=0
        cnt_all=0
    
        for i in range(100):
            dat,lab=get_batch_data()
            l=self.sess.run(self.logit, feed_dict={self.dat_place:dat})
            
            cnt_all+=batchsize
            tep=np.sum(np.argmax(l,axis=1)==lab)
            cnt_true+=tep
            #print ('eval one batch:',tep,'/',batchsize,'-->',tep/batchsize)
            
        print ('\neval once, accu:',cnt_true/cnt_all,'\n')
        
    
    def load_model(self,modelpath=modelpath):
        saver = tf.train.import_meta_graph(op.join(modelpath,'model_keep-49999.meta'))
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        print ('restore weights done!')
        
        
        
        
        
if __name__ == '__main__':
    with tf.Session() as sess:
        tep=cal_tailor(sess)
        tep.eval_model()
        #tep.get_onedimval([0,1])
        #tep.get_values([5,0])
        tep.test_derivative()
    