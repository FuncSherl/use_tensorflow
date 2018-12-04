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

div_step=0.002
num_step=1000
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
        
    def get_onedimval(self, point,dim, num=num_step, step=div_step):
        '''
        :跑模型得出结果
        '''
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
        #print (ret.shape)
        return ret
    
    
    def get_values(self, point, num=num_step, step=div_step, class_choose=0):
        '''
        :以point为基本点进行泰勒拟合，num是取多少点，step是数据间隔，注意有可能因为计算机精确度的原因加上step后结果并不改变
        '''
        kep_val=np.zeros([num]*len(point))
        tp=np.array(point, dtype=np.float32)
        
        for i in range(num):
            tp[0]=point[0]+(i-num//2)*step
            
            print('\n',i,'/',num,'-->',tp)
            tep=self.get_onedimval(tp, 1, num=num, step=step)
            kep_val[i]=tep[:,class_choose]
            
        return kep_val
            
        
    
        '''
        print (tep)
        #fig = plt.figure() 
        plt.scatter(list(range(len(tep))),tep[:,0],s=1,marker='.')
        plt.scatter(list(range(len(tep))),tep[:,1], color="orange",s=1,marker='.')
        plt.scatter(list(range(len(tep))),tep[:,1]+tep[:,0], color="red",s=1,marker='.')
        plt.show()
        '''
        
    def get_all_derivative(self,point, num=num_step, step=div_step):
        '''
        :这里以2位point为例,考虑利用递归
        '''
        all_der=np.zeros([num]*len(point))#这里行列分别代表x的和y的n次导数，虽然应有f(x,y)dxy=f(x,y)dyx但可能不连续
        print (all_der.shape)
        
        
        for i in range(num):
            for j in range(num):
                pass
    def recursive_cal_deribatice(self,indata,all_der,point=[num_step/2,num_step/2],cnt=0):
        '''
        :递归计算行列的导数，对第一行和列，是原数据的求导，对第二行和列，是以f(x,y)dx,y为基础计算，如此可以递归
        indata:输入的数据，一开始输入源数据
        all_der:记录导数的矩阵
        point:输入数据中以那个为中心，同上面的point不同，这里应为[len/2，len/2],注意分别代表x,y
        cnt：当前递归到那一行了
        '''
        
        all_der[cnt,cnt]=indata[point[0],point[1]]
        
        tep_x=indata[point[0]].copy()#固定x的point那一行的data
        tep_y=indata[:,point[1]].copy()
        for i in range(cnt+1, indata.shape[0]):#填充x方向的导数，应该用tep_y
            tep_y=self.cal_derivative(tep_y)
            all_der[i,cnt]=tep_y[point[1]]
        for i in range(cnt+1, indata.shape[1]):#填充y方向的导数，应该用tep_x
            tep_x=self.cal_derivative(tep_x)
            all_der[cnt,i]=tep_x[point[0]]
        
        for i in range(indata.shape[0]):
            indata[i]=self.cal_derivative(indata[i])
        for i in range(indata.shape[1]):
            indata[:,i]=self.cal_derivative(indata[:,i])
            
        if cnt<indata.shape[0]-1:
            self.recursive_cal_deribatice(indata, all_der, point, cnt+1)
        else: return all_der
            
        
        
        
        
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
        #tep.test_derivative()
        tep.get_values([0,0])
        
        
        
        
        
        
    