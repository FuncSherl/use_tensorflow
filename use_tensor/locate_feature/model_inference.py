#coding:utf-8
'''
Created on 2018年10月13日

@author:China

利用擦除法来测试图像关键区域
'''

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from datetime import datetime
import time,cv2,os
import os.path as op

import matplotlib.pyplot as plt

#当前路径下的文件
from  use_tensor.locate_feature.vgg16_ori import *



modelpath='./logs/VOC_2018-10-30_10-01-02_base_lr-0.001000_batchsize-30_maxstep-30000'

if __name__ == '__main__':
    #plt.ion()
    fig=plt.figure()
    #oriimg = fig.add_subplot(121)
    #heaimg = fig.add_subplot(122)

class test_model:
    def __init__(self, sess, modelpath=modelpath):
        self.sess=sess
        self.graph = tf.get_default_graph() 
        self.load_model(modelpath)   
        self.kernel_rate=0.5
        
        self.prob=self.graph.get_tensor_by_name('Softmax:0') 
        self.dat_place=self.graph.get_tensor_by_name('Placeholder:0') 
        self.label_place=self.graph.get_tensor_by_name('Placeholder_1:0') 
        self.training=self.graph.get_tensor_by_name('Placeholder_2:0') 
         
    
    def load_model(self,modelpath):
        saver = tf.train.import_meta_graph(op.join(modelpath,'model_keep-29999.meta'))
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        print ('restore weights done!')
        
            
    
    def getoneimg_probs(self,stride=(1,1), kernel_rate=0.5):
        '''
        stride(h,w)
        kernel_rate:遮挡部分占图像大小的比例
        
        :将一个黑框以卷积的形式滑动，分别输出对应被遮挡一部分图片的prob，每个位置都有一个prob，形成一个小点的图片，每个点代表该点为黑框中心时的prob
        '''
        self.kernel_rate=kernel_rate
        
        imgst,labst=self.sess.run([test_imgs, test_labs])
        
        batchsize=imgst.shape[0]
        
        kep_img=imgst[0].copy()
        
        shape=kep_img.shape
        kernel_h=int(self.kernel_rate*shape[0])#根据rate计算kernel大小，也即遮挡部分的大小
        kernel_w=int(self.kernel_rate*shape[1])
        
        cnt_h=int((shape[0]-kernel_h)/stride[0]+1)#计算卷积后的大小
        cnt_w=int((shape[1]-kernel_w)/stride[1]+1)
        
        ret_prob=[]
        
        #print ('label[0]:',labst[0])
        
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
        
        #原图、prob的卷积后大小的图，每个点是20长度的prob、原图的prob、gt标签
        return kep_img,minedprob,oriprob,labst[0]
    
    def getprob_dis(self,minedprob, oriprob, gt):
        '''
        important:算法实现处
        get the distance between origin prob and that after mined some area
        '''
        return oriprob[gt]-minedprob[gt]
    
    
    
    def proc_probs(self):
        img,minedprob,oriprob,gt=self.getoneimg_probs(kernel_rate=0.2)
        shape=minedprob.shape
        heatmap=np.zeros([shape[0],shape[1]])
        for i in range(shape[0]):
            for j in range(shape[1]):
                heatmap[i][j]=self.getprob_dis(minedprob[i][j], oriprob, gt)
                
        heatmap=self.normalize(heatmap)
        
        #print (heatmap)
        print ('original probs:',oriprob)
        print ('groundtruth:',proc_voc.classes[gt],'->',gt,'->',oriprob[gt],':',oriprob.argmax()==gt)
        
        '''
        if not oriprob.argmax()==gt: 
            print ('this img is not inferenced right, skiping imshow....\n')
            return
        '''
        teptime="{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
        nam=teptime+'_'+proc_voc.classes[gt]+"_"+str(oriprob[gt])+'_'+str(oriprob.argmax()==gt)
        
        
        plt.subplot(121)
        plt.imshow(img)
        #oriimg.cla()
        #oriimg.imshow(img)
        
        plt.subplot(122)
        plt.imshow(heatmap,cmap=plt.get_cmap("gray"))
        #heaimg.cla()
        #heaimg.imshow(heatmap,cmap=plt.get_cmap("gray"))
        #plt.show()
        #plt.suptitle()
        #plt.pause(3)
        dirname=TIMESTAMP+'-heating_maps-rate_'+str(self.kernel_rate)
        if not op.exists(dirname):
            print ('making dir:',dirname)
            os.mkdir(dirname)
        
        
        plt.savefig(op.join(dirname,nam+'.png'))
        
    def normalize(self,img):
        min=np.min(img)
        max=np.max(img)
        
        tep=(img-min)/float(max-min)
        
        return tep
    

if __name__ == '__main__':
    with tf.Session() as sess:
        tep=test_model(sess)
        while 1:
            tep.proc_probs()
    #plt.ioff()
        
    
    
    
    
    
    
    
    
    