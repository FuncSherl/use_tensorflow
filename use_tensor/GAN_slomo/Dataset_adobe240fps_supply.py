#coding:utf-8
'''
Created on 2018年12月10日

@author: sherl
'''

import cv2,os,random
from datetime import datetime
import os.path as op
#from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
from data import create_dataset as cdata

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
train_dir= list( map(lambda x:op.join(cdata.extratdir_train, x) ,os.listdir(cdata.extratdir_train)) )
test_dir=list( map(lambda x:op.join(cdata.extratdir_test, x) ,os.listdir(cdata.extratdir_test)) )

target_imgh=360
target_imgw=640
#print (len(train_dir))
img_channel=3 #图像为3 channel
#train和test中的frame总数
test_frames_sum=8508
train_frames_sum=112064

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def get_train_batchdata(batchsize=10, num_each=3):
    '''
    notice batchsize not bigger than len(train_dir)
    return_shape [batchsize,  img_h, img_w, img_channel*num_each]
    '''
    ret=np.zeros([batchsize,  target_imgh, target_imgw,img_channel*num_each])
    mv_list=random.sample(train_dir, batchsize)
    for ind,i in enumerate(mv_list):
        tep=os.listdir(i)
        tep.sort()
        selectind=random.randint(0,len(tep)-num_each)
        for j in range(num_each):
            frame=cv2.imread(op.join(i, tep[selectind+j]))
            #print (frame.shape)
            frame=cv2.resize(frame, (target_imgw, target_imgh))
            ret[ind, :, :, img_channel*j:img_channel*j+img_channel]=frame
            '''
            print (ret[ind,0,0,img_channel*j:img_channel*j+img_channel])
            print (frame[0,0,:])
            cv2.imshow('test',ret[ind, :, :, img_channel*j:img_channel*j+img_channel].astype(np.uint8))
            cv2.waitKey(0) 
            '''
    return ret


def get_test_batchdata(batchsize=10, num_each=3):
    '''
    notice batchsize not bigger than len(train_dir)
    return_shape [batchsize,  img_h, img_w, img_channel*num_each]
    '''
    ret=np.zeros([batchsize, target_imgh, target_imgw,img_channel*num_each])
    mv_list=random.sample(test_dir, batchsize)
    for ind,i in enumerate(mv_list):
        tep=os.listdir(i)
        tep.sort()
        selectind=random.randint(0,len(tep)-num_each)
        for j in range(num_each):
            frame=cv2.imread(op.join(i, tep[selectind+j]))
            #print (frame.shape)
            frame=cv2.resize(frame, (target_imgw, target_imgh))
            ret[ind, :, :, img_channel*j:img_channel*j+img_channel]=frame
            '''
            cv2.imshow('test',ret[ind,:,:,img_channel*j:img_channel*j+img_channel])
            cv2.waitKey(0)
            '''
    return ret
        
    
    



if __name__ == '__main__':
    res=get_train_batchdata(10,3)
    print (res.shape)
    
    
    
