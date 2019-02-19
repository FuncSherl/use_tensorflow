#coding:utf-8
'''
Created on 2018年12月10日

@author: sherl
'''

import cv2,os,random
from datetime import datetime
import os.path as op
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from use_tensor.GAN_slomo.data import create_dataset as cdata

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
train_dir= list( map(lambda x:op.join(cdata.extratdir_train, x) ,os.listdir(cdata.extratdir_train)) )
test_dir=list( map(lambda x:op.join(cdata.extratdir_test, x) ,os.listdir(cdata.extratdir_test)) )

target_imgh=360
target_imgw=640
#print (len(train_dir))

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def get_train_batchdata(batchsize=10, num_each=3):
    '''
    notice batchsize not bigger than len(train_dir)
    return_shape [batchsize, num_each, img_h, img_w, 3]
    '''
    ret=np.zeros([batchsize, num_each, target_imgh, target_imgw,3])
    mv_list=random.sample(train_dir, batchsize)
    for ind,i in enumerate(mv_list):
        tep=os.listdir(i)
        tep.sort()
        selectind=random.randint(0,len(tep)-num_each)
        for j in range(num_each):
            frame=cv2.imread(op.join(i, tep[selectind+j]))
            #print (frame.shape)
            frame=cv2.resize(frame, (target_imgw, target_imgh))
            '''
            cv2.imshow('test',frame)
            cv2.waitKey(0) 
            '''
            ret[ind, j, :]=frame
    return ret
        
    
    



if __name__ == '__main__':
    get_train_batchdata(10,3)
    
    
    