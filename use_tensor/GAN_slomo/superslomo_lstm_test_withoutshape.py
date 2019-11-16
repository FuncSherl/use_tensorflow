'''
Created on Apr 9, 2019

@author: sherl
'''

import tensorflow as tf
import numpy as np
import os.path as op
#import matplotlib.pyplot as plt
import cv2,os,time
from datetime import datetime
import skimage
import imageio
from superslomo_test_withtime import *


modelpath="Pictures/superslomo/SuperSlomo_2019-11-10_21-07-26_base_lr-0.000100_batchsize-10_maxstep-240000_LSTM_version_EvalWith360p"
modelpath="Pictures/superslomo/SuperSlomo_2019-11-13_17-49-29_base_lr-0.000100_batchsize-10_maxstep-240000_LSTM_train_with_crop"
modelpath="Pictures/superslomo/SuperSlomo_2019-11-13_17-28-10_base_lr-0.000100_batchsize-6_maxstep-240000_TrainWith360pVersion"

modelpath=op.join(homepath, modelpath)

version='Superslomo_v2_lstm_DoubleShape_'


class Slomo_step2_LSTM_doubleshape(Slomo_step2):
    def __init__(self,sess, modelpath=modelpath):
        super().__init__( sess, modelpath=modelpath)
        
        self.outimg = self.graph.get_tensor_by_name("second_outputimg_eval:0")
        self.optical_t_0=self.graph.get_tensor_by_name("second_opticalflow_t_0_eval:0")
        self.optical_t_2=self.graph.get_tensor_by_name("second_opticalflow_t_1_eval:0")
        
        #self.occu_mask=self.graph.get_tensor_by_name("prob_flow1_sigmoid:0")
        
        #placeholders
        self.img_pla= self.graph.get_tensor_by_name('imgs_in_eval:0')
        self.training= self.graph.get_tensor_by_name("training_in:0")
        self.timerates= self.graph.get_tensor_by_name("timerates_in:0")
        #self.last_optical_flow=self.graph.get_tensor_by_name("second_last_flow:0")
        
        print ('self.outimg:',self.outimg)
        
        self.optical_flow_shape=self.optical_t_0.get_shape().as_list() #[12, 180, 320, 2]
        #print (self.optical_flow_shape)
        self.placeimgshape=self.img_pla.get_shape().as_list() #[12, 180, 320, 9]
        self.batch=self.placeimgshape[0]
        self.imgshape=(self.placeimgshape[2], self.placeimgshape[1]) #w*h
        
        self.outimgshape=self.outimg.get_shape().as_list() #self.outimgshape: [12, 180, 320, 3]
        self.videoshape=(self.outimgshape[2], self.outimgshape[1]) #w*h

        self.last_optical_flow=self.graph.get_tensor_by_name("second_last_flow_eval:0")
        self.last_optical_flow_shape=self.last_optical_flow.get_shape().as_list()
        
        #self.out_last_flow=self.graph.get_tensor_by_name("second_unet/strided_slice_89:0")
        self.out_last_flow=self.graph.get_tensor_by_name("second_unet/second_batch_last_flow_eval:0")
        
        

if __name__=='__main__':
    with tf.Session() as sess:
        slomo=Slomo_step2_LSTM_doubleshape(sess)
        slomo.process_video_list(inputvideo, outputvideodir, 6, version)
        #slomo.eval_video_list(inputvideo,  2)
        #slomo.eval_on_middlebury_allframes(middleburey_path)
        
        
        
        
        
        
    
         
    