'''
Created on Sep 27, 2019

@author: sherl
'''
import tensorflow as tf
from data import create_dataset2 as cdata
import numpy as np
import cv2, os, random, time

flow_size_w = int (640 / 2)
flow_size_h = int (360 / 2)
flow_size = [flow_size_h, flow_size_w]
flow_channel = 2

batchsize = 2  # 分别对应正向和反向光流
timestep = 12

kernel_len = 3

output_channel = 2

modelpath = "/home/sherl/Pictures/v24/GAN_2019-08-20_21-49-57_base_lr-0.000200_batchsize-12_maxstep-240000_fix_a_bug_BigProgress"
meta_name = r'model_keep-239999.meta'

# 设置GPU显存按需增长
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)


class Step2_ConvLstm:

    def __init__(self, sess):
        self.sess = sess
        
        # 注意这里的placeholder包含正向和反向的光流
        self.opticalflow_pla = tf.placeholder(tf.float32, [batchsize,  flow_size_h, flow_size_w, flow_channel], name='opticalflow_in')
        
        self.cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[flow_size_h, flow_size_w, flow_channel], \
                                                output_channels=output_channel, kernel_shape=[kernel_len, kernel_len])
        self.state = self.cell.zero_state(batch_size=batchsize, dtype=tf.float32)
        
        
        
    def reset(self):
        self.state = self.cell.zero_state(batch_size=batchsize, dtype=tf.float32)

    def forward_once(self, inputs, state):
        output,state_final=self.cell.call(inputs=inputs,state=state)


if __name__ == '__main__':   
    with tf.Session(config=config) as sess:      
        gan = Step2_ConvLstm(sess)
        

        
        
        
        
        
        
        
        
        
        
        
        
        
