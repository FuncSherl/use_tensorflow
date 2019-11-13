#coding:utf-8
'''
Created on Oct 30, 2019

@author: sherl
'''
'''
implement super slomo(https://arxiv.org/abs/1712.00080) with tensorflow
'''
import cv2,os,random,time
from datetime import datetime
import os.path as op
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from superslomo_tools import *
from data import create_dataset_step2 as cdata
#from data import create_dataset2 as cdata
#import skimage

#this version change output of g to be img
#and chagne img-size to v2's 1/2
#use wgan loss function
#my_novel_conv
#use l2 loss for img clear
#use global rate to mult square loss
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print ('tensorflow version:',tf.__version__,'  path:',tf.__path__)
TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

train_size=10100 
test_size=800
batchsize=6  #train
batchsize_test=batchsize #here it must equal to batchsize,or the placement size will error

#
img_size_w=int (640)
img_size_h=int (360)
img_size=[img_size_h, img_size_w]

base_lr=0.0001 #基础学习率
beta1=0.5
dropout_rate=0.5

maxstep=240000 #训练多少次

decay_steps=12000
decay_rate=0.99

weightclip_min=-0.01
weightclip_max=0.01

#incase_div_zero=1e-10  #这个值大一些可以避免d训得太好，也避免了g梯度

#G_first_channel=12  #不是G的输入channel，而是g的输入经过一次卷积后的输出channel
#D_first_channel=18

#G中unet的层数
G_unet_layercnt=3
G_filter_len=3
G_withbias=True
#G的输出的channel，如果只有双向光流则为4，加上一个mask的话为5
G_optical_channel=4
#the G_squareloss is reducing,at iter 170000, D_loss=2  but square_loss is 0.03,so there requires a rate to make square_loss more clear
G_squareloss_rate_globalstep=8000 

#两个D的用的D_block的层数，即缩小几回
D_1_layercnt=4
D_1_filterlen=3
D_1_withbias=True

D_2_layercnt=D_1_layercnt
D_2_filterlen=3
D_2_withbias=True

#一次输入网络多少图片，这里设定为3帧，利用前后帧预测中间
G_group_img_num=3
img_channel=3
eval_step=int (test_size/batchsize/G_group_img_num)
mean_dataset=[102.1, 109.9, 110.0]  #0->1 is [0.4, 0.43, 0.43]

logdir="./logs_superslomo/SuperSlomo_"+TIMESTAMP+('_base_lr-%f_batchsize-%d_maxstep-%d'%(base_lr,batchsize, maxstep))

kepimgdir=op.join(logdir, "zerostateimgs")
os.makedirs(kepimgdir,  exist_ok=True)

#设置GPU显存按需增长
gpu_options = tf.GPUOptions(allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class SuperSlomo:
    def __init__(self, sess):
        self.sess=sess
        self.global_step = tf.Variable(0.0, name='global_step',dtype=tf.float32, trainable=False)
        
        #for data input
        self.pipline_data_train=cdata.get_pipline_data_train(img_size, batchsize)
        self.pipline_data_test=cdata.get_pipline_data_test(img_size, batchsize_test)
        
        #3个placeholder， img和noise,training 
        self.imgs_pla = tf.placeholder(tf.float32, [batchsize, img_size_h, img_size_w, G_group_img_num*img_channel], name='imgs_in')
        self.imgs_pla_eval = tf.placeholder(tf.float32, [batchsize, img_size_h*2, img_size_w*2, G_group_img_num*img_channel], name='imgs_in_eval')
        self.training=tf.placeholder(tf.bool, name='training_in')  #这里没用上但是为了兼容就保留了
        self.timerates_pla=tf.placeholder(tf.float32, [batchsize], name='timerates_in')
        self.timerates_expand=tf.expand_dims(self.timerates_pla, -1)
        self.timerates_expand=tf.expand_dims(self.timerates_expand, -1)
        self.timerates_expand=tf.expand_dims(self.timerates_expand, -1) #12*1*1*1
        
        print ('placeholders:\n','img_placeholder:',self.imgs_pla,self.timerates_pla)
        #img_placeholder: Tensor("imgs_in:0", shape=(10, 180, 320, 9), dtype=float32) Tensor("timerates_in:0", shape=(10,), dtype=float32)
        
        self.frame0=self.imgs_pla[:,:,:,:img_channel]
        self.frame1=self.imgs_pla[:,:,:,img_channel:img_channel*2]
        self.frame2=self.imgs_pla[:,:,:,img_channel*2:]
        
        #这里用来进行evaluate注意这里只用于输出最终图        
        self.frame0_eval=self.imgs_pla_eval[:,:,:,:img_channel]
        self.frame1_eval=self.imgs_pla_eval[:,:,:,img_channel:img_channel*2]
        self.frame2_eval=self.imgs_pla_eval[:,:,:,img_channel*2:]
        
        with tf.variable_scope("first_unet",  reuse=tf.AUTO_REUSE) as scope:
            firstinput=tf.concat([self.frame0, self.frame2], -1)
            self.first_opticalflow=my_unet( firstinput, 4,training=self.training , withbias=True, withbn=False)  #注意这里是直接作为optical flow
            
            firstinput=tf.concat([self.frame0_eval, self.frame2_eval], -1)
            self.first_opticalflow_eval=my_unet( firstinput, 4,training=self.training , withbias=True, withbn=False)  #注意这里是直接作为optical flow
            #self.first_opticalflow=my_unet_split( firstinput, 4,training=self.training , withbias=True, withbn=True)  #注意这里是直接作为optical flow
            
        self.first_opticalflow_0_1=self.first_opticalflow[:, :, :, :2]
        self.first_opticalflow_0_1=tf.identity(self.first_opticalflow_0_1, name="first_opticalflow_0_1")
        print ('first_opticalflow_0_1:',self.first_opticalflow_0_1)
        self.first_opticalflow_1_0=self.first_opticalflow[:, :, :, 2:]
        self.first_opticalflow_1_0=tf.identity(self.first_opticalflow_1_0, name="first_opticalflow_1_0")
        print ('first_opticalflow_1_0:',self.first_opticalflow_1_0)
        #first_opticalflow_0_1: Tensor("first_opticalflow_0_1:0", shape=(10, 180, 320, 2), dtype=float32)
        #first_opticalflow_1_0: Tensor("first_opticalflow_1_0:0", shape=(10, 180, 320, 2), dtype=float32)
        self.first_opticalflow_0_1_eval=self.first_opticalflow_eval[:, :, :, :2]
        self.first_opticalflow_1_0_eval=self.first_opticalflow_eval[:, :, :, 2:]
        
        #输出光流形状
        self.flow_size_h=self.first_opticalflow_0_1.get_shape().as_list()[1]
        self.flow_size_w=self.first_opticalflow_0_1.get_shape().as_list()[2]
        self.flow_channel=self.first_opticalflow_0_1.get_shape().as_list()[-1]
        
        #eval shape
        self.flow_size_h_eval=self.first_opticalflow_eval.get_shape().as_list()[1]
        self.flow_size_w_eval=self.first_opticalflow_eval.get_shape().as_list()[2]
        
        ########################################################
        self.step2_flow_channel=5
        self.flow_shape=[ self.flow_size_h, self.flow_size_w, self.step2_flow_channel]
        
        #lstm的每个状态（c，h）的形状
        self.state_shape=[2, 1, self.flow_size_h, self.flow_size_w, self.step2_flow_channel]
        self.state_shape_eval=[2, 1, self.flow_size_h_eval, self.flow_size_w_eval, self.step2_flow_channel]
        
        #获取数据时的一些cpu上的参数，用于扩张数据和判定时序
        self.last_flow_init_np=np.zeros(self.state_shape, dtype=np.float32)
        self.last_flow_init_np_eval=np.zeros(self.state_shape_eval, dtype=np.float32)
        print (self.last_flow_init_np.shape, self.last_flow_init_np_eval.shape) #(2, 1, 180, 320, 5) (2, 1, 360, 640, 5)
        ##############################################################
        
        self.last_optical_flow=tf.placeholder(tf.float32, self.state_shape, name='second_last_flow')
        self.last_optical_flow_eval=tf.placeholder(tf.float32, self.state_shape_eval, name='second_last_flow_eval')
        
        #初始化train和test的初始0状态
        self.last_flow_new_train=self.last_flow_init_np
        self.last_flow_new_test=self.last_flow_init_np
        
        #反向光流算中间帧
        self.first_opticalflow_t_0=tf.add( -(1-self.timerates_expand)*self.timerates_expand*self.first_opticalflow_0_1 ,\
                                      self.timerates_expand*self.timerates_expand*self.first_opticalflow_1_0 , name="first_opticalflow_t_0")
        self.first_opticalflow_t_2=tf.add( (1-self.timerates_expand)*(1-self.timerates_expand)*self.first_opticalflow_0_1 ,\
                                      self.timerates_expand*(self.timerates_expand-1)*self.first_opticalflow_1_0, name="first_opticalflow_t_2")
        
        #反向光流算中间帧
        self.first_opticalflow_t_0_eval=tf.add( -(1-self.timerates_expand)*self.timerates_expand*self.first_opticalflow_0_1_eval ,\
                                      self.timerates_expand*self.timerates_expand*self.first_opticalflow_1_0_eval , name="first_opticalflow_t_0_eval")
        self.first_opticalflow_t_2_eval=tf.add( (1-self.timerates_expand)*(1-self.timerates_expand)*self.first_opticalflow_0_1_eval ,\
                                      self.timerates_expand*(self.timerates_expand-1)*self.first_opticalflow_1_0_eval, name="first_opticalflow_t_2_eval")

        #2种方法合成t时刻的帧
        self.first_img_flow_2_t=self.warp_op(self.frame2, -self.first_opticalflow_t_2) #!!!
        self.first_img_flow_0_t=self.warp_op(self.frame0, -self.first_opticalflow_t_0) #!!!
        
        #2种方法合成t时刻的帧
        self.first_img_flow_2_t_eval=self.warp_op(self.frame2_eval, -self.first_opticalflow_t_2_eval) #!!!
        self.first_img_flow_0_t_eval=self.warp_op(self.frame0_eval, -self.first_opticalflow_t_0_eval) #!!!
        
        #虽然论文里用不到第一步的输出中间帧，但是这里也给他输出看看效果
        self.first_output=tf.add( self.timerates_expand*self.first_img_flow_2_t, (1-self.timerates_expand)*self.first_img_flow_0_t , name="first_outputimg")
        print ('first output img:',self.first_output)
        #first output img: Tensor("first_outputimg:0", shape=(10, 180, 320, 3), dtype=float32)
        
        #利用光流前后帧互相合成
        self.first_img_flow_2_0=self.warp_op(self.frame2, -self.first_opticalflow_0_1)  #frame2->frame0
        self.first_img_flow_0_2=self.warp_op(self.frame0, -self.first_opticalflow_1_0)  #frame0->frame2
        
        ####################################################################################################################3
        #第二个unet
        with tf.variable_scope("second_unet",  reuse=tf.AUTO_REUSE) as scope:
            secinput=tf.concat([self.frame0, self.frame2, \
                                self.first_opticalflow_0_1, self.first_opticalflow_1_0, \
                                self.first_opticalflow_t_2, self.first_opticalflow_t_0,\
                                self.first_img_flow_2_t, self.first_img_flow_0_t,\
                                ], -1) #self.last_optical_flow     
            secinput=tf.expand_dims(secinput, 0)
            print ("secinput:",secinput)#secinput: Tensor("second_unet/ExpandDims:0", shape=(1, 10, 180, 320, 20), dtype=float32)  
            
            secinput_eval=tf.concat([self.frame0_eval, self.frame2_eval, \
                                self.first_opticalflow_0_1_eval, self.first_opticalflow_1_0_eval, \
                                self.first_opticalflow_t_2_eval, self.first_opticalflow_t_0_eval,\
                                self.first_img_flow_2_t_eval, self.first_img_flow_0_t_eval,\
                                ], -1) #self.last_optical_flow     
            secinput_eval=tf.expand_dims(secinput_eval, 0)
            print ("secinput_eval:",secinput_eval)#secinput_eval: Tensor("second_unet/ExpandDims_1:0", shape=(1, 10, 360, 640, 20), dtype=float32)
                
            
            
            lstm_input_channel=secinput.get_shape().as_list()[-1]
            self.cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[self.flow_size_h, self.flow_size_w, lstm_input_channel], \
                                                    output_channels=self.step2_flow_channel, kernel_shape=[5, 5])
            
            self.cell_eval = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[self.flow_size_h_eval, self.flow_size_w_eval, lstm_input_channel], \
                                                    output_channels=self.step2_flow_channel, kernel_shape=[5, 5])
            
            lstm_outputs, lstm_state_final = tf.nn.dynamic_rnn(self.cell, inputs =secinput , \
                                            initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.last_optical_flow[0], self.last_optical_flow[1]), time_major = False)
            
            lstm_outputs_eval, lstm_state_final_eval = tf.nn.dynamic_rnn(self.cell_eval, inputs =secinput_eval , \
                                            initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.last_optical_flow_eval[0], self.last_optical_flow_eval[1]), time_major = False)
            
            
            
            self.second_batch_last_flow=tf.stack([lstm_state_final.c, lstm_state_final.h], 0)
            self.second_batch_last_flow=tf.identity(self.second_batch_last_flow, name="second_batch_last_flow")
            print ("second_batch_last_flow:",self.second_batch_last_flow) 
            #second_batch_last_flow: Tensor("second_unet/second_batch_last_flow:0", shape=(2, 1, 180, 320, 5), dtype=float32)
            
            self.second_batch_last_flow_eval=tf.stack([lstm_state_final_eval.c, lstm_state_final_eval.h], 0)
            self.second_batch_last_flow_eval=tf.identity(self.second_batch_last_flow_eval, name="second_batch_last_flow_eval")
            print ("second_batch_last_flow_eval:",self.second_batch_last_flow_eval)
            #Tensor("second_unet/second_batch_last_flow_eval:0", shape=(2, 1, 360, 640, 5), dtype=float32)
            
            self.second_opticalflow=lstm_outputs[0]  
            print ("self.second_opticalflow:",self.second_opticalflow) 
            #self.second_opticalflow: Tensor("second_unet/strided_slice_2:0", shape=(10, 180, 320, 5), dtype=float32)
            
            self.second_opticalflow_eval=lstm_outputs_eval[0]  
            print ("self.second_opticalflow_eval:",self.second_opticalflow_eval)
            #Tensor("second_unet/strided_slice_5:0", shape=(10, 360, 640, 5), dtype=float32)
            
        self.second_opticalflow_t_0=tf.add( self.second_opticalflow[:,:,:,:2],  self.first_opticalflow_t_0, name="second_opticalflow_t_0")
        self.second_opticalflow_t_1=tf.add( self.second_opticalflow[:,:,:,2:4], self.first_opticalflow_t_2, name="second_opticalflow_t_1")
        print ('second_opticalflow_t_0:',self.second_opticalflow_t_0)
        print ('second_opticalflow_t_1:',self.second_opticalflow_t_1)
        #second_opticalflow_t_0: Tensor("second_opticalflow_t_0:0", shape=(10, 180, 320, 2), dtype=float32)
        #second_opticalflow_t_1: Tensor("second_opticalflow_t_1:0", shape=(10, 180, 320, 2), dtype=float32)
        
        self.second_opticalflow_t_0_eval=tf.add( self.second_opticalflow_eval[:,:,:,:2],  self.first_opticalflow_t_0_eval, name="second_opticalflow_t_0_eval")
        self.second_opticalflow_t_1_eval=tf.add( self.second_opticalflow_eval[:,:,:,2:4], self.first_opticalflow_t_2_eval, name="second_opticalflow_t_1_eval")
        print ('second_opticalflow_t_0_eval:',self.second_opticalflow_t_0_eval)
        print ('second_opticalflow_t_1_eval:',self.second_opticalflow_t_1_eval)
        #second_opticalflow_t_0_eval: Tensor("second_opticalflow_t_0_eval:0", shape=(10, 360, 640, 2), dtype=float32)
        #second_opticalflow_t_1_eval: Tensor("second_opticalflow_t_1_eval:0", shape=(10, 360, 640, 2), dtype=float32)
        
        self.vmap_t_0=tf.expand_dims( tf.sigmoid(self.second_opticalflow[:,:,:,-1])  , -1)
        self.vmap_t_1=1-self.vmap_t_0
        
        self.vmap_t_0_eval=tf.expand_dims( tf.sigmoid(self.second_opticalflow_eval[:,:,:,-1])  , -1)
        self.vmap_t_1_eval=1-self.vmap_t_0_eval

        #2种方法合成t时刻的帧
        self.second_img_flow_1_t=self.warp_op(self.frame2, -self.second_opticalflow_t_1) #!!!
        self.second_img_flow_0_t=self.warp_op(self.frame0, -self.second_opticalflow_t_0) #!!!
        
        #2种方法合成t时刻的帧
        self.second_img_flow_1_t_eval=self.warp_op(self.frame2_eval, -self.second_opticalflow_t_1_eval) #!!!
        self.second_img_flow_0_t_eval=self.warp_op(self.frame0_eval, -self.second_opticalflow_t_0_eval) #!!!
        
        #最终输出的图
        print (self.timerates_expand, self.vmap_t_0, self.second_img_flow_0_t)
        #Tensor("ExpandDims_2:0", shape=(10, 1, 1, 1), dtype=float32) Tensor("ExpandDims_3:0", shape=(10, 180, 320, 1), dtype=float32) 
        #Tensor("dense_image_warp_5/Reshape_1:0", shape=(10, 180, 320, 3), dtype=float32)
        self.second_output=tf.div(  ( (1-self.timerates_expand)*self.vmap_t_0*self.second_img_flow_0_t+self.timerates_expand*self.vmap_t_1*self.second_img_flow_1_t),  \
                             ((1-self.timerates_expand)*self.vmap_t_0+self.timerates_expand*self.vmap_t_1) , name="second_outputimg" )
        print ('second output img:',self.second_output)
        #second output img: Tensor("second_outputimg:0", shape=(10, 180, 320, 3), dtype=float32)
        
        self.second_output_eval=tf.div(  ( (1-self.timerates_expand)*self.vmap_t_0_eval*self.second_img_flow_0_t_eval+self.timerates_expand*self.vmap_t_1_eval*self.second_img_flow_1_t_eval),  \
                             ((1-self.timerates_expand)*self.vmap_t_0_eval+self.timerates_expand*self.vmap_t_1_eval) , name="second_outputimg_eval" )
        print ('second output img_eval:',self.second_output_eval)
        #second output img_eval: Tensor("second_outputimg_eval:0", shape=(10, 360, 640, 3), dtype=float32)
        
        #判别器的网络构建
        self.D_1_net_F, self.D_1_net_F_logit=Discriminator_net(self.second_output, name="D1", training=self.training)
        self.D_1_net_T, self.D_1_net_T_logit=Discriminator_net(self.frame1, name="D1", training=self.training)
        #D的loss计算
        self.D_1_net_loss_sum, _, _=self.D_loss_TandF_logits(self.D_1_net_T_logit, self.D_1_net_F_logit, "D_1_net")
        
        #计算loss
        self.second_L1_loss_interframe,self.first_warp_loss,self.second_contex_loss,self.second_local_var_loss_all,self.second_global_var_loss_all,self.second_ssim,self.second_psnr,\
                self.first_L1_loss_interframe, self.first_ssim, self.first_psnr, self.second_GAN_loss_mean_D1=self.loss_cal_all()
                
        #训练G的总loss
        self.G_loss_all=204 * self.second_L1_loss_interframe + 102 *  self.first_warp_loss  + 0.005 * self.second_contex_loss \
                    +self.second_global_var_loss_all
                    #+ self.second_GAN_loss_mean_D1*0.03   
        
        #训练D的总loss
        self.D_loss_all=self.D_1_net_loss_sum
        
        
        #####################################
        self.last_label_train='#'
        self.last_label_test='#'
        self.state_random_row_train=0
        self.state_random_col_train=0
        self.state_flip_train=False
        
        self.state_random_row_test=0
        self.state_random_col_test=0
        self.state_flip_test=False
        
        #为了兼容性
        self.batchsize_inputimg=batchsize
        self.img_size_w=img_size_w
        self.img_size_h=img_size_h
        
        t_vars=tf.trainable_variables()
        print ("trainable vars cnt:",len(t_vars))
        self.first_para=[var for var in t_vars if var.name.startswith('first')]
        self.sec_para=[var for var in t_vars if var.name.startswith('second')]
        self.vgg_para=[var for var in t_vars if var.name.startswith('VGG')]
        self.D_para=[var for var in t_vars if var.name.startswith('D')]
        print ("first param len:",len(self.first_para))
        print ("second param len:",len(self.sec_para))
        print ("VGG param len:",len(self.vgg_para))
        print ("D param len:",len(self.D_para))
        print (self.vgg_para)
        '''
        trainable vars cnt: 114
        first param len: 46
        second param len: 2
        VGG param len: 52
        D param len: 14
        '''
        
        #G训练过程
        self.lr_rate = tf.train.exponential_decay(base_lr,  global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate)
        self.train_op_G = tf.train.AdamOptimizer(self.lr_rate, beta1=beta1, name="superslomo_adam_G").minimize(self.G_loss_all,  \
                                                                                              global_step=self.global_step  , var_list=self.first_para+self.sec_para  )
        
        # weight clipping
        self.clip_D = [p.assign(tf.clip_by_value(p, weightclip_min, weightclip_max)) for p in self.D_para]
        
        #D训练过程
        self.train_op_D= tf.train.AdamOptimizer(self.lr_rate  , beta1=beta1, name="superslomo_adam_D").minimize(self.D_loss_all, var_list=self.D_para)
        
        #最后构建完成后初始化参数 
        self.sess.run(tf.global_variables_initializer())
        
    
    def D_loss_TandF_logits(self, logits_t, logits_f, summaryname='default'):
        self.D_loss_fir=-logits_t #tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_t, labels=tf.ones_like(logits_t))   #real
        
        self.D_loss_sec=logits_f #tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_f, labels=tf.zeros_like(logits_f))  #fake
        
       
        #testing target
        real_loss_mean=tf.reduce_mean(self.D_loss_fir)
        fake_loss_mean=tf.reduce_mean(self.D_loss_sec)
        tf.summary.scalar(summaryname+'_real_loss_mean',real_loss_mean)
        tf.summary.scalar(summaryname+'_fake_loss_mean',fake_loss_mean)
        
        loss_mean=real_loss_mean+fake_loss_mean
        
        tf.summary.scalar(summaryname+'_sum_loss_mean',loss_mean)        
        ############################################################
        
        return loss_mean,real_loss_mean,fake_loss_mean
        
    
    def G_loss_F_logits(self, logits, summaryname='default'):
        self.G_loss_fir=-logits #tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
        loss_mean = tf.reduce_mean(self.G_loss_fir)
        tf.summary.scalar(summaryname+'_loss_mean',loss_mean)
        
        return loss_mean    
    
        
    def loss_cal_all(self,name='superslomo'):           
        #1、conceptual loss
        print ("forming conce loss：")
        tep_G_shape=self.second_output.get_shape().as_list()[1:]
            
        with tf.variable_scope("VGG16",  reuse=tf.AUTO_REUSE):
            contex_Genera =tf.keras.applications.VGG16(include_top=False, input_tensor=self.second_output,  input_shape=tep_G_shape).get_layer("block4_conv3").output
            contex_frame1 =tf.keras.applications.VGG16(include_top=False, input_tensor=self.frame1, input_shape=tep_G_shape).get_layer("block4_conv3").output
            
            contex_loss=   tf.reduce_mean(tf.squared_difference( contex_frame1, contex_Genera), name='step2_Contex_loss')
            print ('loss_mean_conceptual form finished..')
        
        #2、L1 loss
        with tf.variable_scope(name,  reuse=tf.AUTO_REUSE):            
            print ("forming loss:  生成帧与GT")
            L1_loss_interframe =tf.reduce_mean(tf.abs(  self.second_output-self.frame1  ))
            
            warp_loss=tf.reduce_mean(tf.abs(  self.first_img_flow_2_t-self.frame1  ))+tf.reduce_mean(tf.abs(  self.first_img_flow_0_t-self.frame1  ))+\
                      tf.reduce_mean(tf.abs(  self.first_img_flow_2_0-self.frame0  ))+tf.reduce_mean(tf.abs(  self.first_img_flow_0_2-self.frame2  ))
            #这里顺便将第一步的中间结果输出，对比第二部结果能知道第二部加成效果
            first_L1_loss_interframe=tf.reduce_mean(tf.abs(  self.first_output-self.frame1  ))
            
            print ('mean_l1 loss form finished..')
            
        #4 local var loss
        with tf.variable_scope(name,  reuse=tf.AUTO_REUSE):
            local_var_loss_0_2=self.local_var_loss(self.first_opticalflow_0_1)
            local_var_loss_2_0=self.local_var_loss(self.first_opticalflow_1_0)
            #print ("local _var loss:",self.local_var_loss_0_2,  self.G_loss_mean_D1)
            #local _var loss: Tensor("mean_local_var:0", shape=(), dtype=float32) Tensor("Mean_3:0", shape=(), dtype=float32)
            local_var_loss_all=tf.add(local_var_loss_0_2, local_var_loss_2_0, name="step2_local_var_add")
            
        #5 global var loss
        with tf.variable_scope(name,  reuse=tf.AUTO_REUSE):
            global_var_loss_0_2=self.global_var_loss(self.first_opticalflow_0_1)
            global_var_loss_2_0=self.global_var_loss(self.first_opticalflow_1_0)
            global_var_loss_all=tf.add(global_var_loss_0_2, global_var_loss_2_0, name="step2_global_var_add")
            
        #6 SSIM
        with tf.variable_scope(name,  reuse=tf.AUTO_REUSE):
            ssim = tf.image.ssim(self.second_output, self.frame1, max_val=2.0)
            print ("ssim:",ssim)  #ssim: Tensor("Mean_10:0", shape=(12,), dtype=float32)
            
            #这里顺便将第一步的中间结果输出，对比第二部结果能知道第二部加成效果
            first_ssim=tf.image.ssim(self.first_output, self.frame1, max_val=2.0)
            print ("first ssim:",first_ssim) #
            
        #7 PSNR
        with tf.variable_scope(name,  reuse=tf.AUTO_REUSE):
            psnr = tf.image.psnr(self.second_output, self.frame1, max_val=2.0, name="step2_frame1_psnr")
            print ("psnr:", psnr) #psnr: Tensor("G_frame1_psnr/Identity_3:0", shape=(12,), dtype=float32)
            
            #这里顺便将第一步的中间结果输出，对比第二部结果能知道第二部加成效果
            first_psnr = tf.image.psnr(self.first_output, self.frame1, max_val=2.0, name="step1_frame1_psnr")
            print ("first_psnr:", first_psnr) #
            
        #8 GAN loss
        with tf.variable_scope(name,  reuse=tf.AUTO_REUSE):
            G_loss_mean_D1=self.G_loss_F_logits(self.D_1_net_F_logit, 'step2_frame1_GAN')
            print ("G_loss_mean_D1:",G_loss_mean_D1)
            
        return L1_loss_interframe,warp_loss,contex_loss,local_var_loss_all,global_var_loss_all,ssim,psnr,first_L1_loss_interframe, first_ssim, first_psnr, G_loss_mean_D1

    def warp_op(self, images, flow, timerates=1):
        '''
        tf.contrib.image.dense_image_warp(
            image,
            flow,
            name='dense_image_warp'
        )
        pixel value at output[b, j, i, c] is
          images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].
        '''
        return tf.contrib.image.dense_image_warp(images, flow*timerates)

    def local_var_loss(self, flow, kernel_size=10, stride=1):
        '''
        计算局部平滑loss，即每一个卷积范围内的方差之和
        flow:nhwc,channel=2
        '''
        flow_shape=flow.get_shape().as_list()
        #[filter_height, filter_width, in_channels, channel_multiplier]
        common_kernel=tf.ones([kernel_size, kernel_size, flow_shape[-1], 1])
        flow_squ=tf.square(flow)
        #E xi^2
        E_flow_squ=tf.nn.depthwise_conv2d(flow_squ, common_kernel, strides=[1,stride,stride,1], padding="VALID")/(kernel_size*kernel_size)
        
        #(E x)^2
        E_flow    =tf.nn.depthwise_conv2d(flow,     common_kernel, strides=[1,stride,stride,1], padding="VALID")/(kernel_size*kernel_size)
        E_flow=tf.square(E_flow)
        
        local_var=tf.subtract(E_flow_squ, E_flow, name="step2_local_var")
        #local_var: Tensor("local_var:0", shape=(12, 171, 311, 2), dtype=float32)
        print ("local_var:",local_var)
        
        mean_local_var=tf.reduce_mean(local_var, name="step2_mean_local_var")
        
        return mean_local_var
    
    def global_var_loss(self, flow):
        tep=tf.reduce_mean( tf.abs(flow[:, :-1, :, :]-flow[:, 1:, :, :]) )+tf.reduce_mean( tf.abs(flow[:, :, :-1, :]-flow[:, :, 1:, :]) )
        return tep

    def getbatch_train_imgs(self):
        newstate=False
        while True:
            tepimg=self.sess.run(self.pipline_data_train)
            inimg,rate,label=tepimg[0],tepimg[1],tepimg[2]  #注意这里拿到的img是size+6大小的，这里可以进行一个数据扩张，但是扩展一定要连续帧一致
            if str(label[0]).split('_')[0]==str(label[-1]).split('_')[0]: break
            newstate=True
        if str(label[0]).split('_')[0]!=self.last_label_train: #如果刚好上面处于两个视频分界线，这里可能存在误判，这里进行修复
            newstate=True
        self.last_label_train=str(label[0]).split('_')[0]
        
        if newstate:
            self.state_random_row_train=np.random.randint(6)
            self.state_random_col_train=np.random.randint(6)
            self.state_flip_train=np.random.randint(2)
        
        inimg=inimg[:,self.state_random_row_train:self.state_random_row_train+self.img_size_h, self.state_random_col_train:self.state_random_col_train+self.img_size_w]
        if self.state_flip_train:  #左右翻转扩张
            inimg=np.flip(inimg,2)
        
        return self.img2tanh(inimg),rate,newstate
    
    def getbatch_test_imgs(self):
        newstate=False
        while True:
            tepimg=self.sess.run(self.pipline_data_test)
            inimg,rate,label=tepimg[0],tepimg[1],tepimg[2]
            if str(label[0]).split('_')[0]==str(label[-1]).split('_')[0]: break
            newstate=True
        if str(label[0]).split('_')[0]!=self.last_label_test:
            newstate=True
        self.last_label_test=str(label[0]).split('_')[0]
        
        if newstate:
            self.state_random_row_test=np.random.randint(6)
            self.state_random_col_test=np.random.randint(6)
            self.state_flip_test=np.random.randint(2)
        ''' 在测试阶段，不需要数据扩张
        inimg=inimg[:,self.state_random_row_test:self.state_random_row_test+self.img_size_h, self.state_random_col_test:self.state_random_col_test+self.img_size_w]
        if self.state_flip_test:  #INIMG[12，180，320，9]
            inimg=np.flip(inimg, 2)
        '''
        
        return self.img2tanh(inimg),rate,newstate
    
    def img2tanh(self,img):
        #img=tf.cast(img,tf.float32)
        #img-=mean_dataset*3
        return img*1.0/255
    
    def tanh2img(self,tanhd):
        tep= (tanhd)*255
        #print ('tep.shape:',tep.shape)  #tep.shape: (180, 320, 9)
        multly=int(tep.shape[-1]/len(mean_dataset))
        #print ('expanding:',multly)
        #tep+=mean_dataset*multly
        return tep.astype(np.uint8)

    def train_once(self):
        imgdata,rate,newstate=self.getbatch_train_imgs()
        if newstate: 
            self.last_flow_new_train=self.last_flow_init_np
            print ('start from a zero flow!!!')
        
        _,_,_, self.last_flow_new_train,\
        sec_ssim,sec_psnr,      \
        sec_contexloss, sec_L1_loss,first_loss_warp, \
        sec_localloss,sec_globalloss,sec_GAN_loss, loss_all,D_loss_all,\
        first_L1_loss, first_ssim, first_psnr=self.sess.run([self.train_op_D, self.clip_D, self.train_op_G, self.second_batch_last_flow,\
                                    self.second_ssim, self.second_psnr, \
                                    self.second_contex_loss,  self.second_L1_loss_interframe,self.first_warp_loss, \
                                    self.second_local_var_loss_all,self.second_global_var_loss_all,  self.second_GAN_loss_mean_D1, self.G_loss_all, self.D_loss_all,\
                                    self.first_L1_loss_interframe, self.first_ssim, self.first_psnr], \
                      feed_dict={self.imgs_pla:imgdata,self.timerates_pla:rate, self.last_optical_flow:self.last_flow_new_train, self.training:True})
        
        #print ()
        print ("train once:")
        print ("first ssim:",first_ssim)
        print ("second ssim:",sec_ssim)
        
        print ("first psnr:",first_psnr)
        print ("second psnr:",sec_psnr)
        
        print ("second contexloss:",sec_contexloss)
        #print ("contexloss2:",contexloss2)
        print ("second local var loss:",sec_localloss)
        print ("second global var loss:",sec_globalloss)
        print ("second GAN loss:",sec_GAN_loss)
        
        print ("first  l1_loss_all:",first_L1_loss)
        print ("second l1_loss_all:",sec_L1_loss)
        
        print ("first l1_loss_warp:",first_loss_warp)
        print ("G loss_all :",loss_all)
        print ("D loss_all:",D_loss_all)
        
        return loss_all

    def eval_once(self, step,evalstep=100):
        kep_img_dir=op.join(kepimgdir, str(step))
        os.makedirs(kep_img_dir, exist_ok=True)
        
        recording=0  #遇到一个新的状态开始记录，到下一个状态停止
        kep_ssim1=0.0
        kep_psnr1=0.0
        kep_l1loss1=0.0
        kep_l1loss2=0.0
        kep_G_GAN_loss1=0.0
        kep_localloss2=0
        kep_ssim2=0.0
        kep_psnr2=0.0
        kep_loss_warp1=0.0
        kep_contexloss2=0.0
        kep_globalloss2=0
        
        img_cnt=0
        
        for i in range(evalstep):
            imgdata,rate,newstate=self.getbatch_test_imgs()
            if newstate: 
                self.last_flow_new_test=self.last_flow_init_np
                recording+=1
                
            self.last_flow_new_test,\
            step1_flow_0_1,step1_flow_1_0,step1_imgout,step2_flow_t_0,step2_flow_t_1,step2_outimg   ,\
            ssim1,ssim2,psnr1, psnr2,     \
            contexloss2, L1_loss1, L1_loss2,L1_loss_warp1, \
            localloss2,globalloss2,G_GAN_loss1, loss_all=self.sess.run([self.second_batch_last_flow,\
                                    self.first_opticalflow_0_1,self.first_opticalflow_1_0,self.first_output ,self.second_opticalflow_t_0,self.second_opticalflow_t_1, self.second_output              ,\
                                    self.first_ssim,self.second_ssim, self.first_psnr, self.second_psnr,\
                                    self.second_contex_loss,  self.first_L1_loss_interframe, self.second_L1_loss_interframe,self.first_warp_loss, \
                                    self.second_local_var_loss_all,self.second_global_var_loss_all,  self.second_GAN_loss_mean_D1, self.G_loss_all,\
                                    ], \
                      feed_dict={self.imgs_pla:imgdata,self.timerates_pla:rate, self.last_optical_flow:self.last_flow_new_test, self.training:False})
            kep_G_GAN_loss1+=G_GAN_loss1
            kep_ssim1+=np.mean(ssim1)
            kep_ssim2+=np.mean(ssim2)
            kep_psnr1+=np.mean(psnr1)
            kep_psnr2+=np.mean(psnr2)
            kep_l1loss1+=np.mean(L1_loss1)
            kep_l1loss2+=np.mean(L1_loss2)
            kep_loss_warp1+=np.mean(L1_loss_warp1)
            kep_contexloss2+=np.mean(contexloss2)
            #kep_contexloss2+=np.mean(contexloss2)
            kep_localloss2+=localloss2
            kep_globalloss2+=globalloss2
            
            if recording in [1,2]:
                tep=self.form_bigimg(imgdata,step1_flow_0_1,step1_flow_1_0,step1_imgout,step2_flow_t_0,step2_flow_t_1,step2_outimg)
                cv2.imwrite( op.join(kep_img_dir, "recording_"+str(recording)+"_"+str(img_cnt)+'.jpg') ,  tep)
                img_cnt+=1
                
                    
            
        print ("eval ",evalstep,' times:')
        print ('mean ssim1:',kep_ssim1/evalstep)
        print ('mean ssim2:',kep_ssim2/evalstep)
        print ('mean psnr1:',kep_psnr1/evalstep)
        print ('mean psnr2:',kep_psnr2/evalstep)
        print ('mean l1loss of interframe1:',kep_l1loss1/evalstep)
        print ('mean l1loss of interframe2:',kep_l1loss2/evalstep)
        
        print ('mean l1loss_warp1:',kep_loss_warp1/evalstep)
        
        print ('mean contexloss2:', kep_contexloss2/evalstep)
        print ('mean localvar loss2:',kep_localloss2/evalstep)
        print ("mean global loss2:",kep_globalloss2/evalstep)
        print ("mean G_GAN loss2:",kep_G_GAN_loss1/evalstep)
        
        print ("write "+str(img_cnt)+" imgs to:"+kep_img_dir)
        return [kep_ssim1/evalstep,kep_ssim2/evalstep], [kep_psnr1/evalstep,kep_psnr2/evalstep], [kep_l1loss1/evalstep,kep_l1loss2/evalstep], \
            kep_loss_warp1/evalstep, kep_contexloss2/evalstep,  kep_localloss2/evalstep, kep_globalloss2/evalstep, kep_G_GAN_loss1/evalstep
    
    def form_bigimg(self, imgdata,step1_flow_0_1,step1_flow_1_0,step1_imgout,step2_flow_0_1,step2_flow_1_0,step2_outimg):
        #imgdata:[12,180,320,9]
        #根据输入的原始输入3图，第一步的输出前后光流，第一步的输出合成中间帧，第二部的前后光流，第二部的合成中间帧--->合成一张大图
        gap=6
        height=self.batchsize_inputimg*self.img_size_h+ (self.batchsize_inputimg-1)*gap
        width =self.img_size_w*9+(9-1)*gap
        
        ret=np.zeros([height, width, img_channel], dtype=np.uint8)
        imgdata=self.tanh2img(imgdata)
        step1_imgout=self.tanh2img(step1_imgout)
        step2_outimg=self.tanh2img(step2_outimg)
        
        for j in range(self.batchsize_inputimg):
            col=0
            row=j*(self.img_size_h+gap)
            #0,1,2
            for k in range(3): 
                ret[row:row+self.img_size_h, col:col+self.img_size_w]=imgdata[j, :,:,k*img_channel:(k+1)*img_channel]
                col+=self.img_size_w+gap
            #3
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step1_flow_0_1[j])
            col+=self.img_size_w+gap
            #4
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step1_flow_1_0[j])
            col+=self.img_size_w+gap
            #5
            mean_l1=np.mean( np.abs(  self.img2tanh(step1_imgout[j])  -   self.img2tanh(imgdata[j, :,:,1*img_channel:(1+1)*img_channel] ) ) )
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=cv2.putText(step1_imgout[j],'GT_step1_L1loss:'+str(mean_l1),(0,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            col+=self.img_size_w+gap
            #6
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step2_flow_0_1[j])
            col+=self.img_size_w+gap
            #7
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step2_flow_1_0[j])
            col+=self.img_size_w+gap
            #8
            mean_l1=np.mean( np.abs(  self.img2tanh(step2_outimg[j]) - self.img2tanh(imgdata[j, :,:,1*img_channel:(1+1)*img_channel]) ) )
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=cv2.putText(step2_outimg[j],'GT_step2_L1loss:'+str(mean_l1),(0,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            col+=self.img_size_w+gap
        return ret
    
    
    def flow_bgr(self, flow):
        #flow:[h,w,2] no batch
        # Use Hue, Saturation, Value colour model 
        #色调（H），饱和度（S），明度（V）
        #这里色调<->方向
        #饱和度 =255
        #明度<->运动长度
        #黑色代表无运动
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), np.uint8)
        #print (hsv.shape)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        '''
        cv2.imshow("colored flow", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return bgr

if __name__ == '__main__':
    with tf.Session(config=config) as sess:     
        gan = SuperSlomo(sess)
        
        logwriter = tf.summary.FileWriter(logdir,   sess.graph)
        
        # 注意这里制定了保存的参数，避免包含进VGG的参数
        all_saver = tf.train.Saver(var_list=gan.first_para+gan.sec_para,  max_to_keep=2) 


        begin_t=time.time()
        for i in range(maxstep):     
            print ('\n',"{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
                   
            if i==0 or (i+1)%2000==0:#一次测试
                print ('begining to eval:')
                ssim_mean, psnr_mean,interframeloss,warploss, contexloss,localvarloss,globalloss, G_GAN_loss=gan.eval_once(i)
                
                #自己构建summary
                tsummary = tf.Summary()
                #tsummary.value.add(tag='mean prob of real1', simple_value=prob_T)
                #tsummary.value.add(tag='mean prob of fake1', simple_value=prob_F)
                tsummary.value.add(tag='first mean L1_loss of G and GT', simple_value=interframeloss[0])     
                tsummary.value.add(tag='second mean L1_loss of G and GT', simple_value=interframeloss[1]) 
                tsummary.value.add(tag='first mean L1_loss of warps', simple_value=warploss)           
                tsummary.value.add(tag='second mean contexloss', simple_value=contexloss)
                tsummary.value.add(tag='second G GAN loss', simple_value=G_GAN_loss)
                tsummary.value.add(tag='second mean localvar loss', simple_value=localvarloss)
                tsummary.value.add(tag='first mean ssim', simple_value=ssim_mean[0])
                tsummary.value.add(tag='second mean ssim', simple_value=ssim_mean[1])
                tsummary.value.add(tag='first mean psnr', simple_value=psnr_mean[0])
                tsummary.value.add(tag='second mean psnr', simple_value=psnr_mean[1])
                '''
                tsummary.value.add(tag='mean L1_loss of G and GT step2', simple_value=l1loss[1])
                tsummary.value.add(tag='mean contexloss step2', simple_value=contexloss[1])
                tsummary.value.add(tag='mean localvar loss step2', simple_value=localvarloss[1])
                tsummary.value.add(tag='mean ssim step2', simple_value=ssim_mean[1])
                tsummary.value.add(tag='mean psnr step2', simple_value=psnr_mean[1])
                '''
                #写入日志
                logwriter.add_summary(tsummary, i)
                
            if i==0 or (i+1)%1500==0:#保存一波图片
                pass#gan.eval_G_once(i)
                
                
            if (i+1)%2000==0:#保存模型
                print ('saving models...')
                pat=all_saver.save(sess, op.join(logdir,'model_keep'),global_step=i)
                print ('saved at:',pat)
            
            
            stt=time.time()
            print ('%d/%d  start train_once...'%(i,maxstep))
            #lost,sum_log=vgg.train_once(sess) #这里每次训练都run一个summary出来
            gan.train_once()
            #写入日志
            #logwriter.add_summary(sum_log, i)
            #print ('write summary done!')
            
                
            print ('time used:',time.time()-stt,' to be ',1.0/(time.time()-stt),' iters/s', ' left time:',(time.time()-stt)*(maxstep-i)/60/60,' hours')
            
        
        print ('Training done!!!-->time used:',(time.time()-begin_t),'s = ',(time.time()-begin_t)/60/60,' hours')
















