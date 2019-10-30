#coding:utf-8
'''
Created on 2018年12月10日

@author: sherl
'''
import cv2,os,random,time
from datetime import datetime
import os.path as op
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from GAN_tools_v25 import *
from data import create_dataset_step1 as cdata
#from data import create_dataset2 as cdata
import skimage

#this version change output of g to be img
#and chagne img-size to v2's 1/2
#use wgan loss function
#my_novel_conv
#use l2 loss for img clear
#use global rate to mult square loss
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print ('tensorflow version:',tf.__version__,'  path:',tf.__path__)
TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

train_size=112064 
test_size=8508
batchsize=6  #train
batchsize_test=batchsize #here it must equal to batchsize,or the placement size will error

#
img_size_w=int (640/2)
img_size_h=int (360/2)
img_size=[img_size_h, img_size_w]

base_lr=0.0002 #基础学习率
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

logdir="./logs_v25/GAN_"+TIMESTAMP+('_base_lr-%f_batchsize-%d_maxstep-%d'%(base_lr,batchsize, maxstep))

bigimgsdir=op.join(logdir, 'randomimgs')
if not op.exists(bigimgsdir): os.makedirs(bigimgsdir)

#设置GPU显存按需增长
gpu_options = tf.GPUOptions(allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



class GAN_Net:
    def __init__(self, sess):
        self.sess=sess
        self.global_step = tf.Variable(0.0, name='global_step',dtype=tf.float32, trainable=False)
        
        self.G_para=[]
        self.D_para=[]       
        
        #for debug
        self.cnt_tep=0
        self.deb_kep=0
        self.deb_kep2=0
        
        #for data input
        self.pipline_data_train=cdata.get_pipline_data_train(img_size, batchsize)
        self.pipline_data_test=cdata.get_pipline_data_test(img_size, batchsize_test)
        
        #3个placeholder， img和noise,training 
        self.imgs_pla = tf.placeholder(tf.float32, [batchsize, img_size_h, img_size_w, G_group_img_num*img_channel], name='imgs_in')
        self.training=tf.placeholder(tf.bool, name='training_in')
        self.timerates_pla=tf.placeholder(tf.float32, [batchsize], name='timerates_in')
        self.timerates_expand=tf.expand_dims(self.timerates_pla, -1)
        self.timerates_expand=tf.expand_dims(self.timerates_expand, -1)
        self.timerates_expand=tf.expand_dims(self.timerates_expand, -1) #12*1*1*1
        
        print ('placeholders:\n','img_placeholder:',self.imgs_pla,'\ntraining:',self.training)
        '''
        placeholders:
        img_placeholder: Tensor("imgs_in:0", shape=(12, 180, 320, 9), dtype=float32) 
        training: Tensor("training_in:0", dtype=bool)
        '''
        
        self.frame0=self.imgs_pla[:,:,:,:img_channel]
        self.frame1=self.imgs_pla[:,:,:,img_channel:img_channel*2]
        self.frame2=self.imgs_pla[:,:,:,img_channel*2:]
        
        #这里是为了看看第一帧和第3帧的差距，用来给evalte
        self.frame_0_2_L1loss=tf.reduce_mean(  tf.abs(self.frame0-self.frame2) , [1,2,3], name='frame_0_2_L1loss')
        print ('frame_0_2_L1loss:',self.frame_0_2_L1loss)
        #frame0and2=tf.concat([self.frame0, self.frame2], -1) #在第三维度连接起来
        #print ('after concat:',frame0and2)
        #!!!!!!!!!!here is differs from v1,add to Generator output the ori img will reduce the generator difficulty 
        self.G_opticalflow=self.Generator_net(self.frame0, self.frame2)  #注意这里是直接作为optical flow
        ###################3testing
        #到这里trainable var 只有60个，是G的参数
        
        #下面将结果中的代表各个意义分开
        #optical flow[:,:,:,0:2] is frame0->frame2(get frame2 from frame0), [2:]is 2->0
        self.opticalflow_0_2=tf.slice(self.G_opticalflow, [0, 0, 0, 0], [-1, -1, -1, 2], name='G_opticalflow_0_2') #self.G_opticalflow[:,:,:,:2]
        
        #self.prob_flow1=tf.clip_by_value(self.G_opticalflow[:,:,:,2],0,1 , name='prob_flow1_sigmoid')
        #self.prob_flow1=  tf.nn.sigmoid( self.G_opticalflow[:,:,:,2] , name='prob_flow1_sigmoid')  #这里添加了一个置信度，用来选择光流,即相信F0->1还是相信F1->0
        
        self.opticalflow_2_0=tf.slice(self.G_opticalflow, [0, 0, 0, 2], [-1, -1, -1, 2], name='G_opticalflow_2_0') #       self.G_opticalflow[:,:,:,3:]
        print ('original flow:',self.opticalflow_0_2, self.opticalflow_2_0)
        #original flow: Tensor("G_opticalflow_0_2:0", shape=(12, 180, 320, 2), dtype=float32) 
        #Tensor("prob_flow1_sigmoid:0", shape=(12, 180, 320), dtype=float32) 
        #Tensor("G_opticalflow_2_0:0", shape=(12, 180, 320, 2), dtype=float32)
        
        #反向光流算中间帧
        self.opticalflow_t_0=tf.add( -(1-self.timerates_expand)*self.timerates_expand*self.opticalflow_0_2 ,\
                                      self.timerates_expand*self.timerates_expand*self.opticalflow_2_0 , name="G_opticalflow_t_0")
        self.opticalflow_t_2=tf.add( (1-self.timerates_expand)*(1-self.timerates_expand)*self.opticalflow_0_2 ,\
                                      self.timerates_expand*(self.timerates_expand-1)*self.opticalflow_2_0, name="G_opticalflow_t_2")
        
        print ('two optical flow:',self.opticalflow_t_0, self.opticalflow_t_2) 
        #two optical flow: Tensor("G_opticalflow_t_0:0", shape=(12, 180, 320, 2), dtype=float32) 
        #Tensor("G_opticalflow_t_2:0", shape=(12, 180, 320, 2), dtype=float32),
        
        #2种方法合成t时刻的帧
        self.img_flow_2_t=self.warp_op(self.frame2, -self.opticalflow_t_2) #!!!
        self.img_flow_0_t=self.warp_op(self.frame0, -self.opticalflow_t_0) #!!!
        
        self.G_net=tf.add(self.timerates_expand*self.img_flow_2_t , (1-self.timerates_expand)*self.img_flow_0_t, name="G_net_generate" )
        print ('self.G_net:',self.G_net)
        '''
        tep_prob_flow1=tf.expand_dims(self.prob_flow1, -1)
        tep_prob_flow1=tf.tile(tep_prob_flow1, [1,1,1,3])
        #self.G_net=tf.where( tf.greater_equal(tep_prob_flow1, 0.5),  self.img_flow_0_t, self.img_flow_2_t, name='G_net_generate') #这里认为>0.5就是相信frame0
        tep_sujm=tep_prob_flow1*(1-self.timerates_expand)+(1-tep_prob_flow1)*self.timerates_expand
        self.G_net=tf.add(self.img_flow_0_t*tep_prob_flow1*(1-self.timerates_expand)/tep_sujm, \
                          self.img_flow_2_t*(1-tep_prob_flow1)*self.timerates_expand/tep_sujm,  name='G_net_generate')
        '''
        print ('self.G_net:',self.G_net)#self.G_net: Tensor("G_net_generate:0", shape=(12, 180, 320, 3), dtype=float32)
        
        
        #利用光流前后帧互相合成
        self.img_flow_2_0=self.warp_op(self.frame2, self.opticalflow_2_0)  #frame2->frame0
        self.img_flow_0_2=self.warp_op(self.frame0, self.opticalflow_0_2)  #frame0->frame2
        
    
        #D_1的输出 
        frame0_False_2=tf.concat([self.frame0, self.G_net,self.frame2], -1)
        #self.D_linear_net_F, self.D_linear_net_F_logit=self.Discriminator_net_linear(frame0_False_2)
        #self.D_linear_net_T, self.D_linear_net_T_logit=self.Discriminator_net_linear(self.imgs_pla)
        #下面是loss公式
        #self.D_linear_net_loss_sum, self.D_linear_net_loss_T, self.D_linear_net_loss_F=self.D_loss_TandF_logits(self.D_linear_net_T_logit, self.D_linear_net_F_logit, "D_linear_net")
        print ('D1 form finished..')
        #D_2的输出
        '''
        self.D_clear_net_F, self.D_clear_net_F_logit=self.Discriminator_net_clear(self.G_net)
        self.D_clear_net_T, self.D_clear_net_T_logit=self.Discriminator_net_clear(self.frame1)
        #下面是loss公式
        self.D_clear_net_loss_sum, self.D_clear_net_loss_T, self.D_clear_net_loss_F=self.D_loss_TandF_logits(self.D_clear_net_T_logit, \
                                                                                                                self.D_clear_net_F_logit, "D_clear_net")
        
        
        '''
        #self.G_loss_mean_Square=tf.reduce_mean(tf.squared_difference(self.G_net,self.frame1), name='G_clear_square_loss')
        #1、contex loss
        print ("forming conx loss：")
        tep_G_shape=self.G_net.get_shape().as_list()[1:]
        
        with tf.variable_scope("VGG16",  reuse=tf.AUTO_REUSE):
            self.contex_Genera =tf.keras.applications.VGG16(include_top=False, input_tensor=self.G_net,  input_shape=tep_G_shape).get_layer("block4_conv3").output
            self.contex_frame1 =tf.keras.applications.VGG16(include_top=False, input_tensor=self.frame1, input_shape=tep_G_shape).get_layer("block4_conv3").output
        
        self.contex_loss=   tf.reduce_mean(tf.squared_difference( self.contex_frame1, self.contex_Genera), name='G_Contex_loss')
        print ('G_loss_mean_contex form finished..')
        
        #2、L1 loss
        print ("forming L1 loss:生成帧与GT、frame2->frame0与frame0、frame0->frame2与frame2")
        self.L1_loss_interframe =tf.reduce_mean(tf.abs(  self.G_net-self.frame1  ))
        self.L1_loss_all        =tf.reduce_mean(tf.abs(  self.G_net-self.frame1  ) + \
                                                tf.abs(self.img_flow_2_0-self.frame0) + \
                                                tf.abs(self.img_flow_0_2-self.frame2), name='G_clear_l1_loss')
        #self.G_loss_mean_Square=  self.contex_loss*1 + self.L1_loss_all
        print ('G_loss_mean_l1 form finished..')
        
        #3、下面是G的loss
        #self.G_loss_mean_D1=self.G_loss_F_logits(self.D_linear_net_F_logit, 'G_loss_D1')
        #self.G_loss_mean_D2=self.G_loss_F_logits(self.D_clear_net_F_logit, 'G_loss_D2')
        
        #4 local var loss
        self.local_var_loss_0_2=self.local_var_loss(self.opticalflow_0_2)
        self.local_var_loss_2_0=self.local_var_loss(self.opticalflow_2_0)
        #print ("local _var loss:",self.local_var_loss_0_2,  self.G_loss_mean_D1)
        #local _var loss: Tensor("mean_local_var:0", shape=(), dtype=float32) Tensor("Mean_3:0", shape=(), dtype=float32)
        self.local_var_loss_all=tf.add(self.local_var_loss_0_2, self.local_var_loss_2_0, name="local_var_add")
        
        #5 global var loss
        self.global_var_loss_0_2=self.global_var_loss(self.opticalflow_0_2)
        self.global_var_loss_2_0=self.global_var_loss(self.opticalflow_2_0)
        self.global_var_loss_all=tf.add(self.global_var_loss_0_2, self.global_var_loss_2_0, name="global_var_add")
        
        #6 SSIM
        self.ssim = tf.image.ssim(self.G_net, self.frame1, max_val=2.0)
        print ("ssim:",self.ssim)  #ssim: Tensor("Mean_10:0", shape=(12,), dtype=float32)
        
        #7 PSNR
        self.psnr = tf.image.psnr(self.G_net, self.frame1, max_val=2.0, name="G_frame1_psnr")
        print ("psnr:", self.psnr) #psnr: Tensor("G_frame1_psnr/Identity_3:0", shape=(12,), dtype=float32)
        
        #训练生成器的总LOSS   这里将G的loss和contex loss与前面G的loss做一个归一化，这样当D的loss大的时候，说明这时D不可靠，需要多训练D，而相应的减小该D对G的训练影响        
        self.G_loss_all=self.contex_loss*5 + \
                        self.L1_loss_all*10 +\
                        self.global_var_loss_all *0.1
                        #self.local_var_loss_all *0.06
                        #self.G_loss_mean_D1 + \
                        
                        
                        #* (1+self.global_step/G_squareloss_rate_globalstep)# self.G_loss_mean_D2     
                        #W ./tensorflow/core/grappler/optimizers/graph_optimizer_stage.h:241] Failed to run optimizer ArithmeticOptimizer, stage HoistCommonFactor node 
                        #add_8. Error: Node ArithmeticOptimizer/HoistCommonFactor_Add_add_7 is missing output properties at position :0 (num_outputs=0)
                         
        
        #训练判别器D的loss    这里将d的loss与前面G的loss和contex loss做一个归一化，这样当这里D的loss大的时候，说明这时D不可靠，需要多训练D，而相应的减小该D对G的训练影响
        #self.D_loss_all=self.D_linear_net_loss_sum  #+ self.D_clear_net_loss_sum
        
        #还是应该以tf.trainable_variables()为主
        t_vars=tf.trainable_variables()
        print ("trainable vars cnt:",len(t_vars))
        self.G_para=[var for var in t_vars if var.name.startswith('G')]
        self.D_para=[var for var in t_vars if var.name.startswith('D')]
        print (t_vars)
        '''
        trainable vars cnt: 128
        D: AdamOptimizer to maxmize 16 vars..
        G: AdamOptimizer to maxmize 60 vars..
        :这里有个差值128-16-60，差得是VGG网络中的参数52，还是很吃内存的
        '''
        
        # weight clipping
        self.clip_D = [p.assign(tf.clip_by_value(p, weightclip_min, weightclip_max)) for p in self.D_para]
        
        #训练使用
        #self.train_D=self.train_op_D(decay_steps, decay_rate)
        self.train_G=self.train_op_G(decay_steps, decay_rate)  
        #print ("training op names:",self.train_D, self.train_G)
        
        '''
        print ('\nshow all trainable vars:',len(tf.trainable_variables()))
        for i in tf.trainable_variables():
            print (i)
        '''
        print ('\nfirst show G params')
        for ind,i in enumerate(self.G_para): print (ind,i)
        
        print('\nnext is D:\n')
        for ind,i in enumerate(self.D_para): print (ind,i)
        
        print ('\nnext is tf.GraphKeys.UPDATE_OPS:')
        print (tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        
        self.summary_all=tf.summary.merge_all()
        
        init = tf.global_variables_initializer()#初始化tf.Variable
        self.sess.run(init)
    
    
    
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
    
    
    def local_var_loss(self, flow, kernel_size=5, stride=4):
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
        
        local_var=tf.subtract(E_flow_squ, E_flow, name="local_var")
        #local_var: Tensor("local_var:0", shape=(12, 171, 311, 2), dtype=float32)
        print ("local_var:",local_var)
        
        mean_local_var=tf.reduce_mean(local_var, name="mean_local_var")
        
        return mean_local_var
    
    def global_var_loss(self, flow):
        tep=tf.reduce_mean( tf.abs(flow[:, :-1, :, :]-flow[:, 1:, :, :]) )+tf.reduce_mean( tf.abs(flow[:, :, :-1, :]-flow[:, :, 1:, :]) )
        return tep
        
    def img2tanh(self,img):
        #img=tf.cast(img,tf.float32)
        #img-=mean_dataset*3
        return img*2.0/255-1
    
    def tanh2img(self,tanhd):
        tep= (tanhd+1)*255//2
        #print ('tep.shape:',tep.shape)  #tep.shape: (180, 320, 9)
        multly=int(tep.shape[-1]/len(mean_dataset))
        #print ('expanding:',multly)
        #tep+=mean_dataset*multly
        return tep.astype(np.uint8)  
    
    def getbatch_train_imgs(self):
        tepimg=self.sess.run(self.pipline_data_train)
        inimg,rate=tepimg[0],tepimg[1]
        return self.img2tanh(inimg),rate
    
    def getbatch_test_imgs(self):
        tepimg=self.sess.run(self.pipline_data_test)
        inimg,rate=tepimg[0],tepimg[1]
        return self.img2tanh(inimg),rate
    
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
    
    
    def train_op_G(self,decay_steps=8000, decay_rate=0.99, beta1=beta1):
        #self.lr_rate = base_lr
        self.lr_rate = tf.train.exponential_decay(base_lr,  global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate)
        
        print ('G: AdamOptimizer to maxmize %d vars..'%(len(self.G_para)))
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.G_optimizer=tf.train.AdamOptimizer(self.lr_rate  , beta1=beta1)
            
            #for i in optimizer.compute_gradients(self.G_loss_mean, var_list=self.G_para): print (i)
            
            train_op =self.G_optimizer.minimize(self.G_loss_all, global_step=self.global_step,var_list=self.G_para)
        
        return train_op
    
    def train_op_D(self,decay_steps=8000, decay_rate=0.99, beta1=beta1):
        #self.lr_rate = base_lr
        self.lr_rate = tf.train.exponential_decay(base_lr,  global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate)
        
        print ('D: AdamOptimizer to maxmize %d vars..'%(len(self.D_para)))
        
        #这里就不管globalstep了，否则一次迭代会加2次
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_optimizer= tf.train.AdamOptimizer(self.lr_rate  , beta1=beta1)
            train_op=self.D_optimizer.minimize(self.D_loss_all, var_list=self.D_para)   #global_step=self.global_step,
        return train_op
                                                                                                                                                       
    
    
    
    #tensor 范围外   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def flow_bgr(self, flow):
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
    
    def train_once_all(self):
        tepimgs,time_rates=self.getbatch_train_imgs()
        
        lrrate,\
        _ ,  G_loss_L1, G_loss_contex,G_loss_localvar, G_loss_globalvar, G_loss_sum_all, \
        ssim, psnr =\
                                       self.sess.run([self.lr_rate,\
                                                      #self.D_linear_net_T, self.D_linear_net_F,\
                                                      #self.D_clear_net_T, self.D_clear_net_F,  \
                                                      #self.D_linear_net_loss_sum, self.D_loss_all,\
                                                      self.train_G,  self.L1_loss_all, self.contex_loss,self.local_var_loss_all,self.global_var_loss_all, self.G_loss_all,\
                                                      self.ssim,self.psnr]          , \
                                                    feed_dict={  self.imgs_pla:tepimgs ,  self.timerates_pla:time_rates ,self.training:True})
        print ('trained once:')
        print ('lr:',lrrate)
        #print ('D1(D_linear) prob T/F --> ',np.mean(D1_T_prob),'/',np.mean( D1_F_prob))
        #print ('D1 loss_all:',D1_loss)
        #print ('D2(D_clear) prob T/F --> ',np.mean(D2_T_prob),'/', np.mean(D2_F_prob))
        #print ('D2 loss_all:',D2_loss)
        #print ('>>D_loss_sum_all:',D_loss_sum_all)
        #print ('G_loss_D1:',G_loss_D1)
        print ('G_loss_L1:',G_loss_L1)
        print ('G_loss_contex:',G_loss_contex)
        print ('G_loss_localvar:',G_loss_localvar)
        print ('G_loss_globalvar:',G_loss_globalvar)
        print ('G_loss_ssim:',np.mean(ssim))  #target:->1
        print ('G_loss_psnr:',np.mean(psnr))  #target:>30
        print ('>>G_loss_sum_all:',G_loss_sum_all)
        
        return    None
       
    
    def Run_G(self, training=False):
        tepimgs,time_rates=self.getbatch_test_imgs()
        
        inerimg, \
        frame0_2_loss, interframe_GT_L1loss,ssim, psnr,\
        optical_0_2, optical_2_0=self.sess.run([self.G_net, \
                                                self.frame_0_2_L1loss, self.L1_loss_interframe,self.ssim, self.psnr,\
                                                self.opticalflow_0_2, self.opticalflow_2_0], \
                                                feed_dict={self.imgs_pla:tepimgs, self.training:training, self.timerates_pla:time_rates})
        
        return tepimgs, inerimg, interframe_GT_L1loss,  frame0_2_loss,  optical_0_2, optical_2_0, ssim, psnr
    
    
    def Run_D_TandF(self, training=False):
          
        '''
        #这里imgs要求是tanh化过的，即归一化到-1~1 
        training 为false时，bn会用学习的参数bn，因此在训练时的prob和测试时的prob又很大差异
        ''' 
        tepimgs,time_rates=self.getbatch_test_imgs()
        interframe_GT_L1 , L1loss_all,\
        ssim,psnr= self.sess.run([self.L1_loss_interframe, self.L1_loss_all, \
                                  self.ssim,self.psnr], \
                                        feed_dict={self.imgs_pla:tepimgs, self.training:training, self.timerates_pla:time_rates})
        return interframe_GT_L1, L1loss_all,ssim,psnr
    
    
    def eval_G_once(self, step=0):
        desdir=op.join(logdir, str(step))
        #os.makedirs(desdir, exist_ok=True)
        cnt=16
        col_cnt=G_group_img_num+1+2  #3个原始图像+1个生成+2个光流
        #中间用cnt像素的黑色线分隔图片
        bigimg_len=[ img_size_h*cnt+(cnt-1)*cnt, img_size_w*(col_cnt)+(col_cnt-1)*cnt]  #     img_size*cnt+(cnt-1)*cnt
        bigimg_bests=np.zeros([bigimg_len[0],bigimg_len[1],img_channel], dtype=np.uint8)
        
        for i in range(cnt):
            tepimgs, inerimg,  interframe_GT_L1loss, frame0_2_loss, optical_0_2, optical_2_0, ssim, psnr=self.Run_G()
            opticalflow=[optical_0_2, optical_2_0]
            
            #保存原图,这里已经弃用了
            for ind,j in enumerate(tepimgs[:int(cnt/4) ]):  
                #print (j[0][0][0])
                j=self.tanh2img(j) 
                imgname=str(i)+'_'+str(ind)
                dirinstep=op.join(desdir, imgname)
                '''
                for ki in range(G_group_img_num):      
                    im = j[:,:, ki*img_channel:(ki+1)*img_channel]
                    os.makedirs(dirinstep, exist_ok=True)
                    cv2.imwrite(op.join(dirinstep, str(ki)+'.jpg'), im)
                
                cv2.imwrite(op.join(dirinstep, 'fake_D1_'+str(D1_prob[ind])+'_D2_'+str(D2_prob[ind])+'.jpg'  ),inerimg[ind])
                '''
            #每个batch选随机的cnt个合成图片
            #print (probs.shape)
            tep=list(range(batchsize))
            tep=random.sample(tep, 1) #随机取1个图
            #print (tep)
            #every line is [frame1,flow1, frame2,flow2, frame3,fake img] total 6 imgs
            #加上原来的3帧
            for ki in range(G_group_img_num):
                st_x= ki*(img_size_w+cnt) #列
                st_y= i*(img_size_h+cnt) #行
                #pre process images
                pre_imgs=self.tanh2img(tepimgs[tep, :,:, ki*img_channel:(ki+1)*img_channel])[0]
                if ki==0:    pre_imgs=cv2.putText(pre_imgs,'frame0_2_loss:'+str(frame0_2_loss[tep]),(0,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                elif ki==1:  pre_imgs=cv2.putText(pre_imgs,'gan_loss_mean_batch:'+str(interframe_GT_L1loss),(0,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                elif ki==2:  pre_imgs=cv2.putText(pre_imgs,'ssim:'+str(ssim[tep])+" psnr:"+str(psnr[tep]),(0,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                '''
                print (type(pre_imgs))
                print (pre_imgs.shape)
                cv2.imshow('testing', pre_imgs)
                cv2.waitKey()
                '''
                
                bigimg_bests[st_y:st_y+img_size_h, st_x:st_x+img_size_w,:]=pre_imgs
            
            #加上光流图像
            for indki,ki in enumerate(opticalflow):
                #print (ki[tep,...][0].shape)
                #这里注意ki[tep].shape=(1, 180, 320, 2) 很诡异
                tepflow=self.flow_bgr(ki[tep][0])
                st_x= (indki+G_group_img_num)*(img_size_w+cnt) #列
                st_y= i*(img_size_h+cnt) #行
                bigimg_bests[st_y:st_y+img_size_h, st_x:st_x+img_size_w,:]=tepflow
            
            #加上生成图像
            st_x= (G_group_img_num+len(opticalflow))*(img_size_w+cnt) #列
            st_y= i*(img_size_h+cnt) #行
            inerimg=self.tanh2img(inerimg)
            bigimg_bests[st_y:st_y+img_size_h, st_x:st_x+img_size_w,:]=inerimg[tep]
        
        bigimg_name='step-'+str(step)+'_cnt-'+str(cnt)+'_batchsize-'+str(batchsize)+'.png'
        bigimg_dir=op.join(bigimgsdir, bigimg_name)
        
        cv2.imwrite(bigimg_dir, bigimg_bests)
            
        print ('eval_G_once,saved imgs to:',desdir, '\nbestimgs to:',bigimg_dir)
        
    def evla_D_once(self,eval_step=eval_step):
        cnt_real1=0
        cnt_fake1=0
        cnt_L1loss=0
        cnt_L1loss_all=0
        cnt_ssim=0
        cnt_psnr=0
        for i in range(eval_step):
            G_loss_squ_F,L1loss_all,ssim, psnr=self.Run_D_TandF()
            #print ('show prob shape:',probs.shape)  #[32,1]
            #cnt_fake1+=np.mean(prob_F)
            cnt_L1loss+=np.mean(G_loss_squ_F)
            cnt_L1loss_all+=np.mean(L1loss_all)
            #cnt_real1+=np.mean(prob_T)
            cnt_ssim+=np.mean(ssim)
            cnt_psnr+=np.mean(psnr)
            
        return cnt_L1loss/eval_step,cnt_L1loss_all/eval_step,cnt_ssim/eval_step,cnt_psnr/eval_step
        
    
    def Generator_net(self, inputdata1, inputdata2, withbias=G_withbias, filterlen=G_filter_len, layercnt=G_unet_layercnt):
        with tf.variable_scope("G_Net",  reuse=tf.AUTO_REUSE) as scope:
            #tepimg=my_unet(inputdata,  layercnt=G_unet_layercnt,  filterlen=filterlen,  training=self.training, withbias=withbias)
            
            #
            self.G_tanh=my_novel_unet(inputdata1, inputdata2, layercnt=layercnt, outchannel=G_optical_channel, filterlen=filterlen,  training=self.training, withbias=withbias)
            #####################################################################################################################################
            #tanh
            #self.G_tanh= tf.nn.tanh(self.G_tanh, name='G_tanh')
        
        return self.G_tanh
            
                
    
    def Discriminator_net_linear(self, imgs_3, withbias=D_1_withbias, filterlen=D_1_filterlen):
        '''
        :用于判定视频帧连续性的D
        imgs_3:9 channel input imgs[batchsize, h, w, 9]
        '''
        #这里输入的imgs应该是tanh后的，位于-1~1之间
        #cast to float 
        self.imgs_float32=tf.cast(imgs_3, tf.float32)
        
        layer_cnt=D_1_layercnt
        inputshape=imgs_3.get_shape().as_list()
        initchannel=inputshape[-1]
        
            
        with tf.variable_scope("D_1_Net",  reuse=tf.AUTO_REUSE) as scopevar:
            scope=scopevar.name
            tep=my_conv(self.imgs_float32, filterlen+int(layer_cnt/2), initchannel*2, scope+'_start', stride=3, withbias=withbias)
    
            tep=my_batchnorm( tep,self.training, scope)   #第一层不要bn试试
            tep=my_lrelu(tep, scope)
            
            for i in range(layer_cnt):
                stridetep=2+int( (layer_cnt-i)/4 )
                tep=my_D_block(tep, initchannel*( 2**(i+2)), scope+'_Dblock'+str(i),  stride=stridetep, filterlen=filterlen+int( (layer_cnt-i)/2 ), \
                               withbias=withbias, training=self.training)
                print (tep)
            #######################################################################################################################################
            #fc
            tep=my_fc(tep, 1024, scope+'_fc1',  withbias=withbias)
            tep=my_lrelu(tep, scope)
            
            tep=my_dropout(tep, self.training, dropout_rate)
            
            tep=my_fc(tep, 1, scope+'_fc2',  withbias=withbias)
            
            #sigmoid
            self.D_1_sigmoid=tf.nn.sigmoid(tep, name='D_1_sigmoid')
            print (self.D_1_sigmoid)
            
        return self.D_1_sigmoid, tep
    
    
    def Discriminator_net_clear(self, imgs, withbias=D_2_withbias, filterlen=D_2_filterlen):
        '''
        :用于判定生成视频帧的真实性的D
        imgs:input imgs [batchsize, h, w, 3]
        '''
        #这里输入的imgs应该是tanh后的，位于-1~1之间
        #cast to float 
        self.imgs_float32=tf.cast(imgs, tf.float32)
        
        layer_cnt=D_2_layercnt
        inputshape=imgs.get_shape().as_list()
        initchannel=inputshape[-1]
            
        with tf.variable_scope("D_2_Net",  reuse=tf.AUTO_REUSE) as scopevar:
            scope=scopevar.name
            tep=my_conv(self.imgs_float32, filterlen+int(layer_cnt/2), initchannel*2, scope+'_start', stride=3, withbias=withbias)
    
            #tep=my_batchnorm( tep,self.training, scope)   #第一层不要bn试试
            tep=my_lrelu(tep, scope)
            
            for i in range(layer_cnt):
                stridetep=2+int( (layer_cnt-i)/4 )
                tep=my_D_block(tep, initchannel*( 2**(i+2)), scope+'_Dblock'+str(i),stride=stridetep, filterlen=filterlen+int( (layer_cnt-i)/2 ), \
                               withbias=withbias, training=self.training)
                print (tep)
            #######################################################################################################################################
            #fc
            tep=my_fc(tep, 1024, scope+'_fc1',  withbias=withbias)
            tep=my_lrelu(tep, scope)
            
            tep=my_dropout(tep, self.training, dropout_rate)
            
            tep=my_fc(tep, 1, scope+'_fc2',  withbias=withbias)
            
            
            #sigmoid
            self.D_2_sigmoid=tf.nn.sigmoid(tep, name='D_2_sigmoid')
            
        return self.D_2_sigmoid, tep
        
        




if __name__ == '__main__':   
    with tf.Session(config=config) as sess:      
        gan=GAN_Net(sess)
        
        logwriter = tf.summary.FileWriter(logdir,   sess.graph)
        
        all_saver = tf.train.Saver(max_to_keep=2) 


        begin_t=time.time()
        for i in range(maxstep):     
            print ('\n',"{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
                   
            if ((i+1)%2000==0):#一次测试
                print ('begining to eval D:')
                L1loss,L1_loss_all,ssim_mean, psnr_mean=gan.evla_D_once()
                #print ('mean prob of real/fake:',prob_T,prob_F)
                
                #自己构建summary
                tsummary = tf.Summary()
                #tsummary.value.add(tag='mean prob of real1', simple_value=prob_T)
                #tsummary.value.add(tag='mean prob of fake1', simple_value=prob_F)
                tsummary.value.add(tag='mean L1_loss of G and GT', simple_value=L1loss)
                tsummary.value.add(tag='mean L1_loss_all of I0->I2', simple_value=L1_loss_all)
                tsummary.value.add(tag='mean ssim:', simple_value=ssim_mean)
                tsummary.value.add(tag='mean psnr:', simple_value=psnr_mean)
                #写入日志
                logwriter.add_summary(tsummary, i)
                
            if i==0 or (i+1)%1500==0:#保存一波图片
                gan.eval_G_once(i)
                
                
            if (i+1)%2000==0:#保存模型
                print ('saving models...')
                pat=all_saver.save(sess, op.join(logdir,'model_keep'),global_step=i)
                print ('saved at:',pat)
            
            
            stt=time.time()
            print ('%d/%d  start train_once...'%(i,maxstep))
            #lost,sum_log=vgg.train_once(sess) #这里每次训练都run一个summary出来
            sum_log=gan.train_once_all()
            #写入日志
            if sum_log:logwriter.add_summary(sum_log, i)
            #print ('write summary done!')
            
            #######################
            if (i+1)%5==0:#偶尔测试一次
                L1_loss,L1_loss__all,ssim, psnr=gan.evla_D_once(1)
                #print ('once prob of D1 real/fake:',prob_TT,'/',prob_FF)
                print ("once L1 loss of inter frame/all l1 loss:",L1_loss,'/',L1_loss__all)
                print ("once ssim and psnr:",ssim,'/', psnr)
                
            print ('time used:',time.time()-stt,' to be ',1.0/(time.time()-stt),' iters/s', ' left time:',(time.time()-stt)*(maxstep-i)/60/60,' hours')
            
        
        print ('Training done!!!-->time used:',(time.time()-begin_t),'s = ',(time.time()-begin_t)/60/60,' hours')








