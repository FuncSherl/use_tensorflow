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
from superslomo_tensorflow import *
#from data import create_dataset2 as cdata
#import skimage

#this version change output of g to be img
#and chagne img-size to v2's 1/2
#use wgan loss function
#my_novel_conv
#use l2 loss for img clear
#use global rate to mult square loss
version="_TestSplitUnet"

logdir="./logs_superslomo/SuperSlomo_"+TIMESTAMP+('_base_lr-%f_batchsize-%d_maxstep-%d'%(base_lr,batchsize, maxstep))+version

kepimgdir=op.join(logdir, "zerostateimgs")
os.makedirs(kepimgdir,  exist_ok=True)

#设置GPU显存按需增长
gpu_options = tf.GPUOptions(allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class SuperSlomo_TestSplitUnet(SuperSlomo):
    def __init__(self, sess):
        self.sess=sess
        self.global_step = tf.Variable(0.0, name='global_step',dtype=tf.float32, trainable=False)
        
        #for data input
        self.pipline_data_train=cdata.get_pipline_data_train(img_size, batchsize)
        self.pipline_data_test=cdata.get_pipline_data_test(img_size, batchsize_test)
        
        #3个placeholder， img和noise,training 
        self.imgs_pla = tf.placeholder(tf.float32, [batchsize, img_size_h, img_size_w, G_group_img_num*img_channel], name='imgs_in')
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
        
        with tf.variable_scope("first_unet",  reuse=tf.AUTO_REUSE) as scope:
            firstinput=tf.concat([self.frame0, self.frame2], -1)
            #self.first_opticalflow=my_unet( firstinput, 4, withbias=True)  #注意这里是直接作为optical flow
            self.first_opticalflow=my_unet_split( firstinput, 4,training=self.training , withbias=True, withbn=False)  #注意这里是直接作为optical flow
            
        self.first_opticalflow_0_1=self.first_opticalflow[:, :, :, :2]
        self.first_opticalflow_0_1=tf.identity(self.first_opticalflow_0_1, name="first_opticalflow_0_1")
        print ('first_opticalflow_0_1:',self.first_opticalflow_0_1)
        self.first_opticalflow_1_0=self.first_opticalflow[:, :, :, 2:]
        self.first_opticalflow_1_0=tf.identity(self.first_opticalflow_1_0, name="first_opticalflow_1_0")
        print ('first_opticalflow_1_0:',self.first_opticalflow_1_0)
        #first_opticalflow_0_1: Tensor("first_opticalflow_0_1:0", shape=(10, 180, 320, 2), dtype=float32)
        #first_opticalflow_1_0: Tensor("first_opticalflow_1_0:0", shape=(10, 180, 320, 2), dtype=float32)
        
        #输出光流形状
        self.flow_size_h=self.first_opticalflow_0_1.get_shape().as_list()[1]
        self.flow_size_w=self.first_opticalflow_0_1.get_shape().as_list()[2]
        self.flow_channel=self.first_opticalflow_0_1.get_shape().as_list()[-1]
        
        self.flow_shape=[ self.flow_size_h, self.flow_size_w, self.flow_channel*2]
        
        #反向光流算中间帧
        self.first_opticalflow_t_0=tf.add( -(1-self.timerates_expand)*self.timerates_expand*self.first_opticalflow_0_1 ,\
                                      self.timerates_expand*self.timerates_expand*self.first_opticalflow_1_0 , name="first_opticalflow_t_0")
        self.first_opticalflow_t_2=tf.add( (1-self.timerates_expand)*(1-self.timerates_expand)*self.first_opticalflow_0_1 ,\
                                      self.timerates_expand*(self.timerates_expand-1)*self.first_opticalflow_1_0, name="first_opticalflow_t_2")

        #2种方法合成t时刻的帧
        self.first_img_flow_2_t=self.warp_op(self.frame2, -self.first_opticalflow_t_2) #!!!
        self.first_img_flow_0_t=self.warp_op(self.frame0, -self.first_opticalflow_t_0) #!!!
        
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
                                self.first_img_flow_2_t, self.first_img_flow_0_t], -1)
            print (secinput)
            self.second_opticalflow=my_unet( secinput, 5, withbias=True)  #注意这里是直接作为optical flow
        self.second_opticalflow_t_0=tf.add( self.second_opticalflow[:,:,:,:2],  self.first_opticalflow_t_0, name="second_opticalflow_t_0")
        self.second_opticalflow_t_1=tf.add( self.second_opticalflow[:,:,:,2:4], self.first_opticalflow_t_2, name="second_opticalflow_t_1")
        print ('second_opticalflow_t_0:',self.second_opticalflow_t_0)
        print ('second_opticalflow_t_1:',self.second_opticalflow_t_1)
        #second_opticalflow_t_0: Tensor("second_opticalflow_t_0:0", shape=(10, 180, 320, 2), dtype=float32)
        #second_opticalflow_t_1: Tensor("second_opticalflow_t_1:0", shape=(10, 180, 320, 2), dtype=float32)
        
        self.vmap_t_0=tf.expand_dims( tf.sigmoid(self.second_opticalflow[:,:,:,-1])  , -1)
        self.vmap_t_1=1-self.vmap_t_0

        #2种方法合成t时刻的帧
        self.second_img_flow_1_t=self.warp_op(self.frame2, -self.second_opticalflow_t_1) #!!!
        self.second_img_flow_0_t=self.warp_op(self.frame0, -self.second_opticalflow_t_0) #!!!
        
        #最终输出的图
        print (self.timerates_expand, self.vmap_t_0, self.second_img_flow_0_t)
        #Tensor("ExpandDims_2:0", shape=(6, 1, 1, 1), dtype=float32) Tensor("Sigmoid:0", shape=(6, 180, 320, 1), dtype=float32) 
        #Tensor("dense_image_warp_5/Reshape_1:0", shape=(6, 180, 320, 3), dtype=float32)
        self.second_output=tf.div(  ( (1-self.timerates_expand)*self.vmap_t_0*self.second_img_flow_0_t+self.timerates_expand*self.vmap_t_1*self.second_img_flow_1_t),  \
                             ((1-self.timerates_expand)*self.vmap_t_0+self.timerates_expand*self.vmap_t_1) , name="second_outputimg" )
        print ('second output img:',self.second_output)
        #second output img: Tensor("second_outputimg:0", shape=(10, 180, 320, 3), dtype=float32)
        
        #计算loss
        self.second_L1_loss_interframe,self.first_warp_loss,self.second_contex_loss,self.second_local_var_loss_all,self.second_global_var_loss_all,self.second_ssim,self.second_psnr,\
                self.first_L1_loss_interframe, self.first_ssim, self.first_psnr=self.loss_cal_all()
                
        self.G_loss_all=204 * self.second_L1_loss_interframe + 102 * self.first_warp_loss + 0.005 * self.second_contex_loss + self.second_global_var_loss_all        
        
        
        #获取数据时的一些cpu上的参数，用于扩张数据和判定时序
        self.last_flow_init_np=np.zeros(self.flow_shape, dtype=np.float32)
        print (self.last_flow_init_np.shape) #(180, 320, 4)
        
        #初始化train和test的初始0状态
        self.last_flow_new_train=self.last_flow_init_np
        self.last_flow_new_test=self.last_flow_init_np
        
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
        print ("first param len:",len(self.first_para))
        print ("second param len:",len(self.sec_para))
        print ("VGG param len:",len(self.vgg_para))
        print (self.vgg_para)
        '''
        trainable vars cnt: 144
        first param len: 46
        second param len: 46
        VGG param len: 52
        '''
        
        #训练过程
        self.lr_rate = tf.train.exponential_decay(base_lr,  global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate)
        self.train_op = tf.train.AdamOptimizer(self.lr_rate, name="superslomo_adam").minimize(self.G_loss_all,  \
                                                                                              global_step=self.global_step  , var_list=self.first_para+self.sec_para  )
        
        # weight clipping
        #self.clip_D = [p.assign(tf.clip_by_value(p, weightclip_min, weightclip_max)) for p in self.D_para]
        
        #最后构建完成后初始化参数 
        self.sess.run(tf.global_variables_initializer())
        
        
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
            
        
            
        return L1_loss_interframe,warp_loss,contex_loss,local_var_loss_all,global_var_loss_all,ssim,psnr,first_L1_loss_interframe, first_ssim, first_psnr

    def train_once(self):
        imgdata,rate,newstate=self.getbatch_train_imgs()
        if newstate: 
            self.last_flow_new_train=self.last_flow_init_np
            print ('start from a zero flow!!!')
        
        _, \
        sec_ssim,sec_psnr,      \
        sec_contexloss, sec_L1_loss,first_loss_warp, \
        sec_localloss,sec_globalloss,loss_all,\
        first_L1_loss, first_ssim, first_psnr=self.sess.run([self.train_op,\
                                    self.second_ssim, self.second_psnr, \
                                    self.second_contex_loss,  self.second_L1_loss_interframe,self.first_warp_loss, \
                                    self.second_local_var_loss_all,self.second_global_var_loss_all,  self.G_loss_all, \
                                    self.first_L1_loss_interframe, self.first_ssim, self.first_psnr], \
                      feed_dict={self.imgs_pla:imgdata,self.timerates_pla:rate})
        
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
        
        print ("first  l1_loss_all:",first_L1_loss)
        print ("second l1_loss_all:",sec_L1_loss)
        
        print ("first l1_loss_warp:",first_loss_warp)
        print ("loss_all :",loss_all)
        #print ("loss_all 2:",loss_all2)
        
        return loss_all

    def eval_once(self, step,evalstep=100):
        kep_img_dir=op.join(kepimgdir, str(step))
        os.makedirs(kep_img_dir, exist_ok=True)
        
        recording=0  #遇到一个新的状态开始记录，到下一个状态停止
        kep_ssim1=0.0
        kep_psnr1=0.0
        kep_l1loss1=0.0
        kep_l1loss2=0.0
        kep_contexloss1=0.0
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
                
            step1_flow_0_1,step1_flow_1_0,step1_imgout,step2_flow_t_0,step2_flow_t_1,step2_outimg   ,\
            ssim1,ssim2,psnr1, psnr2,     \
            contexloss2, L1_loss1, L1_loss2,L1_loss_warp1, \
            localloss2,globalloss2,loss_all=self.sess.run([self.first_opticalflow_0_1,self.first_opticalflow_1_0,self.first_output ,self.second_opticalflow_t_0,self.second_opticalflow_t_1, self.second_output              ,\
                                    self.first_ssim,self.second_ssim, self.first_psnr, self.second_psnr,\
                                    self.second_contex_loss,  self.first_L1_loss_interframe, self.second_L1_loss_interframe,self.first_warp_loss, \
                                    self.second_local_var_loss_all,self.second_global_var_loss_all,  self.G_loss_all,\
                                    ], \
                      feed_dict={self.imgs_pla:imgdata,self.timerates_pla:rate})
        
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
        print ("write "+str(img_cnt)+" imgs to:"+kep_img_dir)
        return [kep_ssim1/evalstep,kep_ssim2/evalstep], [kep_psnr1/evalstep,kep_psnr2/evalstep], [kep_l1loss1/evalstep,kep_l1loss2/evalstep], kep_loss_warp1/evalstep, kep_contexloss2/evalstep,  kep_localloss2/evalstep, kep_globalloss2/evalstep
    

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
                ssim_mean, psnr_mean,interframeloss,warploss, contexloss,localvarloss,globalloss=gan.eval_once(i)
                
                #自己构建summary
                tsummary = tf.Summary()
                #tsummary.value.add(tag='mean prob of real1', simple_value=prob_T)
                #tsummary.value.add(tag='mean prob of fake1', simple_value=prob_F)
                tsummary.value.add(tag='first mean L1_loss of G and GT', simple_value=interframeloss[0])     
                tsummary.value.add(tag='second mean L1_loss of G and GT', simple_value=interframeloss[1]) 
                tsummary.value.add(tag='first mean L1_loss of warps', simple_value=warploss)           
                tsummary.value.add(tag='second mean contexloss', simple_value=contexloss)
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
















