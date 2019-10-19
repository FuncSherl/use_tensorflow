'''
Created on Sep 27, 2019

@author: sherl
'''
import tensorflow as tf
from datetime import datetime
import os.path as op
from data import create_dataset_step2 as cdata
import  GAN_tools_common as mytools
import numpy as np
import cv2, os, random, time
from GAN_tools_v25 import *

print ('tensorflow version:',tf.__version__,'  path:',tf.__path__)
TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
#图像形状 
flow_channel = 2

#kernel concern
output_channel = 12
batchsize = 1  # 这里一定为1，因为下面会将第一部的输出里面的batchsize当作timestep使用，这里的batchsize默认就是1了
kernel_len = 3


#数据输入的样式，一套3个图，每个图3channel，前后帧和t时刻帧 
group_img_num=3
img_channel=3


timestep = 12  #这里与第一部中的batchsize相同

#hyper param 
dropout=0.5 
leakyrelurate=0.2
stddev=0.01
bias_init=0.0
LR=1e-3  #0.0001
LR_step1=2e-4
decay_steps=12000
decay_rate=0.9
maxstep=240000 #训练多少次

weightclip_min=-0.01
weightclip_max=0.01

mean_dataset=[102.1, 109.9, 110.0]  #0->1 is [0.4, 0.43, 0.43]
modelpath=cdata.modelpath
meta_name = r'model_keep-239999.meta'

# 设置GPU显存按需增长
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)

logdir="./logs_v24_Step2_v2/GAN_"+TIMESTAMP+('_base_lr-%f_batchsize-%d_maxstep-%d'%(LR,batchsize, maxstep))
kepimgdir=op.join(logdir, "zerostateimgs")
os.makedirs(kepimgdir,  exist_ok=True)

class Step2_ConvLstm:

    def __init__(self, sess):
        self.sess = sess

        #加载原模型
        saver = tf.train.import_meta_graph(op.join(modelpath, meta_name) )
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        self.graph = tf.get_default_graph()
        self.global_step = tf.Variable(0.0, name='step2_global_step',dtype=tf.float32, trainable=False)
        
        #placeholders
        self.imgs_pla= self.graph.get_tensor_by_name('imgs_in:0')
        self.training= self.graph.get_tensor_by_name("training_in:0")
        self.timerates= self.graph.get_tensor_by_name("timerates_in:0")
        
        #self.timerates_pla=tf.placeholder(tf.float32, [batchsize], name='step2_inner_timerates_in')
        self.timerates_expand=tf.expand_dims(self.timerates, -1)
        self.timerates_expand=tf.expand_dims(self.timerates_expand, -1)
        self.timerates_expand=tf.expand_dims(self.timerates_expand, -1) #batchsize*1*1*1
        
        #tesorfs
        #这里是第一部的输出光流
        self.optical_0_1=self.graph.get_tensor_by_name("G_opticalflow_0_2:0")
        self.optical_1_0=self.graph.get_tensor_by_name("G_opticalflow_2_0:0")
        self.outimg = self.graph.get_tensor_by_name("G_net_generate:0")
        
        
        
        
        #第一部中的batchsize，这里可以当作timestep使用
        self.batchsize_inputimg=self.imgs_pla.get_shape().as_list()[0]
        
        #输入图像形状
        self.img_size_w=self.imgs_pla.get_shape().as_list()[2]
        self.img_size_h=self.imgs_pla.get_shape().as_list()[1]
        self.img_size=[self.img_size_h, self.img_size_w]
        print (self.imgs_pla) #Tensor("imgs_in:0", shape=(12, 180, 320, 9), dtype=float32)
        self.frame0=self.imgs_pla[:,:,:,:img_channel]
        self.frame1=self.imgs_pla[:,:,:,img_channel:img_channel*2]
        self.frame2=self.imgs_pla[:,:,:,img_channel*2:]
        
        self.pipline_data_train=cdata.get_pipline_data_train(self.img_size, self.batchsize_inputimg)
        self.pipline_data_test=cdata.get_pipline_data_test(self.img_size, self.batchsize_inputimg)
        #输出光流形状
        self.flow_size_h=self.optical_0_1.get_shape().as_list()[1]
        self.flow_size_w=self.optical_0_1.get_shape().as_list()[2]
        self.flow_channel=self.optical_0_1.get_shape().as_list()[-1]
        
        self.flow_shape=[ self.flow_size_h, self.flow_size_w, self.flow_channel*2]
        
        #last flow placeholder
        self.last_optical_flow=tf.placeholder(tf.float32, self.flow_shape, name='step2_last_flow')
        
        
        #这里将batch中的第一组中的前后帧和前后光流拼起来
        input_pla=tf.concat([ self.frame0[0], self.optical_0_1[0], self.optical_1_0[0], self.last_optical_flow, self.frame2[0] ], -1)  #这里将两个光流拼起来 可以考虑将前后帧也拼起来
        print (input_pla)  #Tensor("concat_9:0", shape=(180, 320, 14), dtype=float32)
        
        with tf.variable_scope("STEP2",  reuse=tf.AUTO_REUSE) as scopevar:
            new_flow=self.step2_network(input_pla)
            kep_new_flow=[new_flow]
            
            for ti in range(1,self.batchsize_inputimg):
                input_pla=tf.concat([ self.frame0[ti], self.optical_0_1[ti], self.optical_1_0[ti], new_flow, self.frame2[ti] ], -1) #14
                new_flow=self.step2_network(input_pla)
                kep_new_flow.append(new_flow)
                
            self.flow_next=new_flow
            self.flow_after=tf.stack(kep_new_flow, axis=0, name='step2_opticalflow')
            print ('self.flow_after:',self.flow_after)  #Tensor("STEP2/step2_opticalflow:0", shape=(12, 180, 320, 4), dtype=float32)
            print ('self.flow_next:',self.flow_next)  #Tensor("STEP2/strided_slice_79:0", shape=(180, 320, 4), dtype=float32)
            
        self.last_flow_init_np=np.zeros(self.flow_shape, dtype=np.float32)
        print (self.last_flow_init_np.shape) #(180, 320, 4)
        #初始化train和test的初始0状态
        self.last_flow_new_train=self.last_flow_init_np
        self.last_flow_new_test=self.last_flow_init_np
        
        #########################################################################
        self.opticalflow_0_2=tf.slice(self.flow_after, [0, 0, 0, 0], [-1, -1, -1, 2], name='step2_opticalflow_0_2')
        self.opticalflow_2_0=tf.slice(self.flow_after, [0, 0, 0, 2], [-1, -1, -1, 2], name='step2_opticalflow_2_0')
        print ('original flow:',self.opticalflow_0_2, self.opticalflow_2_0)
        
        #获取数据时的一些cpu上的参数，用于扩张数据和判定时序
        self.last_label_train='#'
        self.last_label_test='#'
        self.state_random_row_train=0
        self.state_random_col_train=0
        self.state_flip_train=False
        
        self.state_random_row_test=0
        self.state_random_col_test=0
        self.state_flip_test=False
        
        t_vars=tf.trainable_variables()
        print ("trainable vars cnt:",len(t_vars))
        self.G_para=[var for var in t_vars if var.name.startswith('G')]
        self.D_para=[var for var in t_vars if var.name.startswith('D')]
        self.STEP2_para=[var for var in t_vars if var.name.startswith('STEP2')]
        print ("G param len:",len(self.G_para))
        print ("D param len:",len(self.D_para))
        print ("STEP2 param len:",len(self.STEP2_para))
        print (self.STEP2_para)
        '''
        trainable vars cnt: 184
        G param len: 60
        D param len: 16
        STEP2 param len: 56
        相比于前面不加第二部的128个，这里注意将VGG与step1中的VGG共享参数，否则会白白多用内存
        '''
        
        # weight clipping
        self.clip_D = [p.assign(tf.clip_by_value(p, weightclip_min, weightclip_max)) for p in self.D_para]
        
        self.step1_train_op,\
        self.step1_L1_loss_all,self.step1_contex_loss,self.step1_local_var_loss_all,self.step1_G_loss_all,\
        self.step1_ssim,self.step1_psnr,self.step1_G_net=self.loss_cal(self.optical_0_1, self.optical_1_0, LR_step1, self.G_para, scopevar.name)
        
        self.step2_train_op,\
        self.step2_L1_loss_all,self.step2_contex_loss,self.step2_local_var_loss_all,self.step2_G_loss_all,\
        self.step2_ssim,self.step2_psnr,self.step2_G_net=self.loss_cal(self.opticalflow_0_2, self.opticalflow_2_0, LR, self.STEP2_para, scopevar.name)
        
        
        #最后构建完成后初始化参数 
        self.sess.run(tf.global_variables_initializer())
        
    def step2_network(self, inputdata, outchannel=4, layercnt=3, filterlen=3, training=True,  withbias=True):
        #input:concat后的单个图[180,320,14]
        #return: new optical flow [180, 320, 4]注意是双向光流
        inputdata=tf.expand_dims(inputdata, 0)
        inputshape=inputdata.get_shape().as_list()
        channel_init=inputshape[-1]
        
        tep=my_conv(inputdata, filterlen+int(layercnt/2), channel_init*2, scopename='unet_down_start', stride=1,  withbias=withbias)
        tep=my_lrelu(tep, 'unet_down_start')
        
        print ('\nforming UNET-->layer:',layercnt)
        print (tep) #Tensor("STEP2/unet_down_start_1/LeakyRelu:0", shape=(1, 180, 320, 28)
        skipcon=[]
        for i in range(layercnt):
            skipcon.append(tep)
            tep=unet_down(tep, channel_init*( 2**(i+2)), 'unet_down_'+str(i), filterlen=filterlen+int( (layercnt-i)/2 ), training=training,withbias=withbias)
            print (tep)
            
        '''
        # 这里不将channel变为两倍了
        tep=unet_down(tep, channel_init*( 2**(i+2)), 'unet_down_'+str(i+1), filterlen=filterlen , withbias=withbias)
        print (tep)
        '''
        
        for i in reversed(range(layercnt)):
            tep=unet_up(tep, channel_init*( 2**(i+1)), skipcon[i],'unet_up_'+str(i), filterlen=filterlen+int( (layercnt-i)/3 ),  training=training,withbias=withbias)
            print (tep)
        
        #这里注意原图的位置一定要在input的最前和后，否者这里需要对应修改concat的位置
        tep=tf.concat([ inputdata[...,:img_channel], tep, inputdata[...,-img_channel:] ], -1)
        tep=my_conv(tep, filterlen, outchannel*2, scopename='unet_up_end0', stride=1, withbias=withbias)
        
        tep=my_batchnorm( tep,training, 'unet_up_end0_bn2')
        tep=my_lrelu(tep, 'unet_up_end0_relu')
        #print (tep)
        
        tep=my_conv(tep, filterlen, outchannel, scopename='unet_up_end1', stride=1, withbias=withbias)
        tep=tf.image.resize_images(tep, [inputshape[1],inputshape[2]], method=tf.image.ResizeMethod.BILINEAR)
        print (tep)  #Tensor("STEP2/unet_up_end1_10/BiasAdd:0", shape=(1, 180, 320, 4), dtype=float32)
        
        return tep[0]
            
    
    
        
    def loss_cal(self,opticalflow_0_2, opticalflow_2_0, lr, varlist, name='STEP2'):
        with tf.variable_scope(name,  reuse=tf.AUTO_REUSE):
            #反向光流算中间帧
            opticalflow_t_0=tf.add( -(1-self.timerates_expand)*self.timerates_expand*opticalflow_0_2 ,\
                                          self.timerates_expand*self.timerates_expand*opticalflow_2_0 , name="step2_opticalflow_t_0")
            opticalflow_t_2=tf.add( (1-self.timerates_expand)*(1-self.timerates_expand)*opticalflow_0_2 ,\
                                          self.timerates_expand*(self.timerates_expand-1)*opticalflow_2_0, name="step2_opticalflow_t_2")
            
            print ('two optical flow:',opticalflow_t_0, opticalflow_t_2)
            #two optical flow: Tensor("step2_opticalflow_t_0:0", shape=(12, 180, 320, 2), dtype=float32) Tensor("step2_opticalflow_t_2:0", shape=(12, 180, 320, 2), dtype=float32)
            
            #2种方法合成t时刻的帧
            img_flow_2_t=self.warp_op(self.frame2, -opticalflow_t_2) #!!!
            img_flow_0_t=self.warp_op(self.frame0, -opticalflow_t_0) #!!!
            
            G_net=tf.add(self.timerates_expand*img_flow_2_t , (1-self.timerates_expand)*img_flow_0_t, name="step2_net_generate" )
            print ("generated iner frame:",G_net)
            #利用光流前后帧互相合成
            img_flow_2_0=self.warp_op(self.frame2, opticalflow_2_0)  #frame2->frame0
            img_flow_0_2=self.warp_op(self.frame0, opticalflow_0_2)  #frame0->frame2
            
            
            #1、contex loss
            print ("forming conx loss：")
            tep_G_shape=G_net.get_shape().as_list()[1:]
            
        with tf.variable_scope("VGG16",  reuse=tf.AUTO_REUSE):
            contex_Genera =tf.keras.applications.VGG16(include_top=False, input_tensor=G_net,  input_shape=tep_G_shape).get_layer("block4_conv3").output
            contex_frame1 =tf.keras.applications.VGG16(include_top=False, input_tensor=self.frame1, input_shape=tep_G_shape).get_layer("block4_conv3").output
            
            contex_loss=   tf.reduce_mean(tf.squared_difference( contex_frame1, contex_Genera), name='step2_Contex_loss')
            print ('step2_loss_mean_contex form finished..')
            
        with tf.variable_scope(name,  reuse=tf.AUTO_REUSE):
            #2、L1 loss
            print ("forming L1 loss:生成帧与GT、frame2->frame0与frame0、frame0->frame2与frame2")
            L1_loss_interframe =tf.reduce_mean(tf.abs(  G_net-self.frame1  ))
            L1_loss_all        =tf.reduce_mean(tf.abs(  G_net-self.frame1  ) + \
                                                    tf.abs(img_flow_2_0-self.frame0) + \
                                                    tf.abs(img_flow_0_2-self.frame2), name='step2_G_clear_l1_loss')
            #self.G_loss_mean_Square=  self.contex_loss*1 + self.L1_loss_all
            print ('step2_loss_mean_l1 form finished..')
            
            #4 local var loss
            local_var_loss_0_2=self.local_var_loss(opticalflow_0_2)
            local_var_loss_2_0=self.local_var_loss(opticalflow_2_0)
            #print ("local _var loss:",self.local_var_loss_0_2,  self.G_loss_mean_D1)
            #local _var loss: Tensor("mean_local_var:0", shape=(), dtype=float32) Tensor("Mean_3:0", shape=(), dtype=float32)
            local_var_loss_all=tf.add(local_var_loss_0_2, local_var_loss_2_0, name="step2_local_var_add")
            
            #5 global var loss
            global_var_loss_0_2=self.global_var_loss(opticalflow_0_2)
            global_var_loss_2_0=self.global_var_loss(opticalflow_2_0)
            global_var_loss_all=tf.add(global_var_loss_0_2, global_var_loss_2_0, name="step2_global_var_add")
            
            #6 SSIM
            ssim = tf.image.ssim(G_net, self.frame1, max_val=2.0)
            print ("ssim:",ssim)  #ssim: Tensor("Mean_10:0", shape=(12,), dtype=float32)
            
            #7 PSNR
            psnr = tf.image.psnr(G_net, self.frame1, max_val=2.0, name="step2_frame1_psnr")
            print ("psnr:", psnr) #psnr: Tensor("G_frame1_psnr/Identity_3:0", shape=(12,), dtype=float32)
            
            G_loss_all=contex_loss*5 + L1_loss_all*10 +  local_var_loss_all*0.06
            
            self.lr_rate = tf.train.exponential_decay(lr,  global_step=self.global_step/2, decay_steps=decay_steps, decay_rate=decay_rate)
            train_op = tf.train.AdamOptimizer(self.lr_rate, name=name+"_step2_adam").minimize(G_loss_all,  global_step=self.global_step,var_list=varlist)
            
        return train_op,L1_loss_all,contex_loss,local_var_loss_all,G_loss_all,ssim,psnr,G_net

    
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
        img-=mean_dataset*3
        return img*2.0/255-1
    
    def tanh2img(self,tanhd):
        tep= (tanhd+1)*255//2
        #print ('tep.shape:',tep.shape)  #tep.shape: (180, 320, 9)
        multly=int(tep.shape[-1]/len(mean_dataset))
        #print ('expanding:',multly)
        tep+=mean_dataset*multly
        return tep.astype(np.uint8)
        

    def train_once(self):
        imgdata,rate,newstate=self.getbatch_train_imgs()
        if newstate: 
            self.last_flow_new_train=self.last_flow_init_np
            print ('start from a zero flow!!!')
        
        _, _,\
        self.last_flow_new_train,   \
        ssim1,ssim2,psnr1,psnr2,      \
        contexloss1, contexloss2,L1_loss1, L1_loss2,\
        localloss1,localloss2,loss_all1,loss_all2=self.sess.run([self.step1_train_op,self.step2_train_op,\
                                    self.flow_next, \
                                    self.step1_ssim, self.step2_ssim, self.step1_psnr, self.step2_psnr, \
                                    self.step1_contex_loss, self.step2_contex_loss, self.step1_L1_loss_all,self.step2_L1_loss_all, \
                                    self.step1_local_var_loss_all, self.step2_local_var_loss_all, self.step1_G_loss_all, self.step2_G_loss_all], \
                      feed_dict={self.imgs_pla:imgdata,self.timerates:rate,self.training:True,  self.last_optical_flow:self.last_flow_new_train})
        
        #print ()
        print ("train once:")
        print ("ssim1:",ssim1)
        print ("ssim2:",ssim2)
        print ("psnr1:",psnr1)
        print ("psnr2:",psnr2)
        print ("contexloss1:",contexloss1)
        print ("contexloss2:",contexloss2)
        print ("local var loss 1:",localloss1)
        print ("local var loss 2:",localloss2)
        print ("l1_loss_all 1:",L1_loss1)
        print ("l1_loss_all 2:",L1_loss2)
        print ("loss_all 1:",loss_all1)
        print ("loss_all 2:",loss_all2)
        
        return [loss_all1, loss_all2]

    def eval_once(self, step,evalstep=100):
        kep_img_dir=op.join(kepimgdir, str(step))
        os.makedirs(kep_img_dir, exist_ok=True)
        
        recording=0  #遇到一个新的状态开始记录，到下一个状态停止
        kep_ssim1=0.0
        kep_psnr1=0.0
        kep_l1loss1=0.0
        kep_contexloss1=0.0
        kep_localloss1=0
        kep_ssim2=0.0
        kep_psnr2=0.0
        kep_l1loss2=0.0
        kep_contexloss2=0.0
        kep_localloss2=0
        
        img_cnt=0
        
        for i in range(evalstep):
            imgdata,rate,newstate=self.getbatch_test_imgs()
            if newstate: 
                self.last_flow_new_test=self.last_flow_init_np
                recording+=1
    
            self.last_flow_new_test,   \
            ssim1,ssim2,psnr1,psnr2,      \
            contexloss1,contexloss2, L1_loss1, L1_loss2, \
            localvarloss1, localvarloss2,loss_all1,loss_all2,\
            step1_flow_0_1,step1_flow_1_0,step1_imgout,step2_flow_0_1,step2_flow_1_0,step2_outimg=self.sess.run([
                                        self.flow_next, \
                                        self.step1_ssim, self.step2_ssim, self.step1_psnr, self.step2_psnr, \
                                        self.step1_contex_loss, self.step2_contex_loss, self.step1_L1_loss_all,self.step2_L1_loss_all,\
                                        self.step1_local_var_loss_all, self.step2_local_var_loss_all, self.step1_G_loss_all, self.step2_G_loss_all,\
                                        self.optical_0_1, self.optical_1_0, self.step1_G_net, self.opticalflow_0_2,self.opticalflow_2_0,self.step2_G_net], \
                          feed_dict={self.imgs_pla:imgdata,self.timerates:rate,self.training:False,  self.last_optical_flow:self.last_flow_new_test})
        
            kep_ssim1+=np.mean(ssim1)
            kep_ssim2+=np.mean(ssim2)
            kep_psnr1+=np.mean(psnr1)
            kep_psnr2+=np.mean(psnr2)
            kep_l1loss1+=np.mean(L1_loss1)
            kep_l1loss2+=np.mean(L1_loss2)
            kep_contexloss1+=np.mean(contexloss1)
            kep_contexloss2+=np.mean(contexloss2)
            kep_localloss1+=localvarloss1
            kep_localloss2+=localvarloss2
            
            if recording==1:
                tep=self.form_bigimg(imgdata,step1_flow_0_1,step1_flow_1_0,step1_imgout,step2_flow_0_1,step2_flow_1_0,step2_outimg)
                cv2.imwrite( op.join(kep_img_dir, "recording_"+str(recording)+"_"+str(img_cnt)+'.jpg') ,  tep)
                img_cnt+=1
                
                    
            
        print ("eval ",evalstep,' times:','\nmean ssim1:',kep_ssim1/evalstep,'  mean ssim2:',kep_ssim2/evalstep,\
               '\nmean psnr1:',kep_psnr1/evalstep,'  mean psnr2:',kep_psnr2/evalstep,\
               '\nmean l1loss1:',kep_l1loss1/evalstep,'   mean l1loss2:',kep_l1loss2/evalstep,\
               '\nmean contexloss1:',kep_contexloss1/evalstep,'   mean contexloss2:',kep_contexloss2/evalstep,\
               '\nmean localvar loss1:',kep_localloss1/evalstep, "   mean localvar loss2:",kep_localloss2/evalstep)
        print ("write "+str(img_cnt)+" imgs to:"+kep_img_dir)
        return [kep_ssim1/evalstep,kep_ssim2/evalstep],[kep_psnr1/evalstep, kep_psnr2/evalstep], \
            [kep_contexloss1/evalstep, kep_contexloss2/evalstep],[kep_l1loss1/evalstep, kep_l1loss2/evalstep], [kep_localloss1/evalstep, kep_localloss2/evalstep]
    
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
            for k in range(group_img_num): 
                ret[row:row+self.img_size_h, col:col+self.img_size_w]=imgdata[j, :,:,k*img_channel:(k+1)*img_channel]
                col+=self.img_size_w+gap
            #3
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step1_flow_0_1[j])
            col+=self.img_size_w+gap
            #4
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step1_flow_1_0[j])
            col+=self.img_size_w+gap
            #5
            mean_l1=np.mean( np.abs(step1_imgout[j]-imgdata[j, :,:,1*img_channel:(1+1)*img_channel]) )*2/255
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=cv2.putText(step1_imgout[j],'GT_step1_L1loss:'+str(mean_l1),(0,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            col+=self.img_size_w+gap
            #6
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step2_flow_0_1[j])
            col+=self.img_size_w+gap
            #7
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step2_flow_1_0[j])
            col+=self.img_size_w+gap
            #8
            mean_l1=np.mean( np.abs(step2_outimg[j]-imgdata[j, :,:,1*img_channel:(1+1)*img_channel]) )*2/255
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
        gan = Step2_ConvLstm(sess)
        
        logwriter = tf.summary.FileWriter(logdir,   sess.graph)
        
        all_saver = tf.train.Saver(max_to_keep=2) 


        begin_t=time.time()
        for i in range(maxstep):     
            print ('\n',"{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
                   
            if i==0 or (i+1)%2000==0:#一次测试
                print ('begining to eval D:')
                ssim_mean, psnr_mean,contexloss,l1loss,localvarloss=gan.eval_once(i)
                
                #自己构建summary
                tsummary = tf.Summary()
                #tsummary.value.add(tag='mean prob of real1', simple_value=prob_T)
                #tsummary.value.add(tag='mean prob of fake1', simple_value=prob_F)
                tsummary.value.add(tag='mean L1_loss of G and GT step1', simple_value=l1loss[0])                
                tsummary.value.add(tag='mean contexloss step1', simple_value=contexloss[0])
                tsummary.value.add(tag='mean localvar loss step1', simple_value=localvarloss[0])
                tsummary.value.add(tag='mean ssim step1', simple_value=ssim_mean[0])
                tsummary.value.add(tag='mean psnr step1', simple_value=psnr_mean[0])
                
                tsummary.value.add(tag='mean L1_loss of G and GT step2', simple_value=l1loss[1])
                tsummary.value.add(tag='mean contexloss step2', simple_value=contexloss[1])
                tsummary.value.add(tag='mean localvar loss step2', simple_value=localvarloss[1])
                tsummary.value.add(tag='mean ssim step2', simple_value=ssim_mean[1])
                tsummary.value.add(tag='mean psnr step2', simple_value=psnr_mean[1])
                
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

        
        
        
        
        
        
        
        
        
        
        
        
        
