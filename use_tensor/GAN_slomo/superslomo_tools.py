#coding:utf-8
'''
Created on 2019年2月19日

@author: sherl
'''
import tensorflow as tf
import numpy as np
from GAN_tools_common import *


#use depthwise_conv to my_novel_conv
    
def unet_up(inputdata, outchannel,skipcon, scopename,filterlen=3, training=True,withbias=True, withbn=False):
    '''
    Upsampling -->conv(channel/2) --> Leaky ReLU --> concat --> Convolution(channel/2) + Leaky ReLU
    '''
    inputshape=inputdata.get_shape().as_list()

    tep=tf.image.resize_bilinear(inputdata, (inputshape[1]*2, inputshape[2]*2) )
    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv1', stride=1, withbias=withbias)
    if withbn: tep=my_batchnorm( tep,training, scopename+'_bn1')
    tep=my_lrelu(tep, scopename, 0.1)

    
    if skipcon is  not None:
        print ('-->concating:',tep, skipcon)
        tshape=skipcon.get_shape().as_list()
        tep=tf.image.resize_bilinear(tep, (tshape[1], tshape[2]) )
            
        tep=tf.concat([tep, skipcon], -1)
    
    #单个conv无法拟合
    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv3', stride=1, withbias=withbias)
    if withbn: tep=my_batchnorm( tep,training, scopename+'_bn3')
    tep=my_lrelu(tep, scopename, 0.1)
    
    return tep

def unet_down(inputdata, outchannel, scopename,stride=1, filterlen=3,training=True, withbias=True, withbn=False):
    '''
    downsampling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    '''
    tep=tf.layers.average_pooling2d(inputdata, 2, 2)

    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv1', stride=stride, withbias=withbias)
    if withbn: tep=my_batchnorm( tep,training, scopename+'_bn1')
    tep=my_lrelu(tep, scopename, 0.1)
    
    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv2', stride=1, withbias=withbias)
    if withbn: tep=my_batchnorm( tep,training, scopename+'_bn2')
    tep=my_lrelu(tep, scopename, 0.1)
    
    return tep
    
def my_unet(inputdata, outChannels, training=True,  withbias=True, withbn=False):
    '''
    layercnt:下降和上升各有几层,原则上应该是一对一
    '''
    tep1=my_conv(inputdata, 7, 32, scopename='unet_start0', stride=1,  withbias=withbias)
    tep1=my_lrelu(tep1, 'unet_start0', 0.1)
    
    tep2=my_conv(tep1, 7, 32, scopename='unet_start1', stride=1,  withbias=withbias)
    tep2=my_lrelu(tep2, 'unet_start1', 0.1)
    
    print (tep2)
    #unet down
    down1=unet_down(tep2, 64, 'unet_down_0', filterlen=5, training=training,withbias=withbias, withbn=withbn)
    
    down2=unet_down(down1, 128, 'unet_down_1', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    down3=unet_down(down2, 256, 'unet_down_2', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    down4=unet_down(down3, 512, 'unet_down_3', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    down5=unet_down(down4, 512, 'unet_down_4', filterlen=3, training=training,withbias=withbias, withbn=withbn)
   
   
    #unet up
    tep=unet_up(down5, 512, down4,'unet_up_0', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 256, down3,'unet_up_1', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 128, down2,'unet_up_2', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 64, down1,'unet_up_3', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 32, tep2,'unet_up_4', training=training,withbias=withbias, withbn=withbn)
    
    #final
    tep=my_conv(tep, 3, outChannels, scopename='unet_end0', stride=1, withbias=withbias)
    
    if withbn: tep=my_batchnorm( tep,training, 'unet_up_end0_bn2')
    tep=my_lrelu(tep, 'unet_end0_relu', 0.1)
    print (tep)
    
    
    return tep

def my_unet_split(inputdata, outChannels, training=True,  withbias=True, withbn=False):
    '''
    layercnt:下降和上升各有几层,原则上应该是一对一
    inputdata:[b, h, w, 6]
    '''
    #print ("my_unet_split:",inputdata)  #Tensor("first_unet/concat:0", shape=(10, 180, 320, 6), dtype=float32)
    frame0=inputdata[:, :, :, :3]
    frame2=inputdata[:, :, :, 3:]
    
    #############################################################################################frame0 down
    frame0_tep1=my_conv(frame0, 7, 16, scopename='unet_start0', stride=1,  withbias=withbias)
    frame0_tep1=my_lrelu(frame0_tep1, 'unet_start0', 0.1)
    
    frame0_tep2=my_conv(frame0_tep1, 7, 16, scopename='unet_start1', stride=1,  withbias=withbias)
    frame0_tep2=my_lrelu(frame0_tep2, 'unet_start1', 0.1)
    
    print (frame0_tep2)
    #unet down
    frame0_down1=unet_down(frame0_tep2, 32, 'unet_down_0', filterlen=5, training=training,withbias=withbias, withbn=withbn)
    
    frame0_down2=unet_down(frame0_down1, 64, 'unet_down_1', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame0_down3=unet_down(frame0_down2, 128, 'unet_down_2', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame0_down4=unet_down(frame0_down3, 256, 'unet_down_3', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame0_down5=unet_down(frame0_down4, 256, 'unet_down_4', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    
    #############################################################################################frame2 down 共享权重
    frame2_tep1=my_conv(frame2, 7, 16, scopename='unet_start0', stride=1,  withbias=withbias)
    frame2_tep1=my_lrelu(frame2_tep1, 'unet_start0', 0.1)
    
    frame2_tep2=my_conv(frame2_tep1, 7, 16, scopename='unet_start1', stride=1,  withbias=withbias)
    frame2_tep2=my_lrelu(frame2_tep2, 'unet_start1', 0.1)
    
    print (frame2_tep2)
    #unet down
    frame2_down1=unet_down(frame2_tep2, 32, 'unet_down_0', filterlen=5, training=training,withbias=withbias, withbn=withbn)
    
    frame2_down2=unet_down(frame2_down1, 64, 'unet_down_1', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame2_down3=unet_down(frame2_down2, 128, 'unet_down_2', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame2_down4=unet_down(frame2_down3, 256, 'unet_down_3', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame2_down5=unet_down(frame2_down4, 256, 'unet_down_4', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
   
    
    ##############################################################################################unet up
    tep=unet_up(tf.concat([frame0_down5, frame2_down5], -1), 512, tf.concat([frame0_down4, frame2_down4], -1),'unet_up_0', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 256, tf.concat([frame0_down3, frame2_down3], -1),'unet_up_1', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 128, tf.concat([frame0_down2, frame2_down2], -1),'unet_up_2', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 64, tf.concat([frame0_down1, frame2_down1], -1),'unet_up_3', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 32, tf.concat([frame0_tep2, frame2_tep2], -1),'unet_up_4', training=training,withbias=withbias, withbn=withbn)
    
    #final
    tep=my_conv(tep, 3, outChannels*2, scopename='unet_end0', stride=1, withbias=withbias)
    
    if withbn: tep=my_batchnorm( tep,training, 'unet_up_end0_bn2')
    tep=my_lrelu(tep, 'unet_end0_relu', 0.1)
    print (tep)
    
    #final2
    tep=my_conv(tep, 3, outChannels, scopename='unet_end1', stride=1, withbias=withbias)
    
    if withbn: tep=my_batchnorm( tep,training, 'unet_up_end1_bn2')
    tep=my_lrelu(tep, 'unet_end1_relu', 0.1)
    print (tep)
    
    
    return tep


def my_unet_part_split(inputdata, outChannels, training=True,  withbias=True, withbn=False):
    '''
    layercnt:下降和上升各有几层,原则上应该是一对一
    inputdata:[b, h, w, 6]
    '''
    #print ("my_unet_split:",inputdata)  #Tensor("first_unet/concat:0", shape=(10, 180, 320, 6), dtype=float32)
    frame0=inputdata[:, :, :, :3]
    frame2=inputdata[:, :, :, 3:]
    
    #############################################################################################frame0 down
    frame0_tep1=my_conv(frame0, 7, 16, scopename='unet_start0', stride=1,  withbias=withbias)
    frame0_tep1=my_lrelu(frame0_tep1, 'unet_start0', 0.1)
    
    frame0_tep2=my_conv(frame0_tep1, 7, 16, scopename='unet_start1', stride=1,  withbias=withbias)
    frame0_tep2=my_lrelu(frame0_tep2, 'unet_start1', 0.1)
    
    print (frame0_tep2)
    #unet down
    frame0_down1=unet_down(frame0_tep2, 32, 'unet_down_0', filterlen=5, training=training,withbias=withbias, withbn=withbn)
    
    frame0_down2=unet_down(frame0_down1, 64, 'unet_down_1', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame0_down3=unet_down(frame0_down2, 128, 'unet_down_2', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    #frame0_down4=unet_down(frame0_down3, 256, 'unet_down_3', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    #frame0_down5=unet_down(frame0_down4, 256, 'unet_down_4', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    
    #############################################################################################frame2 down 共享权重
    frame2_tep1=my_conv(frame2, 7, 16, scopename='unet_start0', stride=1,  withbias=withbias)
    frame2_tep1=my_lrelu(frame2_tep1, 'unet_start0', 0.1)
    
    frame2_tep2=my_conv(frame2_tep1, 7, 16, scopename='unet_start1', stride=1,  withbias=withbias)
    frame2_tep2=my_lrelu(frame2_tep2, 'unet_start1', 0.1)
    
    print (frame2_tep2)
    #unet down
    frame2_down1=unet_down(frame2_tep2, 32, 'unet_down_0', filterlen=5, training=training,withbias=withbias, withbn=withbn)
    
    frame2_down2=unet_down(frame2_down1, 64, 'unet_down_1', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame2_down3=unet_down(frame2_down2, 128, 'unet_down_2', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    #frame2_down4=unet_down(frame2_down3, 256, 'unet_down_3', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    #frame2_down5=unet_down(frame2_down4, 256, 'unet_down_4', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    #############################################################################################unet_down common area
    frame_down4 =unet_down( tf.concat([frame0_down3, frame2_down3],-1) , 512, 'unet_down_3', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    frame_down5 =unet_down(frame_down4, 512, 'unet_down_4', filterlen=3, training=training,withbias=withbias, withbn=withbn)
   
    
    ##############################################################################################unet up
    tep=unet_up(frame_down5, 512, frame_down4,'unet_up_0', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 256, tf.concat([frame0_down3, frame2_down3], -1),'unet_up_1', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 128, tf.concat([frame0_down2, frame2_down2], -1),'unet_up_2', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 64, tf.concat([frame0_down1, frame2_down1], -1),'unet_up_3', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 32, tf.concat([frame0_tep2, frame2_tep2], -1),'unet_up_4', training=training,withbias=withbias, withbn=withbn)
    
    #final
    tep=my_conv(tep, 3, outChannels*2, scopename='unet_end0', stride=1, withbias=withbias)
    
    if withbn: tep=my_batchnorm( tep,training, 'unet_up_end0_bn2')
    tep=my_lrelu(tep, 'unet_end0_relu', 0.1)
    print (tep)
    
    #final2
    tep=my_conv(tep, 3, outChannels, scopename='unet_end1', stride=1, withbias=withbias)
    
    if withbn: tep=my_batchnorm( tep,training, 'unet_up_end1_bn2')
    tep=my_lrelu(tep, 'unet_end1_relu', 0.1)
    print (tep)
    
    
    return tep

def my_unet_split_with_flipconv(inputdata, outChannels, training=True,  withbias=True, withbn=False):
    '''
    layercnt:下降和上升各有几层,原则上应该是一对一
    inputdata:[b, h, w, 6]
    '''
    #print ("my_unet_split:",inputdata)  #Tensor("first_unet/concat:0", shape=(10, 180, 320, 6), dtype=float32)
    frame0=inputdata[:, :, :, :3]
    frame2=inputdata[:, :, :, 3:]
    
    #############################################################################################frame0 down
    frame0_tep1=my_conv(frame0, 7, 16, scopename='unet_start0', stride=1,  withbias=withbias)
    frame0_tep1=my_lrelu(frame0_tep1, 'unet_start0', 0.1)
    
    frame0_tep2=my_conv(frame0_tep1, 7, 16, scopename='unet_start1', stride=1,  withbias=withbias)
    frame0_tep2=my_lrelu(frame0_tep2, 'unet_start1', 0.1)
    
    print (frame0_tep2)
    #unet down
    frame0_down1=unet_down(frame0_tep2, 32, 'unet_down_0', filterlen=5, training=training,withbias=withbias, withbn=withbn)
    
    frame0_down2=unet_down(frame0_down1, 64, 'unet_down_1', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame0_down3=unet_down(frame0_down2, 128, 'unet_down_2', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame0_down4=unet_down(frame0_down3, 256, 'unet_down_3', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame0_down5=unet_down(frame0_down4, 256, 'unet_down_4', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    
    #############################################################################################frame2 down 共享权重
    frame2_tep1=my_conv(frame2, 7, 16, scopename='unet_start0', stride=1,  withbias=withbias)
    frame2_tep1=my_lrelu(frame2_tep1, 'unet_start0', 0.1)
    
    frame2_tep2=my_conv(frame2_tep1, 7, 16, scopename='unet_start1', stride=1,  withbias=withbias)
    frame2_tep2=my_lrelu(frame2_tep2, 'unet_start1', 0.1)
    
    print (frame2_tep2)
    #unet down
    frame2_down1=unet_down(frame2_tep2, 32, 'unet_down_0', filterlen=5, training=training,withbias=withbias, withbn=withbn)
    
    frame2_down2=unet_down(frame2_down1, 64, 'unet_down_1', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame2_down3=unet_down(frame2_down2, 128, 'unet_down_2', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame2_down4=unet_down(frame2_down3, 256, 'unet_down_3', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
    frame2_down5=unet_down(frame2_down4, 256, 'unet_down_4', filterlen=3, training=training,withbias=withbias, withbn=withbn)
    
   
    
    ##############################################################################################unet up
    tep=unet_up(  my_novel_conv_withweight(frame0_down5, frame2_down5,  'unet_up_0_flip'), 512, \
                  my_novel_conv_withweight(frame0_down4, frame2_down4,'unet_up_0_flip_1'),  'unet_up_0'  , training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 256, my_novel_conv_withweight(frame0_down3, frame2_down3,  'unet_up_1_flip'),'unet_up_1', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 128, my_novel_conv_withweight(frame0_down2, frame2_down2,  'unet_up_2_flip'),'unet_up_2', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 64,  my_novel_conv_withweight(frame0_down1, frame2_down1,  'unet_up_3_flip'),'unet_up_3', training=training,withbias=withbias, withbn=withbn)
    
    tep=unet_up(tep, 32,  my_novel_conv_withweight(frame0_tep2,  frame2_tep2,   'unet_up_4_flip'),'unet_up_4', training=training,withbias=withbias, withbn=withbn)
    
    #final
    tep=my_conv(tep, 3, outChannels*2, scopename='unet_end0', stride=1, withbias=withbias)
    
    if withbn: tep=my_batchnorm( tep,training, 'unet_up_end0_bn2')
    tep=my_lrelu(tep, 'unet_end0_relu', 0.1)
    print (tep)
    
    #final2
    tep=my_conv(tep, 3, outChannels, scopename='unet_end1', stride=1, withbias=withbias)
    
    if withbn: tep=my_batchnorm( tep,training, 'unet_up_end1_bn2')
    tep=my_lrelu(tep, 'unet_end1_relu', 0.1)
    print (tep)
    
    
    return tep

def my_novel_conv_withweight(inputdata, inputdata2, scopename,filterlen=3, outchannel=None, stride=1, padding="SAME", reuse=tf.AUTO_REUSE, withbias=False, training=True):
    '''
    stride:这里代表希望将输出大小变为原图的   1/stride (注意同deconv区分)
    '''
    inputshape=inputdata.get_shape().as_list()
    if not outchannel: outchannel= 1 #inputshape[-1] #如果未定义，就等于输入channel
    
    with tf.variable_scope(scopename,  reuse=reuse) as scope: 
        kernel=tf.get_variable('weights', [outchannel, filterlen,filterlen, inputshape[-1]], dtype=datatype, \
                               initializer=tf.random_normal_initializer(stddev=stddev))
        #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
        #但是这个为了进行反转，特意这么设置，后面送进去卷积前要transpose
        tep_kernel=tf.transpose(kernel, [1,2,3,0])
        print ('tep_kernel:',tep_kernel)
        ori_cnn=tf.nn.depthwise_conv2d(inputdata, tep_kernel, strides=[1,stride,stride,1], padding=padding)
                    
        
        #left2right
        tep_kernel=tf.image.flip_left_right(kernel)
        tep_kernel=tf.transpose(tep_kernel, [1,2,3,0])
        print ('tep_kernel:',tep_kernel)
        left_cnn=tf.nn.depthwise_conv2d(inputdata2, tep_kernel, strides=[1,stride,stride,1], padding=padding)
                
        #up 2 down
        tep_kernel=tf.image.flip_up_down(kernel)
        tep_kernel=tf.transpose(tep_kernel, [1,2,3,0])
        print ('tep_kernel:',tep_kernel)
        up_cnn=tf.nn.depthwise_conv2d(inputdata2, tep_kernel, strides=[1,stride,stride,1], padding=padding)
                
        #这里需要一个操作来集合这3个
        one_channel=tf.abs(ori_cnn-left_cnn)
        ano_channel=tf.abs(ori_cnn-up_cnn)
        
        
        if withbias:
            bias=tf.get_variable('bias', [inputshape[-1]], dtype=datatype, initializer=tf.constant_initializer(bias_init))
            one_channel=tf.nn.bias_add(one_channel, bias)
            ano_channel=tf.nn.bias_add(ano_channel, bias)
            
            
        #return tf.concat( [one_channel,ano_channel], -1)
        flip_resu=one_channel+ano_channel
        return tf.concat([inputdata, flip_resu, inputdata2])



def my_D_block(inputdata, outchannel, scopename,stride=2, filterlen=3, withbias=True, training=True, withbn=False):
    tep=my_conv(inputdata, filterlen, outchannel, scopename+'_conv1', stride=stride, withbias=withbias)
    
    if withbn: tep=my_batchnorm( tep,training, scopename)
    tep=my_lrelu(tep, scopename)
    return tep   

def Discriminator_net( imgs, name="D_net", withbias=True, training=True, withbn=False):
        '''
        :用于判定生成视频帧的真实性的D
        imgs:input imgs [batchsize, h, w, ..]
        '''
        #这里输入的imgs应该是tanh后的，位于-1~1之间
        #cast to float    
        with tf.variable_scope(name,  reuse=tf.AUTO_REUSE) as scopevar:
            scope=scopevar.name
            #0
            tep=my_D_block(imgs, 32, scope+'_Dblock_0',stride=2, filterlen=5 ,withbias=withbias, training=training,  withbn=withbn)
            print (tep)
            
            #1
            tep=my_D_block(tep, 64, scope+'_Dblock_1',stride=2, filterlen=4 ,withbias=withbias, training=training,  withbn=withbn)
            print (tep)
            
            #2
            tep=my_D_block(tep, 64, scope+'_Dblock_2',stride=2, filterlen=3 ,withbias=withbias, training=training,  withbn=withbn)
            print (tep)
            
            #3
            tep=my_D_block(tep, 64, scope+'_Dblock_3',stride=2, filterlen=3 ,withbias=withbias, training=training,  withbn=withbn)
            print (tep)
            
            #4
            tep=my_D_block(tep, 64, scope+'_Dblock_4',stride=2, filterlen=3 ,withbias=withbias, training=training,  withbn=withbn)
            print (tep)#Tensor("D_net/D_net_Dblock_4/LeakyRelu:0", shape=(32, 6, 10, 64), dtype=float32)
            
            #######################################################################################################################################
            #fc
            tep=my_fc(tep, 1024, scope+'_fc1',  withbias=withbias)
            tep=my_lrelu(tep, scope)
            
            tep=my_dropout(tep, tf.convert_to_tensor(training, dtype=tf.bool), 0.5)
            
            tep=my_fc(tep, 1, scope+'_fc2',  withbias=withbias)
            
            
            #sigmoid
            D_2_sigmoid=tf.nn.sigmoid(tep, name=scope+'_D_sigmoid')
            
        return D_2_sigmoid, tep



def my_novel_conv_withweight(inputdata, inputdata2, filterlen,    scopename, outchannel=None, stride=1, padding="SAME", reuse=tf.AUTO_REUSE, withbias=False, training=True):
    '''
    stride:这里代表希望将输出大小变为原图的   1/stride (注意同deconv区分)
    '''
    inputshape=inputdata.get_shape().as_list()
    if not outchannel: outchannel= 1 #inputshape[-1] #如果未定义，就等于输入channel
    
    with tf.variable_scope(scopename,  reuse=reuse) as scope: 
        kernel=tf.get_variable('weights', [outchannel, filterlen,filterlen, inputshape[-1]], dtype=datatype, \
                               initializer=tf.random_normal_initializer(stddev=stddev))
        #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
        #但是这个为了进行反转，特意这么设置，后面送进去卷积前要transpose
        tep_kernel=tf.transpose(kernel, [1,2,3,0])
        print ('tep_kernel:',tep_kernel)
        #ori_cnn=tf.nn.depthwise_conv2d(inputdata, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        ori_cnn=tf.nn.conv2d(inputdata, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        
        #left2right
        tep_kernel=tf.image.flip_left_right(kernel)
        tep_kernel=tf.transpose(tep_kernel, [1,2,3,0])
        print ('tep_kernel:',tep_kernel)
        #left_cnn=tf.nn.depthwise_conv2d(inputdata2, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        left_cnn=tf.nn.conv2d(inputdata2, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        
        #up 2 down
        tep_kernel=tf.image.flip_up_down(kernel)
        tep_kernel=tf.transpose(tep_kernel, [1,2,3,0])
        print ('tep_kernel:',tep_kernel)
        #up_cnn=tf.nn.depthwise_conv2d(inputdata2, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        up_cnn=tf.nn.conv2d(inputdata2, tep_kernel, strides=[1,stride,stride,1], padding=padding)
                
        #这里需要一个操作来集合这3个
        one_channel=my_lrelu(ori_cnn-left_cnn)#tf.abs(ori_cnn-left_cnn)
        ano_channel=my_lrelu(ori_cnn-up_cnn)  #tf.abs(ori_cnn-up_cnn)
        
        
        if withbias:
            bias=tf.get_variable('bias', [inputshape[-1]], dtype=datatype, initializer=tf.constant_initializer(bias_init))
            one_channel=tf.nn.bias_add(one_channel, bias)
            ano_channel=tf.nn.bias_add(ano_channel, bias)
            
            
        return tf.concat( [one_channel,ano_channel], -1)



def my_novel_conv(inputdata, inputdata2, filterlen,    scopename, outchannel=None, stride=1, padding="SAME", reuse=tf.AUTO_REUSE, withbias=False, training=True):
    '''
    stride:这里代表希望将输出大小变为原图的   1/stride (注意同deconv区分)
    '''
    inputshape=inputdata.get_shape().as_list()
    if not outchannel: outchannel= 1 #inputshape[-1] #如果未定义，就等于输入channel
    
    conv_kernel_left=np.zeros([outchannel, filterlen,filterlen, inputshape[-1]])
    conv_kernel_left[:, :, :int(filterlen/2), :]=1
    conv_kernel_left_cnt=np.count_nonzero(conv_kernel_left)/inputshape[-1]
    
    #print (conv_kernel_left[0, :, :,0])
    
    conv_kernel_up=np.zeros([outchannel, filterlen,filterlen, inputshape[-1]])
    conv_kernel_up[:, :int(filterlen/2), :,  :]=1
    conv_kernel_up_cnt=np.count_nonzero(conv_kernel_up)/inputshape[-1]
    
    conv_kernel_180=np.zeros([outchannel, filterlen,filterlen, inputshape[-1]])
    conv_kernel_180[:, :int(filterlen/2), :int(filterlen/2),  :]=1
    conv_kernel_180_cnt=np.count_nonzero(conv_kernel_180)/inputshape[-1]
    
    
    with tf.variable_scope(scopename,  reuse=reuse) as scope: 
        #inputdata =my_batchnorm(inputdata , training, 'batchnorm_input1')
        #inputdata2=my_batchnorm(inputdata2, training, 'batchnorm_input2')
        
        kernel_left=tf.Variable(conv_kernel_left, trainable=False, dtype=tf.float32)
        kernel_up=tf.Variable(conv_kernel_up, trainable=False, dtype=tf.float32)
        kernel_180=tf.Variable(conv_kernel_180, trainable=False, dtype=tf.float32)
        #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
        #但是这个为了进行反转，特意这么设置，后面送进去卷积前要transpose
        
        #left_right
        tep_kernel=kernel_left
        tep_kernel=tf.transpose(tep_kernel, [1,2,3,0])
        #print ('tep_kernel:',tep_kernel)
        cnn_ori_left=tf.nn.depthwise_conv2d(inputdata, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        cnn_ori_left2=tf.nn.depthwise_conv2d(inputdata2, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        #!!!!!!!!!!!!!!!!1
        tep_kernel=tf.image.flip_left_right(kernel_left)
        tep_kernel=tf.transpose(tep_kernel, [1,2,3,0])
        #print ('tep_kernel:',tep_kernel)
        cnn_left=tf.nn.depthwise_conv2d(inputdata2, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        cnn_left2=tf.nn.depthwise_conv2d(inputdata, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        
        cnn_left_abs=tf.abs(cnn_ori_left-cnn_left) #[n,h,w,c]
        cnn_left_abs2=tf.abs(cnn_ori_left2-cnn_left2)
        
        cnn_left_arg_min=tf.argmin( tf.stack([cnn_left_abs, cnn_left_abs2], 4) , 4)
        min_datas_left=tf.where( tf.equal(cnn_left_arg_min,0) , cnn_left, cnn_ori_left2)/tf.cast(conv_kernel_left_cnt, tf.float32)
        #min_datas_left=tf.where( tf.equal(cnn_left_arg_min,0) , inputdata, inputdata2)
        
        cnn_left_min=tf.minimum(cnn_left_abs, cnn_left_abs2)  #/tf.cast(conv_kernel_left_cnt, tf.float32)
        #cnn_left_final=cnn_left_min*inputdata+(1-cnn_left_min)*min_datas_left
        
        #up_down!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #up_down
        tep_kernel=kernel_up
        tep_kernel=tf.transpose(tep_kernel, [1,2,3,0])
        #print ('tep_kernel:',tep_kernel)
        cnn_ori_up=tf.nn.depthwise_conv2d(inputdata, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        cnn_ori_up2=tf.nn.depthwise_conv2d(inputdata2, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        #!!!!!!!!!!!!!!!!1
        tep_kernel=tf.image.flip_up_down(kernel_up)
        tep_kernel=tf.transpose(tep_kernel, [1,2,3,0])
        #print ('tep_kernel:',tep_kernel)
        cnn_up=tf.nn.depthwise_conv2d(inputdata2, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        cnn_up2=tf.nn.depthwise_conv2d(inputdata, tep_kernel, strides=[1,stride,stride,1], padding=padding)
        
        cnn_up_abs=tf.abs(cnn_ori_up-cnn_up) #[n,h,w,c]
        cnn_up_abs2=tf.abs(cnn_ori_up2-cnn_up2)
        
        cnn_up_arg_min=tf.argmin( tf.stack([cnn_up_abs, cnn_up_abs2], 4) , 4)
        min_datas_up=tf.where( tf.equal(cnn_up_arg_min,0) , cnn_up, cnn_ori_up2)/tf.cast(conv_kernel_up_cnt, tf.float32)
        #min_datas_up=tf.where( tf.equal(cnn_up_arg_min,0) , inputdata, inputdata2)
        
        cnn_up_min=tf.minimum(cnn_up_abs, cnn_up_abs2)  #/tf.cast(conv_kernel_up_cnt, tf.float32)
        #cnn_up_final=cnn_up_min*inputdata+(1-cnn_up_min)*min_datas_up
        
        
        
        
        #merge
        all_arg_min=tf.argmin( tf.stack([cnn_left_min, cnn_up_min], 4) , 4)
        all_final=tf.where( tf.equal(all_arg_min,0) , min_datas_left, min_datas_up)       
        
        
        if withbias:
            bias=tf.get_variable('bias', [inputshape[-1]], dtype=datatype, initializer=tf.constant_initializer(bias_init))
            all_final=tf.nn.bias_add(all_final, bias)

            
            
        return all_final
    

'''
tf.contrib.image.dense_image_warp(
    image,
    flow,
    name='dense_image_warp'
)
pixel value at output[b, j, i, c] is
  images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].
'''


def my_novel_unet(inputdata,inputdata2, layercnt=3, outchannel=2,  filterlen=3,training=True,  withbias=True):
    '''
    这里将两个输入图片通过同一个特征网络，并保留中间各自特征
    inputdata/inutdata2:frame0/frame1
    '''
    flipconv_method=my_novel_conv_withweight
    inputshape=inputdata.get_shape().as_list()
    channel_init=inputshape[-1]  #3
    skipcon1=[]
    skipcon2=[]
    basechannelinit=6
    ##########################################################################
    #first unet-down input1:the first frame
    tep=my_conv(inputdata , filterlen+2*int(layercnt-1), channel_init*basechannelinit, scopename='unet_down_start', stride=1,  withbias=withbias)
    #tep=my_batchnorm( tep,training, 'trace_1_unet_down_start_bn')
    tep=my_lrelu(tep, 'trace_1_unet_down_start_relu1')
    tep=my_conv(tep , filterlen+2*int(layercnt-1), channel_init*basechannelinit, scopename='unet_down_start2', stride=1,  withbias=withbias)
    tep=my_lrelu(tep, 'trace_1_unet_down_start_relu2')
    print (tep)
    
    print ('\nforming UNET-->layer1:',layercnt)
    
    for i in range(layercnt):
        skipcon1.append(tep)
        tep=unet_down(tep, channel_init*basechannelinit*( 2**(min(layercnt-1 ,i+1))), 'unet_down_'+str(i), stride=2, filterlen=filterlen+2*int( (layercnt-i)-1 ), training=training,withbias=withbias)
        print (tep)
        
    input1_fea=tep
    
    #######################################################################   
    #then unet-down the second frame 
    tep=my_conv(inputdata2, filterlen+2*int(layercnt-1), channel_init*basechannelinit, scopename='unet_down_start', stride=1,  withbias=withbias)
    #tep=my_batchnorm( tep,training, 'trace_2_unet_down_start_bn')
    tep=my_lrelu(tep, 'trace_2_unet_down_start_relu1')
    tep=my_conv(tep , filterlen+2*int(layercnt-1), channel_init*basechannelinit, scopename='unet_down_start2', stride=1,  withbias=withbias)
    tep=my_lrelu(tep, 'trace_2_unet_down_start_relu2') 
    print (tep)
    
    print ('\nforming UNET-->layer2:',layercnt)
    
    for i in range(layercnt):
        skipcon2.append(tep)
        tep=unet_down(tep, channel_init*basechannelinit*( 2**(min(layercnt-1 ,i+1))), 'unet_down_'+str(i), stride=2, filterlen=filterlen+2*int( (layercnt-i)-1 ), training=training,withbias=withbias)
        print (tep)
        
    input2_fea=tep
    
    ##################连接两个部分
    #concating two middle feature
    tep=flipconv_method(input1_fea, input2_fea, filterlen, 'middle_novel_cnn', training=training)
    tep=tf.concat([input1_fea, tep, input2_fea], -1)
    print (tep)
    
    ######################################################up
    #begining unet-up 
    for i in reversed(range(layercnt)):
        tep1=skipcon1[i]
        tep2=skipcon2[i]
        
        skipcon=flipconv_method(tep1, tep2, filterlen+int( 2*(layercnt-i-1) ), 'unet_up_novel_cnn_'+str(i),  training=training)
        skipcon=tf.concat([tep1, skipcon, tep2], -1) #!!!!!
        tep=unet_up(tep, channel_init*basechannelinit*( 2**(min(layercnt-1 ,i+1))), skipcon,'unet_up_'+str(i), stride=2,  filterlen=filterlen+int( 2*(layercnt-i-1) ),  training=training,withbias=withbias)
        print (tep)
        
    #finish the net 
    tep=my_conv(tep, filterlen+2*int(layercnt-1), channel_init*basechannelinit, scopename='unet_up_final_1', stride=1, withbias=withbias)
    #tep=my_batchnorm( tep,training, 'unet_up_final_1_bn')
    tep=my_lrelu(tep, 'unet_up_final_1_relu') 
    print (tep)
    
    #finally channel to 3
    tep=my_conv(tep, filterlen+2*int(layercnt-1), outchannel, scopename='unet_up_final_2', stride=1, withbias=withbias)
    tep=my_lrelu(tep, 'unet_up_final_2_relu')
    print (tep)
    tep=tf.image.resize_images(tep, [inputshape[1],inputshape[2]], method=tf.image.ResizeMethod.BILINEAR)
    
    return tep
    


def test_unet():
    imgs_pla = tf.placeholder(datatype, [32, 360/2, 640/2, 3], name='imgs_in')
    
    with tf.variable_scope("G_Net",  reuse=tf.AUTO_REUSE) as scope:
        tep=my_unet(imgs_pla, 4)
    tep=Discriminator_net(tep)
    trainvars=tf.trainable_variables()
        
    G_para=[i for i in trainvars if i.name.startswith('G')]
    D_para=[i for i in trainvars if i.name.startswith('D')]
    print (len(trainvars),len(G_para),len(D_para))
        
    for i in trainvars:  print (i)


if __name__=='__main__':
    #test_my_find_flip()
    test_unet()
    
                
                
                
                
                
                
                
                