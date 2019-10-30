#coding:utf-8
'''
Created on 2019年2月19日

@author: sherl
'''
import tensorflow as tf
import numpy as np
from GAN_tools_common import *


#use depthwise_conv to my_novel_conv
    
def unet_up(inputdata, outchannel,skipcon, scopename,filterlen=3, training=True,withbias=True):
    '''
    Upsampling -->conv(channel/2) --> Leaky ReLU --> concat --> Convolution(channel/2) + Leaky ReLU
    '''
    inputshape=inputdata.get_shape().as_list()

    tep=tf.image.resize_bilinear(inputdata, (inputshape[1]*2, inputshape[2]*2) )
    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv1', stride=1, withbias=withbias)
    #tep=my_batchnorm( tep,training, scopename+'_bn1')
    tep=my_lrelu(tep, scopename, 0.1)

    
    if skipcon is  not None:
        print ('-->concating:',tep, skipcon)
        tshape=skipcon.get_shape().as_list()
        tep=tf.image.resize_bilinear(tep, (tshape[1], tshape[2]) )
            
        tep=tf.concat([tep, skipcon], -1)
    
    #单个conv无法拟合
    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv3', stride=1, withbias=withbias)
    #tep=my_batchnorm( tep,training, scopename+'_bn3')
    tep=my_lrelu(tep, scopename, 0.1)
    
    return tep

def unet_down(inputdata, outchannel, scopename,stride=1, filterlen=3,training=True, withbias=True):
    '''
    downsampling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    '''
    tep=tf.layers.average_pooling2d(inputdata, 2, 2)

    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv1', stride=stride, withbias=withbias)
    #tep=my_batchnorm( tep,training, scopename+'_bn1')
    tep=my_lrelu(tep, scopename, 0.1)
    
    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv2', stride=1, withbias=withbias)
    #tep=my_batchnorm( tep,training, scopename+'_bn2')
    tep=my_lrelu(tep, scopename, 0.1)
    
    return tep
    
def my_unet(inputdata, outChannels, training=True,  withbias=True):
    '''
    layercnt:下降和上升各有几层,原则上应该是一对一
    '''
    tep1=my_conv(inputdata, 7, 32, scopename='unet_start0', stride=1,  withbias=withbias)
    tep1=my_lrelu(tep1, 'unet_start0', 0.1)
    
    tep2=my_conv(tep1, 7, 32, scopename='unet_start1', stride=1,  withbias=withbias)
    tep2=my_lrelu(tep2, 'unet_start1', 0.1)
    
    print (tep2)
    #unet down
    down1=unet_down(tep2, 64, 'unet_down_0', filterlen=5, training=training,withbias=withbias)
    
    down2=unet_down(down1, 128, 'unet_down_1', filterlen=3, training=training,withbias=withbias)
    
    down3=unet_down(down2, 256, 'unet_down_2', filterlen=3, training=training,withbias=withbias)
    
    down4=unet_down(down3, 512, 'unet_down_3', filterlen=3, training=training,withbias=withbias)
    
    down5=unet_down(down4, 512, 'unet_down_4', filterlen=3, training=training,withbias=withbias)
   
   
    #unet up
    tep=unet_up(down5, 512, down4,'unet_up_0', training=training,withbias=withbias)
    
    tep=unet_up(tep, 256, down3,'unet_up_1', training=training,withbias=withbias)
    
    tep=unet_up(tep, 128, down2,'unet_up_2', training=training,withbias=withbias)
    
    tep=unet_up(tep, 64, down1,'unet_up_3', training=training,withbias=withbias)
    
    tep=unet_up(tep, 32, tep2,'unet_up_4', training=training,withbias=withbias)
    
    #final
    tep=my_conv(tep, 3, outChannels, scopename='unet_end0', stride=1, withbias=withbias)
    
    #tep=my_batchnorm( tep,training, 'unet_up_end0_bn2')
    tep=my_lrelu(tep, 'unet_end0_relu', 0.1)
    print (tep)
    
    
    return tep


def my_D_block(inputdata, outchannel, scopename,stride=2, filterlen=3, withbias=True, training=True):
    tep=my_conv(inputdata, filterlen, outchannel, scopename+'_conv1', stride=stride, withbias=withbias)
    
    #tep=my_batchnorm( tep,training, scopename)
    tep=my_lrelu(tep, scopename)
    return tep   


def test_unet():
    imgs_pla = tf.placeholder(datatype, [32, 360/2, 640/2, 3], name='imgs_in')
    
    with tf.variable_scope("G_Net",  reuse=tf.AUTO_REUSE) as scope:
        tep=my_unet(imgs_pla, 4)
        trainvars=tf.trainable_variables()
        
        print ()
        for i in trainvars:
            print (i)


if __name__=='__main__':
    #test_my_find_flip()
    test_unet()
    
                
                
                
                
                
                
                
                