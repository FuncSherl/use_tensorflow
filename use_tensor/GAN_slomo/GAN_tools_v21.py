#coding:utf-8
'''
Created on 2019年2月19日

@author: sherl
'''
import tensorflow as tf
import numpy as np


#use depthwise_conv to my_novel_conv

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dropout=0.5 
leakyrelurate=0.2
stddev=0.01
bias_init=0.0

datatype=tf.float32

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def my_batchnorm( data,training, scopename):
    return tf.contrib.layers.batch_norm(data,
                                        center=True, #如果为True，有beta偏移量；如果为False，无beta偏移量
                                        decay=0.9,#衰减系数,即有一个moving_mean和一个当前batch的mean，更新moving_mean=moving_mean*decay+(1-decay)*mean
                                            #合适的衰减系数值接近1.0,特别是含多个9的值：0.999,0.99,0.9。如果训练集表现很好而验证/测试集表现得不好，选择小的系数（推荐使用0.9）
                                        updates_collections=None,#其变量默认是tf.GraphKeys.UPDATE_OPS，在训练时提供了一种内置的均值和方差更新机制，
                                            #即通过图中的tf.Graphs.UPDATE_OPS变量来更新，但它是在每次当前批次训练完成后才更新均值和方差，这样就导致当前数据总是使用前一次的均值和方差，
                                            #没有得到最新的更新。所以一般都会将其设置为None，让均值和方差即时更新。这样虽然相比默认值在性能上稍慢点，但是对模型的训练还是有很大帮助的
                                        epsilon=1e-5, #防止除0
                                        scale=True, #如果为True，则乘以gamma。如果为False，gamma则不使用。当下一层是线性的时（例如nn.relu），由于缩放可以由下一层完成,可不要
                                        #reuse=tf.AUTO_REUSE,  #reuse的默认选项是None,此时会继承父scope的reuse标志
                                        #param_initializers=None, # beta, gamma, moving mean and moving variance的优化初始化
                                        #activation_fn=None, #用于激活，默认为线性激活函数
                                        #param_regularizers=None,# beta and gamma正则化优化
                                        #data_format=DATA_FORMAT_NHWC,
                                        is_training=training, # 图层是否处于训练模式。
                                        scope=scopename)
        
def my_deconv(inputdata, filterlen, outchannel,   socpename, stride=2, padding="SAME", reuse=tf.AUTO_REUSE, withbias=True):
    '''
    stride:想将输出图像扩充为原来几倍？ 
    '''
    inputshape=inputdata.get_shape().as_list()
    
    with tf.variable_scope(socpename,  reuse=reuse) as scope:  
        kernel=tf.get_variable('weights', [filterlen,filterlen, outchannel, inputshape[-1]], dtype=datatype,\
                                initializer=tf.random_normal_initializer(stddev=stddev))
        #而tf.nn.conv2d_transpose中的filter参数，是[filter_height, filter_width, out_channels，in_channels]的形式
        ret=tf.nn.conv2d_transpose(inputdata, kernel, \
                                   output_shape=[inputshape[0], int(stride*inputshape[1]), int(stride*inputshape[2]), outchannel], \
                                   strides=[1,stride,stride,1], padding=padding)
            
        if withbias:
            bias=tf.get_variable('bias', [outchannel], dtype=datatype, initializer=tf.constant_initializer(bias_init))
            ret=tf.nn.bias_add(ret, bias)
        return ret
    
    
def my_conv(inputdata, filterlen, outchannel,   scopename, stride=2, padding="SAME", reuse=tf.AUTO_REUSE, withbias=True):
    '''
    stride:这里代表希望将输出大小变为原图的   1/stride (注意同deconv区分)
    '''
    inputshape=inputdata.get_shape().as_list()
    with tf.variable_scope(scopename,  reuse=reuse) as scope: 
        kernel=tf.get_variable('weights', [filterlen,filterlen, inputshape[-1], outchannel], dtype=datatype, \
                               initializer=tf.random_normal_initializer(stddev=stddev))
        #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
                
        ret=tf.nn.conv2d(inputdata, kernel, strides=[1,stride,stride,1], padding=padding)
                
        if withbias:
            bias=tf.get_variable('bias', [outchannel], dtype=datatype, initializer=tf.constant_initializer(bias_init))
            ret=tf.nn.bias_add(ret, bias)
        return ret
    
def my_lrelu(inputdata, scopename='default'):
    with tf.variable_scope(scopename) as scope:
        return tf.nn.leaky_relu(inputdata, leakyrelurate)
    
def my_dropout(inputdata, training, rate=0.5):
    #dropout??
    return tf.cond(training, lambda: tf.nn.dropout(inputdata, rate), lambda: inputdata)

    
def my_fc(inputdata,  outchannel,   scopename,  reuse=tf.AUTO_REUSE, withbias=True):
    inputshape=inputdata.get_shape().as_list()
    #flatten
    tep=tf.reshape(inputdata, [inputshape[0], -1])
    
    #fc
    with tf.variable_scope(scopename,  reuse=reuse) as scope: 
        weight = tf.get_variable('weights', [tep.get_shape()[-1], outchannel], dtype=datatype, \
                                 initializer=tf.random_normal_initializer(stddev=stddev))
        tep=tf.matmul(tep, weight)
        if withbias:
            bias = tf.get_variable('bias', [outchannel], dtype=datatype, initializer=tf.constant_initializer(bias_init))
            tep = tf.nn.bias_add(tep, bias)
        return tep
    
    
def unet_up(inputdata, outchannel,skipcon, scopename,stride=2, filterlen=3, training=True,withbias=True):
    '''
    Upsampling -->conv(channel/2) --> Leaky ReLU --> concat --> Convolution(channel/2) + Leaky ReLU
    '''
    inputshape=inputdata.get_shape().as_list()
    if 1:
        #use blinear to upsample
        tep=tf.image.resize_bilinear(inputdata, (inputshape[1]*stride, inputshape[2]*stride) )
        tep=my_conv(tep, filterlen, outchannel, scopename+'_conv1', stride=1, withbias=withbias)
        tep=my_batchnorm( tep,training, scopename+'_bn1')
        tep=my_lrelu(tep, scopename)
        '''
        #单个cov无法拟合xor操作，而这里需要一个选择pixel的操作，线性操作不行
        tep=my_conv(tep, filterlen, outchannel, scopename+'_conv2', stride=1, withbias=withbias)
        tep=my_batchnorm( tep,training, scopename+'_bn2')
        tep=my_lrelu(tep, scopename)
        '''
    else:
        #use deconv to upsample
        tep=my_deconv(inputdata, filterlen, outchannel, scopename+'_deconv1', stride, withbias=withbias)
        tep=my_batchnorm( tep,training, scopename+'_bn1')
        tep=my_lrelu(tep, scopename)
    
    if skipcon is  not None:
        print ('-->concating:',tep, skipcon)
        tshape=skipcon.get_shape().as_list()
        tep=tf.image.resize_bilinear(tep, (tshape[1], tshape[2]) )
            
        tep=tf.concat([tep, skipcon], -1)
    
    #单个conv无法拟合
    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv3', stride=1, withbias=withbias)
    tep=my_batchnorm( tep,training, scopename+'_bn3')
    tep=my_lrelu(tep, scopename)
    '''
    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv4', stride=1, withbias=withbias)
    tep=my_batchnorm( tep,training, scopename+'_bn4')
    tep=my_lrelu(tep, scopename)
    '''
    
    return tep

def unet_down(inputdata, outchannel, scopename,stride=2, filterlen=3,training=True, withbias=True):
    '''
    downsampling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    '''
    if 0:
        #tep=tf.layers.average_pooling2d(inputdata, stride,stride)
        tep=tf.layers.max_pooling2d(inputdata, stride,stride)
        stride=1
    else:
        tep=inputdata

    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv1', stride=stride, withbias=withbias)
    tep=my_batchnorm( tep,training, scopename+'_bn1')
    tep=my_lrelu(tep, scopename)
    
    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv2', stride=1, withbias=withbias)
    tep=my_batchnorm( tep,training, scopename+'_bn2')
    tep=my_lrelu(tep, scopename)
    
    return tep
    
def my_unet(inputdata, layercnt=3,  filterlen=3,training=True,  withbias=True):
    '''
    layercnt:下降和上升各有几层,原则上应该是一对一
    '''
    
    inputshape=inputdata.get_shape().as_list()
    channel_init=inputshape[-1]
    
    tep=my_conv(inputdata, filterlen+int(layercnt/2), channel_init*2, scopename='unet_down_start', stride=1,  withbias=withbias)
    tep=my_lrelu(tep, 'unet_down_start')
    
    print ('\nforming UNET-->layer:',layercnt)
    print (tep)
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
    

    tep=my_conv(tep, filterlen, 6, scopename='unet_up_end0', stride=1, withbias=withbias)
    
    tep=my_batchnorm( tep,training, 'unet_up_end0_bn2')
    tep=my_lrelu(tep, 'unet_up_end0_relu')
    print (tep)
    
    tep=my_conv(tep, filterlen, 3, scopename='unet_up_end1', stride=1, withbias=withbias)
    tep=tf.image.resize_images(tep, [inputshape[1],inputshape[2]], method=tf.image.ResizeMethod.BILINEAR)
    print (tep)
    
    return tep


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
            
            
        return one_channel,ano_channel



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
    '''
    flipconv_method=my_novel_conv_withweight
    inputshape=inputdata.get_shape().as_list()
    channel_init=inputshape[-1]
    skipcon1=[]
    skipcon2=[]
    ##########################################################################
    #first unet-down input1:the first frame
    tep=my_conv(inputdata , filterlen+2*int(layercnt-1), channel_init*2, scopename='unet_down_start', stride=1,  withbias=withbias)
    #tep=my_batchnorm( tep,training, 'trace_1_unet_down_start_bn')
    tep=my_lrelu(tep, 'trace_1_unet_down_start_relu')
    print (tep)
    
    print ('\nforming UNET-->layer1:',layercnt)
    
    for i in range(layercnt):
        skipcon1.append(tep)
        tep=unet_down(tep, channel_init*( 2**(i+2)), 'unet_down_'+str(i), stride=2, filterlen=filterlen+2*int( (layercnt-i)-1 ), training=training,withbias=withbias)
        print (tep)
        
    input1_fea=tep
    
    #######################################################################   
    #then unet-down the second frame 
    tep=my_conv(inputdata2, filterlen+2*int(layercnt-1), channel_init*2, scopename='unet_down_start', stride=1,  withbias=withbias)
    #tep=my_batchnorm( tep,training, 'trace_2_unet_down_start_bn')
    tep=my_lrelu(tep, 'trace_2_unet_down_start_relu') 
    print (tep)
    
    print ('\nforming UNET-->layer2:',layercnt)
    
    for i in range(layercnt):
        skipcon2.append(tep)
        tep=unet_down(tep, channel_init*( 2**(i+2)), 'unet_down_'+str(i), stride=2, filterlen=filterlen+2*int( (layercnt-i)-1 ), training=training,withbias=withbias)
        print (tep)
        
    input2_fea=tep
    
    ##################连接两个部分
    #concating two middle feature
    tep_f1, tep_f2=flipconv_method(input1_fea, input2_fea, filterlen, 'middle_novel_cnn', training=training)
    tep=tf.concat( [tep_f1, tep_f2], -1)
    print (tep)
    
    ######################################################up
    #begining unet-up 
    for i in reversed(range(layercnt)):
        tep1=skipcon1[i]
        tep2=skipcon2[i]
        tep_f1, tep_f2=tep1, tep2
        
        for j in range(layercnt-i):  #分别为1，2，3层
            tep_f1, tep_f2=flipconv_method(tep_f1, tep_f2, filterlen+int( 2*(layercnt-i-1) ), 'unet_up_novel_cnn_'+str(i),  training=training)
        
        skipcon=tf.concat([tep1, tep_f1, tep_f2, tep2], -1) #!!!!!
        tep=unet_up(tep, channel_init*( 2**(i+1)), skipcon,'unet_up_'+str(i), stride=2,  filterlen=filterlen+int( 2*(layercnt-i-1) ),  training=training,withbias=withbias)
        print (tep)
        
    #finish the net 
    tep=my_conv(tep, filterlen+2*int(layercnt-1), channel_init*2, scopename='unet_up_final_1', stride=1, withbias=withbias)
    tep=my_batchnorm( tep,training, 'unet_up_final_1_bn')
    tep=my_lrelu(tep, 'unet_up_final_1_relu') 
    print (tep)
    
    #finally channel to 3
    tep=my_conv(tep, filterlen+2*int(layercnt-1), outchannel, scopename='unet_up_final_2', stride=1, withbias=withbias)
    
    tep=tf.image.resize_images(tep, [inputshape[1],inputshape[2]], method=tf.image.ResizeMethod.BILINEAR)
    print (tep)
    
    return tep
    
    
def my_D_block(inputdata, outchannel, scopename,stride=2, filterlen=3, withbias=True, training=True):
    tep=my_conv(inputdata, filterlen, outchannel, scopename+'_conv1', stride=stride, withbias=withbias)
    
    tep=my_batchnorm( tep,training, scopename)
    tep=my_lrelu(tep, scopename)
    return tep   


def test_unet():
    imgs_pla = tf.placeholder(datatype, [32, 360/2, 640/2, 3], name='imgs_in')
    
    with tf.variable_scope("G_Net",  reuse=tf.AUTO_REUSE) as scope:
        #tep=my_unet(imgs_pla)
        tep=my_novel_unet(imgs_pla, imgs_pla)
        trainvars=tf.trainable_variables()
        
        print ()
        for i in trainvars:
            print (i)


if __name__=='__main__':
    #test_my_find_flip()
    test_unet()
    
                
                
                
                
                
                
                
                