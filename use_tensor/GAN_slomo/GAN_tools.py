#coding:utf-8
'''
Created on 2019年2月19日

@author: sherl
'''
import tensorflow as tf

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
    
def my_lrelu(inputdata, scopename):
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
    
    
def unet_up(inputdata, outchannel, scopename,stride=2, filterlen=3, withbias=True):
    '''
    Upsampling --> Leaky ReLU --> Convolution + Leaky ReLU
    '''
    inputshape=inputdata.get_shape().as_list()
    #use blinear to upsample
    #tep=tf.image.resize_images(inputdata, (inputshape[1]*stride, inputshape[2]*stride) )
    #use deconv to upsample
    tep=my_deconv(inputdata, filterlen, outchannel, scopename+'_deconv1', stride, withbias=withbias)
    tep=my_lrelu(tep, scopename)
    
    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv1', stride=1, withbias=withbias)
    tep=my_lrelu(tep, scopename)
    
    return tep

def unet_down(inputdata, outchannel, scopename,stride=2, filterlen=3, withbias=True):
    '''
    downsampling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    '''
    tep=my_conv(inputdata, filterlen, outchannel, scopename+'_conv1', stride=stride, withbias=withbias)
    tep=my_lrelu(tep, scopename)
    
    tep=my_conv(tep, filterlen, outchannel, scopename+'_conv2', stride=1, withbias=withbias)
    tep=my_lrelu(tep, scopename)
    
    return tep
    
def my_unet(inputdata, layercnt=5,  filterlen=3, withbias=True):
    '''
    layercnt:下降和上升各有几层,原则上应该是一对一
    '''
    
    inputshape=inputdata.get_shape().as_list()
    channel_init=inputshape[-1]
    
    tep=my_conv(inputdata, filterlen+int(layercnt/2), channel_init*2, scopename='unet_down_start', stride=1,  withbias=withbias)
    tep=my_lrelu(tep, 'unet_down_start')
    print ('\nforming UNET-->layer:',layercnt)
    print (tep)
    for i in range(layercnt):
        tep=unet_down(tep, channel_init*( 2**(i+2)), 'unet_down_'+str(i), filterlen=filterlen+int( (layercnt-i)/2 ), withbias=withbias)
        print (tep)
    
    for i in reversed(range(layercnt)):
        tep=unet_up(tep, channel_init*( 2**(i+1)), 'unet_up_'+str(i), filterlen=filterlen+int( (layercnt-i)/3 ),  withbias=withbias)
        print (tep)
        
    tep=my_conv(tep, filterlen, 3, scopename='unet_up_end', stride=1, withbias=withbias)
    tep=tf.image.resize_images(tep, [inputshape[1],inputshape[2]], method=tf.image.ResizeMethod.BILINEAR)
    print (tep)
    return tep


def my_D_block(inputdata, outchannel, scopename,stride=2, filterlen=3, withbias=True, training=True):
    tep=my_conv(inputdata, filterlen, outchannel, scopename+'_conv1', stride=stride, withbias=withbias)
    
    tep=my_batchnorm( tep,training, scopename)
    tep=my_lrelu(tep, scopename)
    return tep
    



if __name__ == '__main__':
    imgs_pla = tf.placeholder(datatype, [32, 360, 720, 3*2], name='imgs_in')
    
    with tf.variable_scope("G_Net",  reuse=tf.AUTO_REUSE) as scope:
        tep=my_unet(imgs_pla)
        trainvars=tf.trainable_variables()
        
        print ()
        for i in trainvars:
            print (i)
    
    
                
                
                
                
                
                
                
                