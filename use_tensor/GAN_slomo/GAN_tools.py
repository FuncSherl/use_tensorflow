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

batchsize=32

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def my_batchnorm(self, data,training, scope):
        return tf.contrib.layers.batch_norm(data,
                                            center=True, #如果为True，有beta偏移量；如果为False，无beta偏移量
                                            decay=0.9,#衰减系数,即有一个moving_mean和一个当前batch的mean，更新moving_mean=moving_mean*decay+(1-decay)*mean
                                            #合适的衰减系数值接近1.0,特别是含多个9的值：0.999,0.99,0.9。如果训练集表现很好而验证/测试集表现得不好，选择小的系数（推荐使用0.9）
                                            updates_collections=None,
                                            epsilon=1e-5, #防止除0
                                            scale=True, #如果为True，则乘以gamma。如果为False，gamma则不使用。当下一层是线性的时（例如nn.relu），由于缩放可以由下一层完成,可不要
                                            #reuse=tf.AUTO_REUSE,  #reuse的默认选项是None,此时会继承父scope的reuse标志
                                            #param_initializers=None, # beta, gamma, moving mean and moving variance的优化初始化
                                            #activation_fn=None, #用于激活，默认为线性激活函数
                                            #param_regularizers=None,# beta and gamma正则化优化
                                            #data_format=DATA_FORMAT_NHWC,
                                            is_training=training, # 图层是否处于训练模式。
                                            scope=scope)
        
def my_deconv(inputdata, socpename,reuse=tf.AUTO_REUSE, withbias=True):
    with tf.variable_scope(socpename,  reuse=reuse) as scope:  
            kernel=tf.get_variable('weights', [4,4, 128, 128], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
            
            #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
            #而tf.nn.conv2d_transpose中的filter参数，是[filter_height, filter_width, out_channels，in_channels]的形式
            deconv=tf.nn.conv2d_transpose(inputdata, kernel, output_shape=[batchsize, 32, 32, 128], strides=[1,2,2,1], padding="SAME")
            
            if withbias:
                bias=tf.get_variable('bias', [128], dtype=tf.float32, initializer=tf.constant_initializer(bias_init))
                ret=tf.nn.bias_add(deconv, bias)
                
                
                
                
                
                
                
                