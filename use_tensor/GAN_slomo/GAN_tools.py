#coding:utf-8
'''
Created on 2019年2月19日

@author: sherl
'''
import tensorflow as tf

def mybatchnorm(self, data,training, scope):
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
        
