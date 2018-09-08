#coding:utf-8
'''
Created on 2018年9月9日

@author: sherl
'''
import tensorflow as tf

######-------------------------------------------------------------------------------------
cnn1_k=32
cnn1_ksize=5
cnn1_stride=1

pool1_size=3
pool1_stride=2

cnn2_k=64
cnn2_ksize=5
cnn2_stride=1

pool2_size=3
pool2_stride=2

num_class=2

img_size=28

batch_size=64

#-----------------------------------------------------------------------------------panel

def inference(images):
    tf.summary.image('initial_images', images)
    
    with tf.name_scope('cnn1') as scope:
        kernel = tf.Variable( tf.truncated_normal( [cnn1_ksize, cnn1_ksize, 1, cnn1_k], stddev=5e-2)  ,    name='kernels')
        biases = tf.Variable(tf.zeros([cnn1_k]),  name='biases')      
        
                        
        conv = tf.nn.conv2d(images, kernel, [1, cnn1_stride, cnn1_stride, 1], padding='SAME')
        
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        
        tf.summary.image('first_cnn_features',tf.expand_dims(conv1[0], 3), max_outputs=10)
        
        
        
    with tf.name_scope('pool1') as scope:
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, pool1_size, pool1_size, 1], strides=[1, pool1_stride, pool1_stride, 1], padding='SAME', name='pool1')
        # norm1
        #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        
        
    with tf.name_scope('cnn2') as scope:
        kernel = tf.Variable( tf.truncated_normal( [cnn2_ksize, cnn2_ksize, cnn1_k, cnn2_k], stddev=5e-2)  ,    name='kernels')
        biases = tf.Variable(tf.zeros([cnn1_k]),  name='biases')      
        
                        
        conv = tf.nn.conv2d(images, kernel, [1, cnn2_stride, cnn2_stride, 1], padding='SAME')
        
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        
        tf.summary.image('second_cnn_features',tf.expand_dims(conv2[0], 3), max_outputs=10)
    
    with tf.name_scope('pool2') as scope:
         # pool1
        pool2 = tf.nn.max_pool(conv2, ksize=[1, pool2_size, pool2_size, 1], strides=[1, pool2_stride, pool2_stride, 1], padding='SAME', name='pool2')

        
        
        
        
        
        
        
        

if __name__ == '__main__':
    pass