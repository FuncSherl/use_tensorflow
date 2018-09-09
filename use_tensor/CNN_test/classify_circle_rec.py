#coding:utf-8
'''
Created on 2018年9月9日

@author: sherl
'''
import tensorflow as tf
import numpy as np
import cv2


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

fcn1_n=1024

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

        
    with tf.name_scope('fcn1') as scope:
        reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        
        weights = tf.Variable( tf.truncated_normal( [dim, fcn1_n], stddev=5e-2)  ,    name='fcn1')
        biases = tf.Variable(tf.zeros([fcn1_n]),  name='biases')
        
        fcn1 = tf.nn.relu(tf.matmul(pool2, weights) + biases)
        
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable( tf.truncated_normal([fcn1_n, num_class], stddev=5e-2) , name='weights')
        biases = tf.Variable(tf.zeros([num_class]), name='biases')
        logits = tf.matmul(fcn1, weights) + biases
        
    return logits



def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

      Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].

      Returns:
        loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar('loss',loss)
    return loss


def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluate(logits, labels, topk=1):
    top_k_op = tf.nn.in_top_k(logits, labels, topk)
    cnt=tf.reduce_sum(tf.cast(top_k_op,tf.int32))

    return cnt

def gen_images(batchsize=batch_size, imgsize=img_size, channel=1):
    image=np.zeros([batchsize, imgsize, imgsize, channel], dtype=np.uint8)
    label=np.zeros([batchsize], dtype=np.int32)
    
    for i in range(batchsize):
        tep=np.random.randint(num_class)
        
        label[i]=tep
        
        if tep:           #为真时
            h=np.random.randint(int(imgsize*1/5), int(imgsize*3/5))
            w=np.random.randint(int(imgsize*1/5), int(imgsize*3/5))
            
            stx=np.random.randint(int(imgsize*1/5), int(imgsize*3/5))
            sty=np.random.randint(int(imgsize*1/5), int(imgsize*3/5))
            
            
            cv2.rectangle(image[i],(stx,sty),(stx+w,sty+h), np.random.randint(140,250),3)
            
            #cv2.imshow('test',image[i])
            #cv2.waitKey()
            pass
        else:
            '''
            cv2.circle(img, (50,50), 10, (0,0,255),-1)

            #img:图像，圆心坐标，圆半径，颜色，线宽度(-1：表示对封闭图像进行内部填满)
            '''
            r=np.random.randint(int(imgsize*1/5), int(imgsize*3/5))
            
            stx=np.random.randint(int(imgsize*1/5), int(imgsize*3/5))
            sty=np.random.randint(int(imgsize*1/5), int(imgsize*3/5))
            
            cv2.circle(image[i], (stx,sty),r, (np.random.randint(140,250)) ,3)
           
        ''' 
        print (tep)
        cv2.imshow('test',image[i])
        cv2.waitKey()
        '''
    return image,label
    
    
        
        
        

if __name__ == '__main__':
    gen_images(imgsize=400)
    pass