#coding:utf-8
'''
Created on 2018年9月9日

@author: sherl
'''
import tensorflow as tf
import numpy as np
import cv2,time
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now())


######-------------------------------------------------------------------------------------
cnn1_k=6  #有几个核
cnn1_ksize=5  #每个核的大小
cnn1_stride=1  #步长

pool1_size=3  #每个核的大小
pool1_stride=2 #步长

cnn2_k=6
cnn2_ksize=5
cnn2_stride=1

pool2_size=3
pool2_stride=2

fcn1_n=1024

num_class=2

img_size=32
lr=1e-6

batch_size=64

stdev_init=0.1
#-----------------------------------------------------------------------------------panel

def inference(images):
    '''
    input:[batch, w,h, channel]
    
    net structure:   cnn1->pool1->cnn2->pool2->fc1->fc2
    '''
    tf.summary.image('initial_images', images,max_outputs=4)
    
    with tf.name_scope('cnn1') as scope:
        kernel = tf.Variable( tf.truncated_normal( [cnn1_ksize, cnn1_ksize, images.get_shape().as_list()[-1], cnn1_k], stddev=stdev_init)  ,    name='kernels')
        biases = tf.Variable(tf.zeros([cnn1_k]),  name='biases')      
        
                        
        conv = tf.nn.conv2d(images, kernel, [1, cnn1_stride, cnn1_stride, 1], padding='SAME')
        
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope)
        
        #这里kernel的最后一维为1，就作为灰度图输出，需要交换一些维度，将原本最后一维out_num当作batch
        tf.summary.image('first_cnn_kernels',tf.transpose(kernel, perm=[3,0,1,2]), max_outputs=6)
        
        #这里输出的特征图为[batch, outw, outh, out_channel],取其第一个batch，将最后的out_channel当作batch，添上一维（就是最后加了个1），当作灰度图输出
        tf.summary.image('first_cnn_features',tf.expand_dims(   tf.transpose(conv1[0], perm=[2,0,1]),   3), max_outputs=6)
        
        
        
    with tf.name_scope('pool1') as scope:
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, pool1_size, pool1_size, 1], strides=[1, pool1_stride, pool1_stride, 1], padding='SAME', name='pool1')
        # norm1
        #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        #tf.summary.image('first_pool_features',tf.expand_dims(pool1[0], 3), max_outputs=cnn1_k)
        #输出同上
        tf.summary.image('first_pool_features',tf.expand_dims(   tf.transpose(pool1[0], perm=[2,0,1]),   3), max_outputs=6)
        
        
    with tf.name_scope('cnn2') as scope:
        kernel = tf.Variable( tf.truncated_normal( [cnn2_ksize, cnn2_ksize, cnn1_k, cnn2_k], stddev=stdev_init)  ,    name='kernels')
        biases = tf.Variable(tf.zeros([cnn2_k]),  name='biases')      
        
                        
        conv = tf.nn.conv2d(pool1, kernel, [1, cnn2_stride, cnn2_stride, 1], padding='SAME')
        
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope)
        
        #同上，这里kernel第2维不是1了就将cnn2_k个kernel中取出来一个，将其每一层当作一个灰度图输出
        tf.summary.image('second_cnn_kernels',   tf.expand_dims(tf.transpose(kernel, perm=[3,2,0,1])[0]    ,3)      ,    max_outputs=6)
        tf.summary.image('second_cnn_features',tf.expand_dims(   tf.transpose(conv2[0], perm=[2,0,1])   , 3), max_outputs=6)
    
    with tf.name_scope('pool2') as scope:
         # pool1
        pool2 = tf.nn.max_pool(conv2, ksize=[1, pool2_size, pool2_size, 1], strides=[1, pool2_stride, pool2_stride, 1], padding='SAME', name='pool2')
        #tf.summary.image('second_pool_features',tf.expand_dims(pool2[0], 3), max_outputs=10)
        tf.summary.image('second_pool_features',tf.expand_dims(   tf.transpose(pool2[0], perm=[2,0,1]),   3), max_outputs=6)
        
    with tf.name_scope('fcn1') as scope:
        reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        
        weights = tf.Variable( tf.truncated_normal( [dim, fcn1_n], stddev=stdev_init)  ,    name='fcn1')
        biases = tf.Variable(tf.zeros([fcn1_n]),  name='biases')
        
        fcn1 = tf.nn.relu(tf.matmul(reshape, weights) + biases)
        
        #h_fc1_drop = tf.nn.dropout(h_fc1,0.5)
        
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable( tf.truncated_normal([fcn1_n, num_class], stddev=stdev_init) , name='weights')
        biases = tf.Variable(tf.zeros([num_class]), name='biases')
        logits = tf.matmul(fcn1, weights) + biases
        
    return logits



def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

      Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].

      Returns:
        losst: Loss tensor of type float.
    """
    #labels = tf.to_int64(labels)
    losst=tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    #print ('loss shape:',losst.get_shape().as_list())
    cross_entropy_mean = tf.reduce_mean(losst)
    
    tf.summary.scalar('loss',cross_entropy_mean)
    
    return cross_entropy_mean


def training(losst, learning_rate):
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr_rate = tf.train.exponential_decay(learning_rate,  global_step=global_step, decay_steps=1000, decay_rate=0.99)
    
    tf.summary.scalar('learning rate', lr_rate)
    
    optimizer = tf.train.GradientDescentOptimizer(lr_rate)
    
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(losst, global_step=global_step)
    return train_op


def evaluate(logits, labels, topk=1):
    top_k_op = tf.nn.in_top_k(logits, labels, topk)
    cnt=tf.reduce_sum(tf.cast(top_k_op,tf.int32))
    
    tf.summary.scalar('accuracy rate:', (cnt)/labels.shape[0])
    return cnt

def gen_images(batchsize=batch_size, imgsize=img_size, channel=1):
    image=np.zeros([batchsize, imgsize, imgsize, channel], dtype=np.float32)
    label=np.zeros([batchsize], dtype=np.int32)
    
    for i in range(batchsize):
        tep=np.random.randint(num_class)
        
        label[i]=tep
        
        if tep:           #为真时
            #随机生成宽高和左上角位置
            h=np.random.randint(int(imgsize*3/10), int(imgsize*3/5))
            w=np.random.randint(int(imgsize*3/10), int(imgsize*3/5))
            
            stx=np.random.randint(int(imgsize*1/5), int(imgsize*3/5))
            sty=np.random.randint(int(imgsize*1/5), int(imgsize*3/5))
            
            
            cv2.rectangle(image[i],(stx,sty),(stx+w,sty+h), np.random.randint(140,250),int (imgsize/20))
            
            #cv2.imshow('test',image[i])
            #cv2.waitKey()
            pass
        else:
            '''
            cv2.circle(img, (50,50), 10, (0,0,255),-1)

            #img:图像，圆心坐标，圆半径，颜色，线宽度(-1：表示对封闭图像进行内部填满)
            
            '''
            #随机生成圆心和半径
            r=np.random.randint(int(imgsize*3/10), int(imgsize*3/5))
            
            stx=np.random.randint(int(imgsize*1/5), int(imgsize*3/5))
            sty=np.random.randint(int(imgsize*1/5), int(imgsize*3/5))
            
            cv2.circle(image[i], (stx,sty),r, (np.random.randint(140,250)) ,int (imgsize/20))
           
        ''' 
        print (image[i])
        print (tep)
        cv2.imshow('test',image[i])
        cv2.waitKey()
        '''
    return image,label


def start(lr=lr):
    dat_place = tf.placeholder(tf.float32, shape=(batch_size, img_size,img_size,1))
    label_place= tf.placeholder(tf.int32, shape=(batch_size))
    
    logits=inference(dat_place)
    los=loss(logits, label_place)
    
    train_op=training(los, lr)
    eval_op=evaluate(logits, label_place)
    
    #合并上面每个summary，就不必一个一个运行了
    merged = tf.summary.merge_all()
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()#初始化tf.Variable
        sess.run(init)
        
        #tensorboard里面按文件夹分，这里利用时间分开
        writer = tf.summary.FileWriter("./logs/"+TIMESTAMP, sess.graph)
        
        sttime=time.time()
        
        for i in range(3000):
            
            #print (dat)\
            stt=time.time()
            
            
            dat,lab=gen_images()#generate new images生成一批数据
            _, loss_value, summary_resu , tep= sess.run([train_op, los, merged, logits], feed_dict={dat_place:dat, label_place:lab})
            
            #写入日志
            writer.add_summary(summary_resu, i)
            
            if i%10==0:        #显示一次
                #print (tep)
                print (i, 'time:',time.time()-stt)
                print ('training-loss:',loss_value,'\n')
            
            if (i+1)%100==0:#测试一次
                truecnt=0
                cnt_all=0
                for j in range(200):
                    dat,lab=gen_images()#generate new images
                    
                    eval_resu,loss_value=sess.run([eval_op, los], feed_dict={dat_place:dat, label_place:lab})
                    truecnt+=eval_resu
                    cnt_all+=lab.shape[0]
                    #print (i, 'time:',time.time()-stt)
                    #print ('evaluate-loss:',loss_value,'\n')
                
                print ('!!!!!!!!evaluate:!!!!!!!!!!!!',float(truecnt)/cnt_all,'\n')
            
            
        print('training done! time used:',time.time()-sttime)
    
        
        
        

if __name__ == '__main__':
    start()
    pass