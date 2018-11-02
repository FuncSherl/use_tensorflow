import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from datetime import datetime
import time,cv2
import os.path as op
import use_tensor.locate_feature.proc_voc as proc_voc


#----------------------------------------------------------------------------------panel
train_size=12000 #训练集规模
eval_size=5000 #测试集规模


out_class=20 #输出类别数目，这里用voc有20类
batchsize=30

base_lr=0.001 #基础学习率
maxstep=30000 #训练多少次

decay_steps=1000
decay_rate=0.9

dropout_rate=0.5

#---------------------------------------------------------------------------------------
TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

#利用pipline方式读入数据,由于用同一个网络跑train和test，就先sess.run获取到数据再利用placeholder将数据传入网络
train_imgs,train_labs=proc_voc.read_tfrecord_batch('./VOC_train_data-2018-11-02',batchsize)
test_imgs, test_labs=proc_voc.read_tfrecord_batch('./VOC_test_data-2018-11-02',batchsize)


class vgg16:
    def __init__(self,  weights=None, sess=None):
        self.imgs = tf.placeholder(tf.float32, [batchsize, 224, 224, 3])
        self.labs = tf.placeholder(tf.int32, [batchsize])
        self.training=tf.placeholder(tf.bool)
        self.dropout=dropout_rate
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        #先构建网络结构
        self.convlayers()
        self.fc_layers()
        
        #再构建train等操作
        self.loss=self.getloss()#loss operation
        self.train_op=self.training_op(baselr=base_lr,decay_steps=decay_steps, decay_rate=decay_rate)
        self.eval_batch_op=self.eval_batch(topk=1)
        

        self.probs = tf.nn.softmax(self.fc3l)
        
        for i in self.__dict__:
            print('dict in vgg16:',i,self.__dict__[i])
        
        self.merged = tf.summary.merge_all()
        
        init = tf.global_variables_initializer()#初始化tf.Variable,虽然后面会有初始化权重过程，但是最后一层是要根据任务来的,无法finetune，其参数需要随机初始化
        sess.run(init)
        
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
        
            
            
    def getloss(self):
        losst=tf.losses.sparse_softmax_cross_entropy(labels=self.labs, logits=self.fc3l )#这里应该传入softmax之前的tensor
        cross_entropy_mean = tf.reduce_mean(losst)
        
        tf.summary.scalar('loss',cross_entropy_mean)
        
        return cross_entropy_mean
    
    def training_op(self,baselr=base_lr,decay_steps=1000, decay_rate=0.99):
        lr_rate = tf.train.exponential_decay(baselr,  global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate)
        
        tf.summary.scalar('learning rate', lr_rate)
    
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        print ('GradientDescentOptimizer1 to minimize %d vars..'%(len(self.parameters)))
        train_op1 = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss, global_step=self.global_step,var_list=self.parameters)
        
        print ('GradientDescentOptimizer2 to minimize %d vars..'%(len(self.parameters_last)))
        train_op2 = tf.train.GradientDescentOptimizer(lr_rate*10).minimize(self.loss, var_list=self.parameters_last)#这里不应有globalstep，否则会再加1？
        train_op = tf.group(train_op1, train_op2)

        return train_op
    
    
    def train_once(self, sess):
        imgst,labels=sess.run([train_imgs,train_labs])
        
        #训练时启用dropout
        los,_, sum_ret=sess.run([self.loss, self.train_op, self.merged], feed_dict={self.imgs: imgst,self.labs: labels, self.training:True})
        return los,sum_ret

    
    def eval_batch(self,topk=1):#测试一个batch的operation
        top_k_op = tf.nn.in_top_k(self.fc3l, self.labs, topk)
        cnt=tf.reduce_sum(tf.cast(top_k_op,tf.int32))
        
        #tf.summary.scalar('accuracy rate:', (cnt)/labels.shape[0])
        tf.summary.scalar('accuracy rate:', cnt/self.labs.shape[0])
        return cnt
    
    def eval_once(self,sess, eval_size=eval_size):
        cnt_true=0.0
        
        batch_num=int(eval_size/batchsize)+1
        for i in range(batch_num):
            #get test datas
            imgst,labst=sess.run([test_imgs, test_labs])
            
            eval_b=self.eval_batch_op
            
            #how many true in every batch
            #这里是测试，应设置training为false
            batch_true = sess.run([eval_b], feed_dict={self.imgs: imgst, self.labs: labst, self.training:False})[0]
            
            print (i,'/',batch_num,'  eval one batch:',batch_true,'/',batchsize,'-->',float(batch_true)/batchsize)
            
            #top_k_op = tf.nn.in_top_k(self.probs, labst, topk)
            cnt_true+=batch_true   # tf.reduce_sum(tf.cast(top_k_op,tf.int32))
        
        rate_true=cnt_true/(batch_num*batchsize)
        
        return rate_true


    def convlayers(self):
        self.parameters = []
        self.parameters_last=[]

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')


    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]
        
        #dropout1
        self.fc1=tf.cond(self.training, lambda: tf.nn.dropout(self.fc1, self.dropout), lambda: self.fc1)
        
        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]
            
        #dropout2
        self.fc2=tf.cond(self.training, lambda: tf.nn.dropout(self.fc2, self.dropout), lambda: self.fc2)

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, out_class],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[out_class], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters_last += [fc3w, fc3b]    #here we want to finetune,so shouldn't init the weight with model
            
            

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print (i, k, np.shape(weights[k]))
            if i<len(self.parameters):
                sess.run(self.parameters[i].assign(weights[k]))
                print ('load weight of ',self.parameters[i],'\n')
            else:
                print ('skip:',k)
                pass
                #sess.run(self.parameters[i].initializer)






if __name__ == '__main__':
    logdir="./logs/VOC_"+TIMESTAMP+('_base_lr-%f_batchsize-%d_maxstep-%d'%(base_lr,batchsize, maxstep))
    

    
    with tf.Session() as sess:
        logwriter = tf.summary.FileWriter(logdir,   sess.graph)        
        
        #imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg16( r'./vgg16_weights.npz', sess)
        
        all_saver = tf.train.Saver(max_to_keep=2) 
                
        
        #再开始前先测试一次，看下准确率，也将summary加入tf中，免得后面tf.summary.merge_all()没将这个eval_once里面的summary算进去
        acc=vgg.eval_once(sess)
        print ('eval the test datas:',acc)
        
        begin_t=time.time()
        for i in range(maxstep):            
            if ((i+1)%300==0):
                acc=vgg.eval_once(sess)
                print ('training at %d eval the test datas:'%i,acc)
                
                print ('saving models...')
                pat=all_saver.save(sess, op.join(logdir,'model_keep'),global_step=i)
                print ('saved at:',pat)
            
            stt=time.time()
            print ('\n%d/%d  start train_once...'%(i,maxstep))
            lost,sum_log=vgg.train_once(sess) #这里每次训练都run一个summary出来
            
            #写入日志
            logwriter.add_summary(sum_log, i)
            #print ('write summary done!')
            
            print ('train once-->loss:',lost,'  time:',time.time()-stt)
        
        print ('Training done!!!-->time used:',(time.time()-begin_t))
        
        
        
        '''
        img1 = imread('laska.png', mode='RGB')
        img1 = imresize(img1, (224, 224))
    
        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print (prob[p])
        '''
            
        '''
        
        vgg = vgg16.Vgg16()
        vgg.build(image)
        feature_map = vgg.pool5
        mask = yournetwork(feature_map)
        '''
            
        
        
