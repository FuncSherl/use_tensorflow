#coding:utf-8
'''
Created on 2018年9月9日

@author: sherl
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import cv2,time
import os.path as op
from datetime import datetime
import cifar10_input,cifar10

TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

######-------------------------------------------------------------------------------------
cnn1_k=64  #有几个核
cnn1_ksize=4  #每个核的大小
cnn1_stride=1  #步长

pool1_size=2  #每个核的大小
pool1_stride=2 #步长

cnn2_k=64
cnn2_ksize=5
cnn2_stride=1

pool2_size=2
pool2_stride=2

fcn1_n=1024

num_class=10


#-----------------------------------------------------------------------------------net params
img_size=cifar10_input.IMAGE_SIZE
lr=0.01

batch_size=36
maxiter=50000
max_output=6

stdev_init=0.1
#-----------------------------------------------------------------------------------panel

def inference(images):
    '''
    input:[batch, w,h, channel]
    
    net structure:   cnn1->pool1->cnn2->pool2->fc1->fc2
    '''
    tf.summary.image('initial_images', images,max_outputs=max_output)
    
    with tf.variable_scope('cnn1') as scope:
        kernel = tf.get_variable(  'kernels',[cnn1_ksize, cnn1_ksize, images.get_shape().as_list()[-1], cnn1_k]  ,
                                   initializer=tf.truncated_normal_initializer(stddev=stdev_init))
        
        biases = tf.get_variable('biases', [cnn1_k],  
                                   initializer=tf.constant_initializer(0))      
        
        
                        
        conv = tf.nn.conv2d(images, kernel, [1, cnn1_stride, cnn1_stride, 1], padding='SAME')
        
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        
        #这里kernel的最后一维为1，就作为灰度图输出，需要交换一些维度，将原本最后一维out_num当作batch
        tf.summary.image('first_cnn_kernels',tf.transpose(kernel, perm=[3,0,1,2]), max_outputs=max_output)
        
        #这里输出的特征图为[batch, outw, outh, out_channel],取其第一个batch，将最后的out_channel当作batch，添上一维（就是最后加了个1），当作灰度图输出
        tf.summary.image('first_cnn_features',tf.expand_dims(   tf.transpose(conv1[0], perm=[2,0,1]),   3), max_outputs=max_output)
        
        tf.summary.histogram('first_cnn_biases',biases)
        tf.summary.histogram('first_cnn_kernels',kernel)
        
        
    with tf.variable_scope('pool1') as scope:
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, pool1_size, pool1_size, 1], strides=[1, pool1_stride, pool1_stride, 1], padding='SAME', name='pool1')
        # norm1
        #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        #tf.summary.image('first_pool_features',tf.expand_dims(pool1[0], 3), max_outputs=cnn1_k)
        #输出同上
        tf.summary.image('first_pool_features',tf.expand_dims(   tf.transpose(pool1[0], perm=[2,0,1]),   3), max_outputs=max_output)
        
        
    with tf.variable_scope('cnn2') as scope:
        kernel = tf.get_variable( 'kernels', [cnn2_ksize, cnn2_ksize, cnn1_k, cnn2_k]  ,    
                                  initializer=tf.truncated_normal_initializer(stddev=stdev_init))
        
        biases = tf.get_variable('biases',[cnn2_k], 
                                  initializer=tf.constant_initializer(0))      
        
                        
        conv = tf.nn.conv2d(pool1, kernel, [1, cnn2_stride, cnn2_stride, 1], padding='SAME')
        
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        
        #同上，这里kernel第2维不是1了就将cnn2_k个kernel中取出来一个，将其每一层当作一个灰度图输出
        tf.summary.image('second_cnn_kernels',   tf.expand_dims(tf.transpose(kernel, perm=[3,2,0,1])[0]    ,3)      ,    max_outputs=max_output)
        tf.summary.image('second_cnn_features',tf.expand_dims(   tf.transpose(conv2[0], perm=[2,0,1])   , 3), max_outputs=max_output)
        
        tf.summary.histogram('second_cnn_biases',biases)
        tf.summary.histogram('second_cnn_kernels',kernel)
    
    with tf.variable_scope('pool2') as scope:
        # pool1
        pool2 = tf.nn.max_pool(conv2, ksize=[1, pool2_size, pool2_size, 1], strides=[1, pool2_stride, pool2_stride, 1], padding='SAME', name='pool2')
        #tf.summary.image('second_pool_features',tf.expand_dims(pool2[0], 3), max_outputs=10)
        tf.summary.image('second_pool_features',tf.expand_dims(   tf.transpose(pool2[0], perm=[2,0,1]),   3), max_outputs=max_output)
        
    with tf.variable_scope('fcn1') as scope:
        reshape = tf.reshape(pool2, [pool2.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        
        weights = tf.get_variable('fcn1', [dim, fcn1_n],
                                  initializer=tf.truncated_normal_initializer(stddev=stdev_init))
        
        biases = tf.get_variable('biases', [fcn1_n],
                                 initializer=tf.constant_initializer(0))
        
        fcn1 = tf.nn.relu(tf.matmul(reshape, weights) + biases)
        
        #h_fc1_drop = tf.nn.dropout(h_fc1,0.5)
        
    with tf.variable_scope('softmax_linear'):
        weights = tf.get_variable('weights',  [fcn1_n, num_class],
                                  initializer=tf.truncated_normal_initializer(stddev=stdev_init))
        
        biases = tf.get_variable('biases', [num_class], initializer=tf.constant_initializer(0))
        logits = tf.matmul(fcn1, weights) + biases
        
    return logits


def back_inference():
    with tf.variable_scope('cnn1', reuse=True) as scope:
        kernel = tf.get_variable(name='kernels')
        biases = tf.get_variable(name='biases') 



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

def softmax(logits):
    return  tf.nn.softmax(logits=logits, name='softmax')
    


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

# mnist = input_data.read_data_sets("mnist_data/")#, one_hot=True
# def gen_mnistimg(batchsize=batch_size,train=True):
#      
#     if train:
#         batch_xs, batch_ys = mnist.train.next_batch(batchsize)
#     else:
#         batch_xs, batch_ys = mnist.test.next_batch(batchsize)
#      
#     batch_xs=batch_xs.reshape([batchsize,28,28,1])
#      
#     #print (mnist.test.images.shape)
#     #print ( mnist.validation.images.shape)
#     '''
#     for ind,i in enumerate(batch_xs):
#         print (batch_ys[ind])
#         cv2.imshow('test',i)
#         cv2.waitKey()
#     '''
#     return batch_xs,batch_ys

    

def gen_rec_circle_images(batchsize=batch_size, imgsize=img_size, channel=1):
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

def index2xy(i=0):#x代表横向，y代表竖向
    tel=int(img_size/cnn1_ksize)
    x=int((i)%tel)*cnn1_ksize
    y=int((i)/tel)*cnn1_ksize
        
    return x,y



def genimages_same(dat, lab):
    #dat,lab=gen_images()#generate new images
    shp=dat.shape
    
    #dat[0,:,:,:]=255#for test 
    #这里求图片均值
    means=np.mean(dat[0], (0,1))
    
    for i in range(1,shp[0]):
        dat[i]=dat[0].copy()
        lab[i]=lab[0]
        
        x,y=index2xy(i-1)
        
        dat[i,y:y+cnn1_ksize, x:x+cnn1_ksize]=[0]*shp[-1]#means #
        
        '''
        cv2.imshow('test',dat[i])
        cv2.waitKey()
        print(dat[i,x:x+cnn1_ksize,y:y+cnn1_ksize])
        '''
    return dat,lab

def test_backinference(sess, softmax_op, eval_op, dat_place, label_place):
    dat,lab=sess.run([images_test, labels_test])#gen_img(train=False)#generate new images
    so_op,evals=sess.run([softmax_op,eval_op], feed_dict={dat_place:dat, label_place:lab})
    
    if lab[0]!=np.argmax(so_op[0]): return
        
    dat,lab=genimages_same(dat,lab)#generate new images生成一批数据 
        
    so_op2,evals2=sess.run([softmax_op,eval_op], feed_dict={dat_place:dat, label_place:lab})
        
    print('right cnt:',evals,'->',evals2,'/',lab.shape[0])
    print('origin:',so_op[0],' lable:',lab[0])
    
    if evals<evals2: return
    
    kep_diff=[]
    for ind,i in enumerate(so_op2):
        #这里应该判断
        tep_dis=i-so_op[0]
        dist = np.linalg.norm(tep_dis)
        
        kep_diff.append(tep_dis[lab[0]])
        
        print (ind,'  test a image:',tep_dis,'  distance:',dist,'   ',np.argmax(i)==lab[ind])
    
    print ('the label kep_diff:',kep_diff)
    
    mean_thre=sum(kep_diff)/len(kep_diff)
    
    ano_dat=dat[0].copy()
    
    cv2.imshow('origin', cv2.cvtColor(cv2.resize(dat[0],(img_size*10,img_size*10), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    
    for ind ,i in enumerate(kep_diff): 
        x,y=index2xy(ind)
        #print(type(x))
        if i > mean_thre:
            dat[0,y:y+cnn1_ksize, x:x+cnn1_ksize]=[255,0,0]
        else:
            ano_dat[y:y+cnn1_ksize, x:x+cnn1_ksize]=[255,0,0]
    
    tepimg=cv2.resize(dat[0],(img_size*10,img_size*10), interpolation=cv2.INTER_CUBIC)
    
    #因为cv里面读入图片默认是bgr，显示时bgr才能正常显示，这里是rgb的图片，要转成bgr才能显示正常
    cv2.imshow('test_>mean', cv2.cvtColor(tepimg, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    #----------------------------------------------------
    tepimg=cv2.resize(ano_dat,(img_size*10,img_size*10), interpolation=cv2.INTER_CUBIC)
    
    #因为cv里面读入图片默认是bgr，显示时bgr才能正常显示，这里是rgb的图片，要转成bgr才能显示正常
    cv2.imshow('test_<mean', cv2.cvtColor(tepimg, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    
    
    
cifar10_dir=r'./cifar10_data/cifar-10-batches-bin'
'''
下面这两个是图操作，利用pipline实现了的数据读取
'''
#训练集
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=cifar10_dir, batch_size=batch_size)
#cifar10_input类中带的distorted_inputs()函数可以产生训练需要的数据，包括特征和label，返回封装好的tensor，每次执行都会生成一个batch_size大小的数据。
#测试集
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir=cifar10_dir, batch_size=batch_size)



def load_model(dirs):
    '''
    graph = tf.get_default_graph()  
       prob_op = graph.get_operation_by_name('prob') # 这个只是获取了operation， 至于有什么用还不知道  
  prediction = graph.get_tensor_by_name('prob:0') # 获取之前prob那个操作的输出，即prediction
    '''
    #fc2 = tf.stop_gradient(fc2)
    graph = tf.get_default_graph() 
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(op.join(dirs,'model_keep-29999.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(dirs))
        
        #启动线程开始填充数据
        coord = tf.train.Coordinator()#this helps manage the threads,but without it ,it still works
        threads = tf.train.start_queue_runners(coord=coord)#
        
        softmax_op=graph.get_tensor_by_name('softmax:0') 
        eval_op=graph.get_tensor_by_name('Sum:0') 
        dat_place=graph.get_tensor_by_name('Placeholder:0') 
        label_place=graph.get_tensor_by_name('Placeholder_1:0') 
        
        
        for i in range(100):
            test_backinference(sess, softmax_op, eval_op, dat_place, label_place)

def start(lr=lr):
    #这里是用placeholder方法的inference，用于后面xiu'gai
    dat_place = tf.placeholder(tf.float32, shape=(batch_size, img_size,img_size,3))
    label_place= tf.placeholder(tf.int32, shape=(batch_size))
    
    
    logits=inference(dat_place)
    softmax_op=softmax(logits)
    
    los=loss(softmax_op, label_place)
    
    train_op=training(los, lr)
    eval_op=evaluate(logits, label_place)
    
    print (softmax_op)
    
    
    
    
    #---------------------!!!!!!!!!!!!!!!!!!!!
    '''
    logits_train=inference(images_train)
    los_train=loss(logits_train, labels_train)
    
    train_op=training(los_train, lr)
    #eval_op=evaluate(logits, label_place)
    softmax_op=softmax(logits_train)
    '''
    #-----------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #合并上面每个summary，就不必一个一个运行了
    merged = tf.summary.merge_all()
    logdir="./logs/cifar10_"+TIMESTAMP+('_cnn1-%d_cnn2-%d_fcn1-%d'%(cnn1_k,cnn2_k, fcn1_n))
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()#初始化tf.Variable
        sess.run(init)
        
        #启动线程开始填充数据
        coord = tf.train.Coordinator()#this helps manage the threads,but without it ,it still works
        threads = tf.train.start_queue_runners(coord=coord)#
        
        #tensorboard里面按文件夹分，这里利用时间分开
        #print ("./logs/"+TIMESTAMP+('_cnn1%d_cnn2%d_fcn1%d'%(cnn1_k,cnn2_k, fcn1_n)))
        writer = tf.summary.FileWriter(logdir,   sess.graph)
        
        '''
        a如果您只想保留4个最新型号，并希望在培训期间每2小时保存一个型号，则可以使用max_to_keep和keep_checkpoint_every_n_hours
        '''
        all_saver = tf.train.Saver(max_to_keep=2) 
        
        sttime=time.time()
        
        for i in range(maxiter):
            
            #print (dat)\
            stt=time.time()
            
            
            dat,lab= sess.run([images_train, labels_train])#gen_img()#generate new images生成一批数据
            _, loss_value, summary_resu , tep= sess.run([train_op, los, merged, logits], feed_dict={dat_place:dat, label_place:lab})
            
            #写入日志
            writer.add_summary(summary_resu, i)
            
            if i%10==0:        #显示一次
                #print (tep)
                print (i, 'time:',time.time()-stt)
                print ('training-loss:',loss_value,'\n')
            
            if (i+1)%300==0:#测试一次
                truecnt=0
                cnt_all=0
                for j in range(int(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/batch_size)):
                    dat,lab=sess.run([images_test, labels_test])#gen_img(train=False)#generate new images
                    
                    eval_resu,loss_value=sess.run([eval_op, los], feed_dict={dat_place:dat, label_place:lab})
                    truecnt+=eval_resu
                    cnt_all+=lab.shape[0]
                    #print (i, 'time:',time.time()-stt)
                    #print ('evaluate-loss:',loss_value,'\n')
                
                print ('!!!!!!!!evaluate:!!!!!!!!!!!!',float(truecnt)/cnt_all,'\n')
                all_saver.save(sess, op.join(logdir,'model_keep'),global_step=i)
        
        coord.request_stop()
        coord.join(threads)
        print('training done! time used:',time.time()-sttime)
        
        
        
        #后面就试下反向
        for i in range(100):
            test_backinference(sess, softmax_op, eval_op, dat_place, label_place)
        
        '''
        dat,lab=gen_images()#generate new images
        so_op,evals=sess.run([softmax_op,eval_op], feed_dict={dat_place:dat, label_place:lab})
        
        dat,lab=genimages_same(dat,lab)#generate new images生成一批数据 
        
        so_op2,evals2=sess.run([softmax_op,eval_op], feed_dict={dat_place:dat, label_place:lab})
        
        print('right cnt:',evals2,'/',lab.shape[0])
        print('origin:',so_op[0],' lable:',lab[0])
        for ind,i in enumerate(so_op2):
            print (ind,' test a image:',i,' ',np.argmax(i)==lab[ind])
            dist = np.linalg.norm(i - so_op[0])
            if dist>0: 
                x,y=index2xy(ind)
                #print(type(x))
                dat[0,y:y+cnn1_ksize, x:x+cnn1_ksize]=[255]
            
        cv2.imshow('test',dat[0])
        cv2.waitKey()
        '''
        
        

if __name__ == '__main__':
    #genimages_same()
    ''''''
    #gen_mnistimg()
    start()
    #back_inference()
    #load_model(r'logs/cifar10_2018-09-21_21-00-50_cnn1-64_cnn2-64_fcn1-1024')
    for i in tf.trainable_variables():
        print (i)
    
        
        
        
        
        
        
    