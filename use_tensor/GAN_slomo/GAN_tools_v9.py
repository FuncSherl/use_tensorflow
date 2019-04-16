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


def my_find_flip(inputdata, inputdata2, filterlen,    scopename, reuse=tf.AUTO_REUSE):
    '''
    filterlen:用这么大的范围内找对称点
    '''
    inputshape=inputdata.get_shape().as_list() #n,h,w,c
    cnt_ind=int ( inputshape[1]*inputshape[2] )
    width=int(inputshape[2] )
    height=int(inputshape[1])
    shifting=int(filterlen/2)
    
    with tf.variable_scope(scopename,  reuse=reuse) as scope: 
        #ret=tf.Variable(np.zeros(shape= inputshape, dtype=np.float32),dtype=tf.float32, trainable=False)
        ret=tf.TensorArray(tf.float32, cnt_ind, element_shape=[inputshape[0], inputshape[-1]])
        
        #ret=tf.get_variable(scope.name+'_ret_var', inputshape, dtype=tf.float32,  initializer=tf.zeros_initializer(), trainable=False)
        #ret=tf.transpose(ret, [1,2,0,3]) #[h,w,n,c]
        #print (ret)
        
        ind=tf.constant(0, dtype=tf.int32)
        loop=[ind,  ret ]
        
        def cond(ind, _):
            return ind<cnt_ind
            
        def body(ind, flow):
            #nonlocal ret
            #ret=tf.Variable(np.zeros(shape=[inputshape[1], inputshape[2], inputshape[0], inputshape[3]], dtype=np.float32))
            
            row=tf.cast( ind/width, tf.int32)
            col=tf.cast( ind%width, tf.int32)
            
            st_row=tf.maximum(row-shifting, 0)
            ed_row=tf.minimum(row+shifting+1, height)
            st_col=tf.maximum(col-shifting, 0)
            ed_col=tf.minimum(col+shifting+1, width)
            
            indata1=inputdata[:, st_row:ed_row, st_col:ed_col, :]
            indata2=inputdata2[:, st_row:ed_row, st_col:ed_col, :]
            
            indata2_left=tf.image.flip_left_right(indata2)
            indata2_up  =tf.image.flip_up_down(indata2)
            indata2_up_left=tf.image.flip_left_right(indata2_up)
            
            indata1_left=tf.abs(indata1-indata2_left)
            indata1_up  =tf.abs(indata1-indata2_up  )
            indata1_up_left=tf.abs(indata1-indata2_up_left)
            
            #indata1_left_expan=tf.expand_dims(indata1_left, 1)
            #indata1_up_expan=tf.expand_dims(indata1_up, 1)
            #indata1_up_left_expan=tf.expand_dims(indata1_up_left, 1)
            
            stack_all=tf.stack([indata1_left, indata1_up, indata1_up_left], 1) #[n,3,h,w,c]
            first_min=tf.reduce_min(stack_all, [1]) #[n,h,w,c]
            sec_min=tf.reduce_min(first_min, [1,2], keep_dims=True) #[n,1,1,c]
            
            min_bool= tf.equal(first_min,sec_min)
            tep=tf.where(min_bool,indata1 , tf.zeros_like(indata1))
            nozerocnt=tf.count_nonzero(tep, [1,2])
            tep=tf.reduce_mean(tep, [1,2])*tf.cast( (ed_row-st_row)*(ed_col-st_col), tf.float32)/tf.cast(nozerocnt, tf.float32) #[n,c]
            
            #ret=tf.concat( [ret, tf.expand_dims(tep, 0)], 0 ) 
            
            #print (st_row)
            #tf.scatter_update()
            
            #print (ret) #Tensor("test/while/Identity_1:0", shape=(32, 32, 12, 3), dtype=float32)
            #ret=tf.scatter_nd_update( ret, [[row, col]], [tf.cast(tep, tf.float32)] )
            '''
            with tf.control_dependencies([flow]):
                flow=tf.assign(ret[:,row, col, :],tep)
            '''
            #flow=tf.cond(tf.equal(ind, 0),  lambda:tf.expand_dims(tep, 0), lambda:tf.concat( [flow, tf.expand_dims(tep, 0)], 0 ) )
            flow=flow.write(ind, tep)
            
            #flow=tf.tile(sec_min, [1, height, width, 1])
            #flow=tf.cond(ind<cnt_ind, lambda: tf.assign(ret[:,row, col, :],tep), lambda:ret)
            
            return tf.add(ind, 1), flow
        
        ind,rett=tf.while_loop(cond, body, loop,  ) #shape_invariants=[tf.TensorShape([]),    tf.TensorShape( [ None, inputshape[0], inputshape[-1] ] )   ] 
        
        rett=rett.stack()
        print (rett) #[h*w, n, c]
        rett=tf.transpose(rett, [1,0,2] )
        rett=tf.reshape(rett, inputshape )
        
        return rett
    
def my_find_flip_no_tensor(inputdata, inputdata2, filterlen,    scopename, reuse=tf.AUTO_REUSE):
    '''
    filterlen:用这么大的范围内找对称点
    '''
    inputshape=inputdata.get_shape().as_list() #n,h,w,c
    cnt_ind=int ( inputshape[1]*inputshape[2] )
    width=int(inputshape[2] )
    height=int(inputshape[1])
    shifting=int(filterlen/2)
    
    kep_tep=[]
    debug=[]
    with tf.variable_scope(scopename,  reuse=reuse) as scope:
        for ind in range(cnt_ind):
            row=int( ind/width)
            col=ind%width
            st_row=max(row-shifting, 0)
            ed_row=min(row+shifting+1, height)
            st_col=max(col-shifting, 0)
            ed_col=min(col+shifting+1, width)
            
            indata1=inputdata[:, st_row:ed_row, st_col:ed_col, :]
            indata2=inputdata2[:, st_row:ed_row, st_col:ed_col, :]
            
            indata2_left=tf.image.flip_left_right(indata2)
            indata2_up  =tf.image.flip_up_down(indata2)
            indata2_up_left=tf.image.flip_left_right(indata2_up)
               
            indata1_left=tf.abs(indata1-indata2_left)
            indata1_up  =tf.abs(indata1-indata2_up  )
            indata1_up_left=tf.abs(indata1-indata2_up_left)
            
            
            stack_all=tf.stack([indata1_left, indata1_up, indata1_up_left], 1) #[n,3,h,w,c]
            first_min=tf.reduce_min(stack_all, [1]) #[n,h,w,c]
            
            sec_min=tf.reduce_min(first_min, [1,2], keep_dims=True) #[n,1,1,c]
            
            #debug.append(sec_min)
                
            min_bool= tf.equal(first_min, sec_min)
            
            #debug.append(min_bool)
            
            tep=tf.where(min_bool,indata1 , tf.zeros_like(indata1))
            
            nozerocnt=tf.count_nonzero(tep, [1,2])
            
            #debug.append(tep)
            
            tep=tf.reduce_mean(tep, [1,2])*tf.cast( (ed_row-st_row)*(ed_col-st_col), tf.float32)/tf.cast(nozerocnt, tf.float32)  #[n,c]
            #debug.append(tep)
            
            kep_tep.append(tep)
            print (ind,'/',cnt_ind,tep)
            
        stack_tep=tf.stack(kep_tep, 1)   #[ n,h*w, c]
        stack_tep=tf.reshape(stack_tep, inputshape) #[n,h,w,c]
        #stack_tep=tf.transpose(stack_tep, [2,0,1,3])
        print (stack_tep)
        return stack_tep
    

def test_my_find_flip():
    A=np.array([[[[1,2,3], \
                [2,1,3],\
                [6,4,2]]]])
    
    C=np.array([[[[1,2,1], \
                [2,6,4],\
                [3,4,2]]]])
    
    B = np.array([[ [[1], [4],[6]],\
              [[7],[10],[9]]  ]])
    
    A=tf.constant(B, dtype=tf.float32)
    C=tf.constant(B, dtype=tf.float32)
    
    
    print (A.shape, B.shape)
    with tf.Session() as sess:  
        
        #sess.run(tf.local_variables_initializer())
        
        tep=my_find_flip(A,C,2,'test')
        #tep=my_find_flip_no_tensor(A,C,2,'test')
        
        sess.run(tf.global_variables_initializer())
        
        print(sess.run(tep)) 
        #print(sess.run(tep)) 
        #print(sess.run(tep)) 
        #print(sess.run(tep)) 



def my_novel_conv(inputdata, inputdata2, filterlen,    scopename, outchannel=None, stride=1, padding="SAME", reuse=tf.AUTO_REUSE, withbias=True):
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
        one_channel=(ori_cnn+left_cnn)/2.0
        ano_channel=(ori_cnn+up_cnn)/2.0
        
        
        if withbias:
            bias=tf.get_variable('bias', [inputshape[-1]], dtype=datatype, initializer=tf.constant_initializer(bias_init))
            one_channel=tf.nn.bias_add(one_channel, bias)
            ano_channel=tf.nn.bias_add(ano_channel, bias)
            
            
        return one_channel,ano_channel
    

def my_novel_unet(inputdata,inputdata2, layercnt=3,  filterlen=3,training=True,  withbias=True):
    '''
    这里将两个输入图片通过同一个特征网络，并保留中间各自特征
    '''
    flipconv_method=my_find_flip
    inputshape=inputdata.get_shape().as_list()
    channel_init=inputshape[-1]
    skipcon1=[]
    skipcon2=[]
    ##########################################################################
    #first unet-down input1:the first frame
    tep=my_conv(inputdata , filterlen+int(layercnt/2), channel_init*2, scopename='unet_down_start', stride=1,  withbias=withbias)
    #tep=my_batchnorm( tep,training, 'trace_1_unet_down_start_bn')
    tep=my_lrelu(tep, 'trace_1_unet_down_start_relu')
    print (tep)
    
    print ('\nforming UNET-->layer1:',layercnt)
    
    for i in range(layercnt):
        skipcon1.append(tep)
        tep=unet_down(tep, channel_init*( 2**(i+2)), 'unet_down_'+str(i), stride=2, filterlen=filterlen+int( (layercnt-i)/2 ), training=training,withbias=withbias)
        print (tep)
        
    input1_fea=tep
    
    #######################################################################   
    #then unet-down the second frame 
    tep=my_conv(inputdata2, filterlen+int(layercnt/2), channel_init*2, scopename='unet_down_start', stride=1,  withbias=withbias)
    #tep=my_batchnorm( tep,training, 'trace_2_unet_down_start_bn')
    tep=my_lrelu(tep, 'trace_2_unet_down_start_relu') 
    print (tep)
    
    print ('\nforming UNET-->layer2:',layercnt)
    
    for i in range(layercnt):
        skipcon2.append(tep)
        tep=unet_down(tep, channel_init*( 2**(i+2)), 'unet_down_'+str(i), stride=2, filterlen=filterlen+int( (layercnt-i)/2 ), training=training,withbias=withbias)
        print (tep)
        
    input2_fea=tep
    
    ##################连接两个部分
    #concating two middle feature
    tep=flipconv_method(input1_fea, input2_fea, filterlen, 'middle_novel_cnn')
    
    print (tep)
    
    ######################################################up
    #begining unet-up 
    for i in reversed(range(layercnt)):
        tep1=skipcon1[i]
        tep2=skipcon2[i]
        
        skipcon=flipconv_method(tep1, tep2, filterlen+int( 2**(layercnt-i) )+2, 'unet_up_novel_cnn_'+str(i))
        
        tep=unet_up(tep, channel_init*( 2**(i+1)), skipcon,'unet_up_'+str(i), stride=2,  filterlen=filterlen+int( (layercnt-i)/3 ),  training=training,withbias=withbias)
        print (tep)
        
    #finish the net 
    tep=my_conv(tep, filterlen, channel_init*2, scopename='unet_up_final_1', stride=1, withbias=withbias)
    tep=my_batchnorm( tep,training, 'unet_up_final_1_bn')
    tep=my_lrelu(tep, 'unet_up_final_1_relu') 
    print (tep)
    
    #finally channel to 3
    tep=my_conv(tep, filterlen, channel_init, scopename='unet_up_final_2', stride=1, withbias=withbias)
    
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
    test_my_find_flip()
    
                
                
                
                
                
                
                
                