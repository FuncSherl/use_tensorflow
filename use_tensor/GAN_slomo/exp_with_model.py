'''
Created on Mar 1, 2019

@author: root
'''

import tensorflow as tf
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import cv2

modelpath=r'/home/sherl/git/use_tensorflow/use_tensor/GAN_slomo/logs_v3/GAN_2019-02-25_15-28-42_base_lr-0.000200_batchsize-16_maxstep-160000'

'''
first show G params
0 <tf.Variable 'G_Net/unet_down_start/weights:0' shape=(5, 5, 6, 12) dtype=float32_ref>
1 <tf.Variable 'G_Net/unet_down_start/bias:0' shape=(12,) dtype=float32_ref>
2 <tf.Variable 'G_Net/unet_down_0_conv1/weights:0' shape=(5, 5, 12, 24) dtype=float32_ref>
3 <tf.Variable 'G_Net/unet_down_0_conv1/bias:0' shape=(24,) dtype=float32_ref>
4 <tf.Variable 'G_Net/unet_down_0_conv2/weights:0' shape=(5, 5, 24, 24) dtype=float32_ref>
5 <tf.Variable 'G_Net/unet_down_0_conv2/bias:0' shape=(24,) dtype=float32_ref>
6 <tf.Variable 'G_Net/unet_down_1_conv1/weights:0' shape=(5, 5, 24, 48) dtype=float32_ref>
7 <tf.Variable 'G_Net/unet_down_1_conv1/bias:0' shape=(48,) dtype=float32_ref>
8 <tf.Variable 'G_Net/unet_down_1_conv2/weights:0' shape=(5, 5, 48, 48) dtype=float32_ref>
9 <tf.Variable 'G_Net/unet_down_1_conv2/bias:0' shape=(48,) dtype=float32_ref>
10 <tf.Variable 'G_Net/unet_down_2_conv1/weights:0' shape=(4, 4, 48, 96) dtype=float32_ref>
11 <tf.Variable 'G_Net/unet_down_2_conv1/bias:0' shape=(96,) dtype=float32_ref>
12 <tf.Variable 'G_Net/unet_down_2_conv2/weights:0' shape=(4, 4, 96, 96) dtype=float32_ref>
13 <tf.Variable 'G_Net/unet_down_2_conv2/bias:0' shape=(96,) dtype=float32_ref>
14 <tf.Variable 'G_Net/unet_up_2_deconv1/weights:0' shape=(4, 4, 48, 96) dtype=float32_ref>
15 <tf.Variable 'G_Net/unet_up_2_deconv1/bias:0' shape=(48,) dtype=float32_ref>
16 <tf.Variable 'G_Net/unet_up_2_conv1/weights:0' shape=(4, 4, 48, 48) dtype=float32_ref>
17 <tf.Variable 'G_Net/unet_up_2_conv1/bias:0' shape=(48,) dtype=float32_ref>
18 <tf.Variable 'G_Net/unet_up_1_deconv1/weights:0' shape=(4, 4, 24, 48) dtype=float32_ref>
19 <tf.Variable 'G_Net/unet_up_1_deconv1/bias:0' shape=(24,) dtype=float32_ref>
20 <tf.Variable 'G_Net/unet_up_1_conv1/weights:0' shape=(4, 4, 24, 24) dtype=float32_ref>
21 <tf.Variable 'G_Net/unet_up_1_conv1/bias:0' shape=(24,) dtype=float32_ref>
22 <tf.Variable 'G_Net/unet_up_0_deconv1/weights:0' shape=(5, 5, 12, 24) dtype=float32_ref>
23 <tf.Variable 'G_Net/unet_up_0_deconv1/bias:0' shape=(12,) dtype=float32_ref>
24 <tf.Variable 'G_Net/unet_up_0_conv1/weights:0' shape=(5, 5, 12, 12) dtype=float32_ref>
25 <tf.Variable 'G_Net/unet_up_0_conv1/bias:0' shape=(12,) dtype=float32_ref>
26 <tf.Variable 'G_Net/unet_up_end/weights:0' shape=(4, 4, 12, 3) dtype=float32_ref>
27 <tf.Variable 'G_Net/unet_up_end/bias:0' shape=(3,) dtype=float32_ref>

next is D:

0 <tf.Variable 'D_1_Net/D_1_Net_start/weights:0' shape=(5, 5, 9, 18) dtype=float32_ref>
1 <tf.Variable 'D_1_Net/D_1_Net_start/bias:0' shape=(18,) dtype=float32_ref>
2 <tf.Variable 'D_1_Net/D_1_Net_Dblock0_conv1/weights:0' shape=(5, 5, 18, 36) dtype=float32_ref>
3 <tf.Variable 'D_1_Net/D_1_Net_Dblock0_conv1/bias:0' shape=(36,) dtype=float32_ref>
4 <tf.Variable 'D_1_Net/D_1_Net_Dblock0/beta:0' shape=(36,) dtype=float32_ref>
5 <tf.Variable 'D_1_Net/D_1_Net_Dblock0/gamma:0' shape=(36,) dtype=float32_ref>
6 <tf.Variable 'D_1_Net/D_1_Net_Dblock1_conv1/weights:0' shape=(4, 4, 36, 72) dtype=float32_ref>
7 <tf.Variable 'D_1_Net/D_1_Net_Dblock1_conv1/bias:0' shape=(72,) dtype=float32_ref>
8 <tf.Variable 'D_1_Net/D_1_Net_Dblock1/beta:0' shape=(72,) dtype=float32_ref>
9 <tf.Variable 'D_1_Net/D_1_Net_Dblock1/gamma:0' shape=(72,) dtype=float32_ref>
10 <tf.Variable 'D_1_Net/D_1_Net_Dblock2_conv1/weights:0' shape=(4, 4, 72, 144) dtype=float32_ref>
11 <tf.Variable 'D_1_Net/D_1_Net_Dblock2_conv1/bias:0' shape=(144,) dtype=float32_ref>
12 <tf.Variable 'D_1_Net/D_1_Net_Dblock2/beta:0' shape=(144,) dtype=float32_ref>
13 <tf.Variable 'D_1_Net/D_1_Net_Dblock2/gamma:0' shape=(144,) dtype=float32_ref>
14 <tf.Variable 'D_1_Net/D_1_Net_Dblock3_conv1/weights:0' shape=(3, 3, 144, 288) dtype=float32_ref>
15 <tf.Variable 'D_1_Net/D_1_Net_Dblock3_conv1/bias:0' shape=(288,) dtype=float32_ref>
16 <tf.Variable 'D_1_Net/D_1_Net_Dblock3/beta:0' shape=(288,) dtype=float32_ref>
17 <tf.Variable 'D_1_Net/D_1_Net_Dblock3/gamma:0' shape=(288,) dtype=float32_ref>
18 <tf.Variable 'D_1_Net/D_1_Net_fc1/weights:0' shape=(4320, 1024) dtype=float32_ref>
19 <tf.Variable 'D_1_Net/D_1_Net_fc1/bias:0' shape=(1024,) dtype=float32_ref>
20 <tf.Variable 'D_1_Net/D_1_Net_fc2/weights:0' shape=(1024, 1) dtype=float32_ref>
21 <tf.Variable 'D_1_Net/D_1_Net_fc2/bias:0' shape=(1,) dtype=float32_ref>
22 <tf.Variable 'D_2_Net/D_2_Net_start/weights:0' shape=(5, 5, 3, 6) dtype=float32_ref>
23 <tf.Variable 'D_2_Net/D_2_Net_start/bias:0' shape=(6,) dtype=float32_ref>
24 <tf.Variable 'D_2_Net/D_2_Net_Dblock0_conv1/weights:0' shape=(5, 5, 6, 12) dtype=float32_ref>
25 <tf.Variable 'D_2_Net/D_2_Net_Dblock0_conv1/bias:0' shape=(12,) dtype=float32_ref>
26 <tf.Variable 'D_2_Net/D_2_Net_Dblock0/beta:0' shape=(12,) dtype=float32_ref>
27 <tf.Variable 'D_2_Net/D_2_Net_Dblock0/gamma:0' shape=(12,) dtype=float32_ref>
28 <tf.Variable 'D_2_Net/D_2_Net_Dblock1_conv1/weights:0' shape=(4, 4, 12, 24) dtype=float32_ref>
29 <tf.Variable 'D_2_Net/D_2_Net_Dblock1_conv1/bias:0' shape=(24,) dtype=float32_ref>
30 <tf.Variable 'D_2_Net/D_2_Net_Dblock1/beta:0' shape=(24,) dtype=float32_ref>
31 <tf.Variable 'D_2_Net/D_2_Net_Dblock1/gamma:0' shape=(24,) dtype=float32_ref>
32 <tf.Variable 'D_2_Net/D_2_Net_Dblock2_conv1/weights:0' shape=(4, 4, 24, 48) dtype=float32_ref>
33 <tf.Variable 'D_2_Net/D_2_Net_Dblock2_conv1/bias:0' shape=(48,) dtype=float32_ref>
34 <tf.Variable 'D_2_Net/D_2_Net_Dblock2/beta:0' shape=(48,) dtype=float32_ref>
35 <tf.Variable 'D_2_Net/D_2_Net_Dblock2/gamma:0' shape=(48,) dtype=float32_ref>
36 <tf.Variable 'D_2_Net/D_2_Net_Dblock3_conv1/weights:0' shape=(3, 3, 48, 96) dtype=float32_ref>
37 <tf.Variable 'D_2_Net/D_2_Net_Dblock3_conv1/bias:0' shape=(96,) dtype=float32_ref>
38 <tf.Variable 'D_2_Net/D_2_Net_Dblock3/beta:0' shape=(96,) dtype=float32_ref>
39 <tf.Variable 'D_2_Net/D_2_Net_Dblock3/gamma:0' shape=(96,) dtype=float32_ref>
40 <tf.Variable 'D_2_Net/D_2_Net_fc1/weights:0' shape=(1440, 1024) dtype=float32_ref>
41 <tf.Variable 'D_2_Net/D_2_Net_fc1/bias:0' shape=(1024,) dtype=float32_ref>
42 <tf.Variable 'D_2_Net/D_2_Net_fc2/weights:0' shape=(1024, 1) dtype=float32_ref>
43 <tf.Variable 'D_2_Net/D_2_Net_fc2/bias:0' shape=(1,) dtype=float32_ref>
'''
'''
img=np.zeros([180,360,3],dtype=np.uint8)
img=cv2.putText(img,'there 0 error(s):',(0,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
cv2.imshow('t',img)
cv2.waitKey()
'''


'''
class FizzBuzz():
    def __init__(self, length=30):
        self.length = length  # 程序需要执行的序列长度
        self.array = tf.Variable([[i,i] for i in range(1, length+1)], dtype=tf.float32, trainable=False)  # 最后程序返回的结果
        self.graph = tf.while_loop(self.cond, self.body, [1, self.array],)   # 对每一个值进行循环判断

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run([self.graph])    

    def cond(self, i, _):
        return (tf.less(i, self.length+1)) # 判断是否是最后一个值

    def body(self, i, flow):
        #flow = tf.cond( tf.equal(tf.mod(i, 5), 0),     lambda: tf.assign(self.array[i - 1], [tf.cast(i, tf.float32), 0]  ),lambda: self.array,  )
        with tf.control_dependencies([flow]):
            flow=tf.cond(tf.equal(tf.mod(i, 5), 0),lambda: self.array,    lambda:tf.assign(self.array[i - 1], [tf.cast(i, tf.float32), 0]  ) )
        return (tf.add(i, 1), flow)

if __name__ == '__main__':
    fizzbuzz = FizzBuzz(length=60)
    print(fizzbuzz.run())
    print(fizzbuzz.run())
    print(fizzbuzz.run())
    



#https://zhuanlan.zhihu.com/p/31958302

class FizzBuzz2():
    def __init__(self, inputdata, inputdata2, filterlen,    scopename, reuse=tf.AUTO_REUSE):
        inputshape=inputdata.get_shape().as_list() #n,h,w,c
        self.cnt_ind=int ( inputshape[1]*inputshape[2] )
        self.width=int(inputshape[2] )
        self.height=int(inputshape[1])
        self.shifting=int(filterlen/2)
        self.ind=tf.constant(0, dtype=tf.int32)
        self.inputdata=inputdata
        self.inputdata2=inputdata2
        
        self.array = tf.get_variable('ret_var', inputshape, dtype=tf.float32,  initializer=tf.zeros_initializer(), trainable=False)
        loop=[self.ind, self.array]
        
        self.graph = tf.while_loop(self.cond, self.body, loop)   # 对每一个值进行循环判断

    def run(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.graph)    

    def cond(self, ind, _):
        return ind<self.cnt_ind

    def body(self, ind, flow):
        row=tf.cast( ind/self.width, tf.int32)
        col=ind%self.width
        st_row=tf.maximum(row-self.shifting, 0)
        ed_row=tf.minimum(row+self.shifting+1, self.height)
        st_col=tf.maximum(col-self.shifting, 0)
        ed_col=tf.minimum(col+self.shifting+1, self.width)
            
        indata1=self.inputdata[:, st_row:ed_row, st_col:ed_col, :]
        indata2=self.inputdata2[:, st_row:ed_row, st_col:ed_col, :]
            
        indata2_left=tf.image.flip_left_right(indata2)
        indata2_up  =tf.image.flip_up_down(indata2)
        indata2_up_left=tf.image.flip_left_right(indata2_up)
            
        indata1_left=tf.abs(indata1-indata2_left)
        indata1_up  =tf.abs(indata1-indata2_up  )
        indata1_up_left=tf.abs(indata1-indata2_up_left)
            
            
            
        stack_all=tf.stack([indata1_left, indata1_up, indata1_up_left], 1) #[n,3,h,w,c]
        first_min=tf.reduce_min(stack_all, [1]) #[n,h,w,c]
        sec_min=tf.reduce_min(first_min, [1,2], keep_dims=True) #[n,1,1,c]
            
        min_bool= tf.equal(first_min, sec_min)
        tep=tf.where(min_bool,indata1 , tf.zeros_like(indata1))
        nozerocnt=tf.count_nonzero(tep, [1,2])
        tep=tf.reduce_mean(tep, [1,2])*tf.cast( (ed_row-st_row)*(ed_col-st_col), tf.float32)/tf.cast(nozerocnt, tf.float32) #[n,c]
            
        with tf.control_dependencies([flow]):
            flow=tf.assign(self.array[:,row, col, :],tep)
            #flow=tf.cond(  tf.less(ind, self.cnt_ind), lambda: tf.assign(self.array[:,row, col, :],tep), lambda:self.array)
            
        return (tf.add(ind, 1), flow)

if __name__ == '__main__':
    B = np.array([[ [[1,2,3], [4,5,6],[6,5,4]],\
              [[7,8,9],[10,11,12],[9,7,4]]  ]])
    
    A=tf.constant(B, dtype=tf.float32)
    C=tf.constant(B, dtype=tf.float32)
    
    print (B.shape)
    
    fizzbuzz = FizzBuzz2(A,C,2,'test')
    print(fizzbuzz.run())
    print(fizzbuzz.run())
    print(fizzbuzz.run())
    print(fizzbuzz.run())
    
'''

a = tf.Variable(np.ones([3,2,3]))
index_a = tf.where(a>0)
print (index_a)
tep=tf.gather_nd(a, index_a)
print (tep)
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tep))



if __name__ == '__main__...':
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(op.join(modelpath, r'model_keep-159999.meta') )
        saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        
        # get weights
        graph = tf.get_default_graph()
        D_1_w = graph.get_tensor_by_name("D_1_Net/D_1_Net_start/weights:0")
        
        tep=sess.run(D_1_w)
         
        for i in range(3):
            print (tep[:,:,i*3:(i+1)*3,0])
            cv2.imshow('test',tep[:,:,i*3:(i+1)*3,0])
            cv2.waitKey()
        
        
        
        
        

