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

if __name__ == '__main__':
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
        
        
        
        
        

