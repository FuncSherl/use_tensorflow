'''
Created on Apr 9, 2019

@author: sherl
'''

import tensorflow as tf
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import cv2

modelpath=r'/home/sherl/workspaces/git/use_tensorflow/use_tensor/GAN_slomo/logs_v8/GAN_2019-04-04_20-17-08_base_lr-0.000200_batchsize-12_maxstep-240000'
meta_name=r'model_keep-239999.meta'

def tanh2img(tanhd):
    tep= (tanhd+1)*255//2
    return tep.astype(np.uint8)  


with tf.Session() as sess:
    saver = tf.train.import_meta_graph(op.join(modelpath, meta_name) )
    saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        
    # get weights
    graph = tf.get_default_graph()
    outimg = graph.get_tensor_by_name("G_Net/G_tanh:0")
    
    tep=sess.run(outimg)
         
    