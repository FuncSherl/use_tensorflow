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
inputvideo ='./testing_gif/original.mp4'
outputvideo='./testing_gif/myslomo.avi'


class Slomo:
    def __init__(self,sess):
        self.sess=sess
        saver = tf.train.import_meta_graph(op.join(modelpath, meta_name) )
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        
        # get weights
        self.graph = tf.get_default_graph()
        self.outimg = self.graph.get_tensor_by_name("G_Net/G_tanh:0")
        
        self.img_pla= self.graph.get_tensor_by_name('imgs_in:0')
        self.training= self.graph.get_tensor_by_name("training_in:0")
        
        self.placeimgshape=self.img_pla.get_shape().as_list() #[12, 180, 320, 9]
        self.imgshape=(self.placeimgshape[2], self.placeimgshape[1]) #w*h
        
        self.outimgshape=self.outimg.get_shape().as_list() #self.outimgshape: [12, 180, 320, 3]
        self.videoshape=(self.outimgshape[2], self.outimgshape[1]) #w*h
        
    def getframe_inbetween(self,frame0,frame2):          
        frame0=cv2.resize(frame0, self.imgshape)
        frame2=cv2.resize(frame2, self.imgshape)
              
        placetep=np.zeros(self.placeimgshape)
        print (placetep.shape)
        
        placetep[0,:,:,:3]=frame0
        placetep[0,:,:,6:]=frame2
        placetep=self.img2tanh(placetep)
        ########################################
        
        out=self.sess.run(self.outimg, feed_dict={  self.img_pla:placetep , self.training:False})[0]
        print (out.shape)
        
        ##################################
        out=self.tanh2img(out)
        return out
    
    def process_video(self, inpath=inputvideo, outpath=outputvideo):
        videoCapture = cv2.VideoCapture(inpath)  
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps=int (videoCapture.get(cv2.CAP_PROP_FPS) )
        
        frame_cnt=videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
        
        print ('original video: size:',size, '  fps:', fps)
        
        videoWrite = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, self.videoshape )
        print ('output video:',outputvideo,'\nsize:',self.videoshape, '  fps:', fps)
        
        success, frame0= videoCapture.read()
        success, frame1= videoCapture.read()
        cnt=2
        while success and (frame1 is not None):
            print (cnt,'/',frame_cnt)
            tepimg=self.getframe_inbetween(frame0, frame1)
            videoWrite.write(frame0)
            videoWrite.write(tepimg)
            
            frame0=frame1
            cnt+=1
            success, frame1= videoCapture.read()
        
        videoWrite.write(frame0)
        
        videoWrite.release()
        videoCapture.release()
        
        
    

    def img2tanh(self,img):
        #img=tf.cast(img,tf.float32)
        return img*2.0/255-1

    def tanh2img(self,tanhd):
        tep= (tanhd+1)*255//2
        return tep.astype(np.uint8)  


with tf.Session() as sess:
    slomo=Slomo(sess)
    slomo.process_video(inputvideo, outputvideo)
    
         
    