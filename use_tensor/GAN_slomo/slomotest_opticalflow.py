'''
Created on Apr 9, 2019

@author: sherl
'''

import tensorflow as tf
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import cv2,os,time

modelpath=r'/home/sherl/Pictures/v20_GAN_2019-05-12_15-44-41_base_lr-0.000200_batchsize-12_maxstep-240000'
meta_name=r'model_keep-239999.meta'

testvideodir='./testing_gif'
inputvideo =op.join(testvideodir, 'original.mp4')
outputvideo=op.join( testvideodir, 'myslomo.avi')
os.makedirs(testvideodir,  exist_ok=True)

class Slomo_flow:
    def __init__(self,sess):
        self.sess=sess
        saver = tf.train.import_meta_graph(op.join(modelpath, meta_name) )
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        
        # get weights
        self.graph = tf.get_default_graph()
        self.outimg = self.graph.get_tensor_by_name("add_2:0")
        
        self.img_pla= self.graph.get_tensor_by_name('imgs_in:0')
        self.training= self.graph.get_tensor_by_name("training_in:0")
        self.timerates= self.graph.get_tensor_by_name("timerates_in:0")
        
        
        self.placeimgshape=self.img_pla.get_shape().as_list() #[12, 180, 320, 9]
        self.batch=self.placeimgshape[0]
        self.imgshape=(self.placeimgshape[2], self.placeimgshape[1]) #w*h
        
        self.outimgshape=self.outimg.get_shape().as_list() #self.outimgshape: [12, 180, 320, 3]
        self.videoshape=(self.outimgshape[2], self.outimgshape[1]) #w*h
        
    def getframes_throw_flow(self, frame0, frame2, cnt):
        '''
        cnt:中间插入几帧
        '''
        if cnt>self.batch: 
            print ('error:insert frames cnt should <= batchsize:',self.batch)
            return None
            
        timerates=[i*1.0/(cnt+1) for i in range(1,self.batch+1)]
        
        frame0=cv2.resize(frame0, self.imgshape)
        frame2=cv2.resize(frame2, self.imgshape)
        
        placetep=np.zeros(self.placeimgshape)
        for i in range(cnt):
            placetep[i,:,:,:3]=frame0
            placetep[i,:,:,6:]=frame2
        
        placetep=self.img2tanh(placetep)
        out=self.sess.run(self.outimg, feed_dict={  self.img_pla:placetep , self.training:False, self.timerates:timerates})
        return self.tanh2img(out[:cnt])
        

    
    def process_video(self, interpola_cnt=7, inpath=inputvideo, outpath=outputvideo):
        videoCapture = cv2.VideoCapture(inpath)  
        
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps=int (videoCapture.get(cv2.CAP_PROP_FPS) )
        frame_cnt=videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
        
        print ('video:',inpath)
        print ('size:',size, '  fps:',fps,'  frame_cnt:',frame_cnt)
        
        videoWrite = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int (fps), self.videoshape )
        print ('output video:',outputvideo,'\nsize:',self.videoshape, '  fps:', fps)
        
        
        success, frame0= videoCapture.read()
        frame0=cv2.resize(frame0, self.videoshape)
        
        success, frame1= videoCapture.read()
        frame1=cv2.resize(frame1, self.videoshape)
        
        cnt=0
        while success and (frame1 is not None):
            if frame0 is not None: videoWrite.write(frame0)
            sttime=time.time()              
            outimgs=self.getframes_throw_flow(frame0, frame1, interpola_cnt)
            
            print (outimgs.shape)
            for i in outimgs:      
                print (i.shape) 
                videoWrite.write(i)
            #cv2.imshow('t', tepimg)
            #cv2.waitKey()
            
            frame0=frame1
            cnt+=1
            print (cnt,'/',frame_cnt,'  time gap:',time.time()-sttime)
            success, frame1= videoCapture.read()
            
        videoWrite.write(frame0)
        
        videoWrite.release()
        videoCapture.release()
        self.show_video_info( outpath)
        
    def show_video_info(self, inpath):
        videoCapture = cv2.VideoCapture(inpath)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps=int (videoCapture.get(cv2.CAP_PROP_FPS) )
        
        frame_cnt=videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
        
        print ('video:',inpath)
        print ('size:',size, '  fps:',fps,'  frame_cnt:',frame_cnt)
        
        videoCapture.release()
        return size, fps, frame_cnt
    

    def img2tanh(self,img):
        #img=tf.cast(img,tf.float32)
        return img*2.0/255-1

    def tanh2img(self,tanhd):
        tep= (tanhd+1)*255//2
        return tep.astype(np.uint8)  


with tf.Session() as sess:
    slomo=Slomo_flow(sess)
    slomo.process_video(12, inputvideo, outputvideo)
    
         
    