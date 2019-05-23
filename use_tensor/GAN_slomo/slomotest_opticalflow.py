'''
Created on Apr 9, 2019

@author: sherl
'''

import tensorflow as tf
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import cv2,os,time
from datetime import datetime

modelpath="/home/sherl/Pictures/v22_GAN_2019-05-21_21-38-30_base_lr-0.000200_batchsize-12_maxstep-240000_selectChannelWillLeadToNoise"
#modelpath=r'/home/sherl/Pictures/v20_GAN_2019-05-13_19-24-10_base_lr-0.000200_batchsize-12_maxstep-240000'
meta_name=r'model_keep-239999.meta'

TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

testvideodir='./testing_gif'
inputvideo =op.join(testvideodir, 'original.mp4')
outputvideo=op.join( testvideodir, TIMESTAMP+'_myslomo.avi')
os.makedirs(testvideodir,  exist_ok=True)

class Slomo_flow:
    def __init__(self,sess):
        self.sess=sess
        saver = tf.train.import_meta_graph(op.join(modelpath, meta_name) )
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        
        # get weights
        self.graph = tf.get_default_graph()
        self.outimg = self.graph.get_tensor_by_name("G_net_generate:0")
        self.optical_t_0=self.graph.get_tensor_by_name("add:0")
        self.optical_t_2=self.graph.get_tensor_by_name("add_1:0")
        self.optical_0_1=self.graph.get_tensor_by_name("G_opticalflow_0_2:0")
        self.optical_1_0=self.graph.get_tensor_by_name("G_opticalflow_2_0:0")
        
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
    
    def getflow_to_frames(self, frame0, frame2, cnt):
        '''
        #这里先resize光流，在合成帧，保持原视频分辨率
        '''
        if cnt>self.batch: 
            print ('error:insert frames cnt should <= batchsize:',self.batch)
            return None
        fshape=frame0.shape
        resize_sha=(fshape[1], fshape[0]) #width,height
        timerates=[i*1.0/(cnt+1) for i in range(1,self.batch+1)]
        placetep=np.zeros(self.placeimgshape)
        for i in range(cnt):
            placetep[i,:,:,:3]=cv2.resize(frame0, self.imgshape)
            placetep[i,:,:,6:]=cv2.resize(frame2, self.imgshape)
        
        placetep=self.img2tanh(placetep)
        
        flowt_0,flowt_2, flow0_1, flow1_0=self.sess.run([self.optical_t_0, self.optical_t_2, self.optical_0_1, self.optical_1_0], feed_dict={  self.img_pla:placetep , self.training:False, self.timerates:timerates})
        
        X, Y = np.meshgrid(np.arange(fshape[1]), np.arange(fshape[0]))  #w,h
        xy=np.array( np.stack([Y,X], -1), dtype=np.float32)
        
        #out[x,y]=src[mapx[x,y], mapy[x,y]] or  map[x,y]
        #print (flowt_0.shape,xy.shape)
        out=[]
        for i in range(cnt):
            tep0=xy+cv2.resize(flowt_0[i], resize_sha)
            tep1=xy+cv2.resize(flowt_2[i], resize_sha)
            
            tep0=tep0.astype(np.float32)
            tep1=tep1.astype(np.float32)
            
            #print (tep0)
            #print (tep0[1,2])
            
            #实验中如果是xy=np.array( np.stack([X，Y], -1), dtype=np.float32)，
            #直接将xy作为map送入remap函数中指定映射，则输出和原图一摸一样的图像，要知道前面xy里的[i,j]处对应的值为[j,i],这是一个cv2里的大坑，即map里的（x，y）分别为宽和高，而不是行列坐标 
            
            tepframe0=cv2.remap(frame0, tep0[:,:,1], tep0[:,:,0],  interpolation=cv2.INTER_LINEAR)
            tepframe1=cv2.remap(frame2, tep1[:,:,1], tep1[:,:,0],  interpolation=cv2.INTER_LINEAR)
            #print (tepframe0[1,2])
            final=tepframe1*timerates[i]+(1-timerates[i])*tepframe0
            #final=tepframe1
            out.append(final)
        out=np.array(out, dtype=np.uint8)
    
        return out
    
    
    def process_video(self, interpola_cnt=7, inpath=inputvideo, outpath=outputvideo, keep_shape=True):
        videoCapture = cv2.VideoCapture(inpath)  
        
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps=int (videoCapture.get(cv2.CAP_PROP_FPS) )
        frame_cnt=videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
        
        print ('video:',inpath)
        print ('size:',size, '  fps:',fps,'  frame_cnt:',frame_cnt)
        
        if not keep_shape:
            videoWrite = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int (fps), self.videoshape )
            print ('output video:',outputvideo,'\nsize:',self.videoshape, '  fps:', fps)
        else:
            videoWrite = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int (fps), size )
            print ('output video:',outputvideo,'\nsize:',size, '  fps:', fps)
        
        success, frame0= videoCapture.read()
        if not keep_shape: frame0=cv2.resize(frame0, self.videoshape)
        
        success, frame1= videoCapture.read()
        
        
        cnt=0
        while success and (frame1 is not None):
            if frame0 is not None: videoWrite.write(frame0)
            
            if not keep_shape: frame1=cv2.resize(frame1, self.videoshape)
            
            sttime=time.time()              
            
            if not keep_shape: outimgs=self.getframes_throw_flow(frame0, frame1, interpola_cnt)
            else: outimgs=self.getflow_to_frames(frame0, frame1, interpola_cnt)
            
            print ('get iner frame shape:',outimgs.shape, outimgs.dtype)
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
    
    def flow_bgr(self, flow):
        # Use Hue, Saturation, Value colour model 
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        '''
        cv2.imshow("colored flow", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return bgr


with tf.Session() as sess:
    slomo=Slomo_flow(sess)
    slomo.process_video(12, inputvideo, outputvideo, keep_shape=False)
    
         
    