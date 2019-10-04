'''
Created on Sep 27, 2019

@author: sherl
'''
import tensorflow as tf
from datetime import datetime
import os.path as op
from data import create_dataset_step2 as cdata
import numpy as np
import cv2, os, random, time

print ('tensorflow version:',tf.__version__,'  path:',tf.__path__)
TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
#图像形状 
flow_channel = 2

#kernel concern
output_channel = 12
batchsize = 1  # 这里一定为1，因为下面会将第一部的输出里面的batchsize当作timestep使用，这里的batchsize默认就是1了
kernel_len = 3


#数据输入的样式，一套3个图，每个图3channel，前后帧和t时刻帧 
group_img_num=3
img_channel=3


timestep = 12  #这里与第一部中的batchsize相同

#hyper param 
dropout=0.5 
leakyrelurate=0.2
stddev=0.01
bias_init=0.0
LR=1e-3
maxstep=240000 #训练多少次

mean_dataset=[102.1, 109.9, 110.0]  #0->1 is [0.4, 0.43, 0.43]
modelpath=cdata.modelpath
meta_name = r'model_keep-239999.meta'

# 设置GPU显存按需增长
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)

logdir="./logs_v24_Step2/GAN_"+TIMESTAMP+('_base_lr-%f_batchsize-%d_maxstep-%d'%(LR,batchsize, maxstep))
kepimgdir=op.join(logdir, "zerostateimgs")
os.makedirs(kepimgdir,  exist_ok=True)

class Step2_ConvLstm:

    def __init__(self, sess):
        self.sess = sess

        #加载原模型
        saver = tf.train.import_meta_graph(op.join(modelpath, meta_name) )
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        self.graph = tf.get_default_graph()
        
        #placeholders
        self.imgs_pla= self.graph.get_tensor_by_name('imgs_in:0')
        self.training= self.graph.get_tensor_by_name("training_in:0")
        self.timerates= self.graph.get_tensor_by_name("timerates_in:0")
        
        #tesorfs
        self.optical_0_1=self.graph.get_tensor_by_name("G_opticalflow_0_2:0")
        self.optical_1_0=self.graph.get_tensor_by_name("G_opticalflow_2_0:0")
        self.outimg = self.graph.get_tensor_by_name("G_net_generate:0")
        
        # 注意这里构建lstm的输入是根据第一部的输出直接来的
        #self.input_pla = tf.placeholder(tf.float32, [batchsize,  flow_size_h, flow_size_w, input_channel], name='step2_opticalflow_in')
        self.input_pla=tf.concat([self.optical_0_1, self.optical_1_0], -1)  #这里将两个光流拼起来 可以考虑将前后帧也拼起来
        self.lstm_input_channel=self.input_pla.get_shape().as_list()[-1]
        print (self.input_pla)  #Tensor("concat_9:0", shape=(12, 180, 320, 4), dtype=float32)
        
        #第一部中的batchsize，这里可以当作timestep使用
        self.batchsize_inputimg=self.imgs_pla.get_shape().as_list()[0]
        
        #输入图像形状
        self.img_size_w=self.imgs_pla.get_shape().as_list()[2]
        self.img_size_h=self.imgs_pla.get_shape().as_list()[1]
        self.img_size=[self.img_size_h, self.img_size_w]
        print (self.imgs_pla) #Tensor("imgs_in:0", shape=(12, 180, 320, 9), dtype=float32)
        
        self.pipline_data_train=cdata.get_pipline_data_train(self.img_size, self.batchsize_inputimg)
        self.pipline_data_test=cdata.get_pipline_data_test(self.img_size, self.batchsize_inputimg)
        #输出光流形状
        self.flow_size_h=self.optical_0_1.get_shape().as_list()[1]
        self.flow_size_w=self.optical_0_1.get_shape().as_list()[2]
        self.flow_channel=self.optical_0_1.get_shape().as_list()[-1]
        
        #lstm的每个状态（c，h）的形状
        self.state_shape=[batchsize, self.flow_size_h, self.flow_size_w, output_channel]
        
        #state placeholder
        self.state_pla_c = tf.placeholder(tf.float32, self.state_shape, name='step2_state_in_c')
        self.state_pla_h = tf.placeholder(tf.float32, self.state_shape, name='step2_state_in_h')
        
        #self.imgs_pla = tf.placeholder(tf.float32, [batchsize, img_size_h, img_size_w, group_img_num*img_channel], name='step2_oriimgs_in')
        self.frame0=self.imgs_pla[:,:,:,:img_channel]
        self.frame1=self.imgs_pla[:,:,:,img_channel:img_channel*2]
        self.frame2=self.imgs_pla[:,:,:,img_channel*2:]
        
        #self.timerates_pla=tf.placeholder(tf.float32, [batchsize], name='step2_inner_timerates_in')
        self.timerates_expand=tf.expand_dims(self.timerates, -1)
        self.timerates_expand=tf.expand_dims(self.timerates_expand, -1)
        self.timerates_expand=tf.expand_dims(self.timerates_expand, -1) #batchsize*1*1*1
        
        self.cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[self.flow_size_h, self.flow_size_w, self.lstm_input_channel], \
                                                output_channels=output_channel, kernel_shape=[kernel_len, kernel_len])
        
        self.state_init = self.cell.zero_state(batch_size=batchsize, dtype=tf.float32)
        self.state_init_np=( np.zeros(self.state_shape, dtype=np.float32), np.zeros(self.state_shape, dtype=np.float32) )
        print (self.state_init) #LSTMStateTuple(c=<tf.Tensor 'ConvLSTMCellZeroState/zeros:0' shape=(2, 180, 320, 12) dtype=float32>, h=<tf.Tensor 'ConvLSTMCellZeroState/zeros_1:0' shape=(2, 180, 320, 12) dtype=float32>)
        self.state_new_train=self.state_init_np
        self.state_new_test=self.state_init_np
        #这里开始搞lstm了
        self.input_dynamic_lstm=tf.expand_dims(self.input_pla, 0)   #这里默认lstm的输入batchsize=1，注意，设置里batchsize必须为1
        print (self.input_dynamic_lstm)  #Tensor("ExpandDims_6:0", shape=(1, 12, 180, 320, 4), dtype=float32)
        self.outputs, self.state_final = tf.nn.dynamic_rnn(self.cell, inputs =self.input_dynamic_lstm , initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.state_pla_c, self.state_pla_h), time_major = False)
        
        #self.output,self.state_final=self.cell.call(inputs=self.input_pla,state=(self.state_pla_c, self.state_pla_h) )
        print (self.outputs,self.state_final)  
        #Tensor("rnn/transpose_1:0", shape=(1, 12, 180, 320, 12), dtype=float32) 
        #LSTMStateTuple(c=<tf.Tensor 'rnn/while/Exit_3:0' shape=(1, 180, 320, 12) dtype=float32>, h=<tf.Tensor 'rnn/while/Exit_4:0' shape=(1, 180, 320, 12) dtype=float32>)
        
        
        self.flow_after=self.final_convlayer(self.outputs[0])
        print (self.flow_after) #Tensor("final_convlayer/BiasAdd:0", shape=(12, 180, 320, 4), dtype=float32)
        
        self.opticalflow_0_2=tf.slice(self.flow_after, [0, 0, 0, 0], [-1, -1, -1, 2], name='step2_opticalflow_0_2')
        self.opticalflow_2_0=tf.slice(self.flow_after, [0, 0, 0, 2], [-1, -1, -1, 2], name='step2_opticalflow_2_0')
        print ('original flow:',self.opticalflow_0_2, self.opticalflow_2_0)
        #original flow: Tensor("step2_opticalflow_0_2:0", shape=(12, 180, 320, 2), dtype=float32) Tensor("step2_opticalflow_2_0:0", shape=(12, 180, 320, 2), dtype=float32)
        
        #反向光流算中间帧
        self.opticalflow_t_0=tf.add( -(1-self.timerates_expand)*self.timerates_expand*self.opticalflow_0_2 ,\
                                      self.timerates_expand*self.timerates_expand*self.opticalflow_2_0 , name="step2_opticalflow_t_0")
        self.opticalflow_t_2=tf.add( (1-self.timerates_expand)*(1-self.timerates_expand)*self.opticalflow_0_2 ,\
                                      self.timerates_expand*(self.timerates_expand-1)*self.opticalflow_2_0, name="step2_opticalflow_t_2")
        
        print ('two optical flow:',self.opticalflow_t_0, self.opticalflow_t_2)
        #two optical flow: Tensor("step2_opticalflow_t_0:0", shape=(12, 180, 320, 2), dtype=float32) Tensor("step2_opticalflow_t_2:0", shape=(12, 180, 320, 2), dtype=float32)
        
        #2种方法合成t时刻的帧
        self.img_flow_2_t=self.warp_op(self.frame2, -self.opticalflow_t_2) #!!!
        self.img_flow_0_t=self.warp_op(self.frame0, -self.opticalflow_t_0) #!!!
        
        self.G_net=tf.add(self.timerates_expand*self.img_flow_2_t , (1-self.timerates_expand)*self.img_flow_0_t, name="step2_net_generate" )
        
        #利用光流前后帧互相合成
        self.img_flow_2_0=self.warp_op(self.frame2, self.opticalflow_2_0)  #frame2->frame0
        self.img_flow_0_2=self.warp_op(self.frame0, self.opticalflow_0_2)  #frame0->frame2
        
        
        #1、contex loss
        print ("forming conx loss：")
        tep_G_shape=self.G_net.get_shape().as_list()[1:]
        
        self.contex_Genera =tf.keras.applications.VGG16(include_top=False, input_tensor=self.G_net,  input_shape=tep_G_shape).get_layer("block4_conv3").output
        self.contex_frame1 =tf.keras.applications.VGG16(include_top=False, input_tensor=self.frame1, input_shape=tep_G_shape).get_layer("block4_conv3").output
        
        self.contex_loss=   tf.reduce_mean(tf.squared_difference( self.contex_frame1, self.contex_Genera), name='step2_Contex_loss')
        print ('step2_loss_mean_contex form finished..')
        
        
        #2、L1 loss
        print ("forming L1 loss:生成帧与GT、frame2->frame0与frame0、frame0->frame2与frame2")
        self.L1_loss_interframe =tf.reduce_mean(tf.abs(  self.G_net-self.frame1  ))
        self.L1_loss_all        =tf.reduce_mean(tf.abs(  self.G_net-self.frame1  ) + \
                                                tf.abs(self.img_flow_2_0-self.frame0) + \
                                                tf.abs(self.img_flow_0_2-self.frame2), name='G_clear_l1_loss')
        #self.G_loss_mean_Square=  self.contex_loss*1 + self.L1_loss_all
        print ('step2_loss_mean_l1 form finished..')
        
        #6 SSIM
        self.ssim = tf.image.ssim(self.G_net, self.frame1, max_val=2.0)
        print ("ssim:",self.ssim)  #ssim: Tensor("Mean_10:0", shape=(12,), dtype=float32)
        
        #7 PSNR
        self.psnr = tf.image.psnr(self.G_net, self.frame1, max_val=2.0, name="step2_frame1_psnr")
        print ("psnr:", self.psnr) #psnr: Tensor("G_frame1_psnr/Identity_3:0", shape=(12,), dtype=float32)
        
        
        self.G_loss_all=self.contex_loss + self.L1_loss_all
        
        self.train_op = tf.train.AdamOptimizer(LR, name="step2_adam").minimize(self.G_loss_all)
        
        t_vars=tf.trainable_variables()
        print ("trainable vars cnt:",len(t_vars))
        
        
        
        #最后构建完成后初始化参数 
        self.sess.run(tf.global_variables_initializer())

        
    
    def warp_op(self, images, flow, timerates=1):
        '''
        tf.contrib.image.dense_image_warp(
            image,
            flow,
            name='dense_image_warp'
        )
        pixel value at output[b, j, i, c] is
          images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].
        '''
        return tf.contrib.image.dense_image_warp(images, flow*timerates)
    
    #最终输出加上一个conv层才是得到的光流
    def final_convlayer(self, inputdata, filterlen=kernel_len, outchannel=flow_channel*2,   scopename="final_convlayer"):
        inputshape=inputdata.get_shape().as_list()
        
        with tf.variable_scope(scopename,  reuse=tf.AUTO_REUSE) as scope: 
            kernel=tf.get_variable('weights', [filterlen,filterlen, inputshape[-1], outchannel], dtype=tf.float32, \
                                   initializer=tf.random_normal_initializer(stddev=stddev))
            #tf.nn.conv2d中的filter参数，是[filter_height, filter_width, in_channels, out_channels]的形式，
                    
            ret=tf.nn.conv2d(inputdata, kernel, strides=[1,1,1,1], padding="SAME")
                    
            bias=tf.get_variable('bias', [outchannel], dtype=tf.float32, initializer=tf.constant_initializer(bias_init))
            ret=tf.nn.bias_add(ret, bias)
        return ret
        
    def getbatch_train_imgs(self):
        newstate=False
        while True:
            tepimg=self.sess.run(self.pipline_data_train)
            inimg,rate,label=tepimg[0],tepimg[1],tepimg[2]
            if str(label[0]).split('_')[0]==str(label[-1]).split('_')[0]: break
            newstate=True
        return self.img2tanh(inimg),rate,newstate
    
    def getbatch_test_imgs(self):
        newstate=False
        while True:
            tepimg=self.sess.run(self.pipline_data_test)
            inimg,rate,label=tepimg[0],tepimg[1],tepimg[2]
            if str(label[0]).split('_')[0]==str(label[-1]).split('_')[0]: break
            newstate=True
        return self.img2tanh(inimg),rate,newstate
    
    def img2tanh(self,img):
        #img=tf.cast(img,tf.float32)
        img-=mean_dataset*3
        return img*2.0/255-1
    
    def tanh2img(self,tanhd):
        tep= (tanhd+1)*255//2
        #print ('tep.shape:',tep.shape)  #tep.shape: (180, 320, 9)
        multly=int(tep.shape[-1]/len(mean_dataset))
        #print ('expanding:',multly)
        tep+=mean_dataset*multly
        return tep.astype(np.uint8)
        

    def train_once(self):
        imgdata,rate,newstate=self.getbatch_train_imgs()
        if newstate: 
            self.state_new_train=self.state_init_np
            print ('start from a zero state!!!')
        
        _, \
        self.state_new_train,   \
        ssim,psnr,      \
        contexloss, L1_loss, loss_all=self.sess.run([self.train_op,\
                                    self.state_final, \
                                    self.ssim, self.psnr, self.contex_loss, self.L1_loss_all, self.G_loss_all], \
                      feed_dict={self.imgs_pla:imgdata,self.timerates:rate,self.training:True,  self.state_pla_c:self.state_new_train[0],  self.state_pla_h:self.state_new_train[1]})
        
        #print ()
        print ("train once:")
        print ("ssim:",ssim)
        print ("psnr:",psnr)
        print ("contexloss:",contexloss)
        print ("l1_loss_all:",L1_loss)
        print ("loss_all:",loss_all)
        
        return ssim,psnr,contexloss,L1_loss

    def eval_once(self, step,evalstep=100):
        kep_img_dir=op.join(kepimgdir, str(step))
        os.makedirs(kep_img_dir, exist_ok=True)
        
        recording=0  #遇到一个新的状态开始记录，到下一个状态停止
        kep_ssim=0.0
        kep_psnr=0.0
        kep_l1loss=0.0
        kep_contexloss=0.0
        
        img_cnt=0
        
        for i in range(evalstep):
            imgdata,rate,newstate=self.getbatch_test_imgs()
            if newstate: 
                self.state_new_test=self.state_init_np
                recording+=1
    
            self.state_new_test,   \
            ssim,psnr,      \
            contexloss, L1_loss, loss_all,\
            step1_flow_0_1,step1_flow_1_0,step1_imgout,step2_flow_0_1,step2_flow_1_0,step2_outimg=self.sess.run([
                                        self.state_final, \
                                        self.ssim, self.psnr, self.contex_loss, self.L1_loss_all, self.G_loss_all,\
                                        self.optical_0_1, self.optical_1_0, self.outimg, self.opticalflow_0_2,self.opticalflow_2_0,self.G_net], \
                          feed_dict={self.imgs_pla:imgdata,self.timerates:rate,self.training:False,  self.state_pla_c:self.state_new_test[0],  self.state_pla_h:self.state_new_test[1]})
        
            kep_ssim+=np.mean(ssim)
            kep_psnr+=np.mean(psnr)
            kep_l1loss+=np.mean(L1_loss)
            kep_contexloss+=np.mean(contexloss)
            
            if recording==1:
                tep=self.form_bigimg(imgdata,step1_flow_0_1,step1_flow_1_0,step1_imgout,step2_flow_0_1,step2_flow_1_0,step2_outimg)
                cv2.imwrite( op.join(kep_img_dir, "recording_"+str(recording)+"_"+str(img_cnt)+'.jpg') ,  tep)
                img_cnt+=1
                
                    
            
        print ("eval ",evalstep,' times:','\nmean ssim:',kep_ssim/evalstep,' mean psnr:',kep_psnr/evalstep,'\nmean l1loss:',kep_l1loss/evalstep,' mean contexloss:',kep_contexloss/evalstep)
        print ("write "+str(img_cnt)+" imgs to:"+kep_img_dir)
        return kep_ssim/evalstep,kep_psnr/evalstep,kep_contexloss/evalstep,kep_l1loss/evalstep
    
    def form_bigimg(self, imgdata,step1_flow_0_1,step1_flow_1_0,step1_imgout,step2_flow_0_1,step2_flow_1_0,step2_outimg):
        #imgdata:[12,180,320,9]
        #根据输入的原始输入3图，第一步的输出前后光流，第一步的输出合成中间帧，第二部的前后光流，第二部的合成中间帧--->合成一张大图
        gap=6
        height=self.batchsize_inputimg*self.img_size_h+ (self.batchsize_inputimg-1)*gap
        width =self.img_size_w*9+(9-1)*gap
        
        ret=np.zeros([height, width, img_channel], dtype=np.uint8)
        imgdata=self.tanh2img(imgdata)
        step1_imgout=self.tanh2img(step1_imgout)
        step2_outimg=self.tanh2img(step2_outimg)
        
        for j in range(self.batchsize_inputimg):
            col=0
            row=j*(self.img_size_h+gap)
            #0,1,2
            for k in range(group_img_num): 
                ret[row:row+self.img_size_h, col:col+self.img_size_w]=imgdata[j, :,:,k*img_channel:(k+1)*img_channel]
                col+=self.img_size_w+gap
            #3
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step1_flow_0_1[j])
            col+=self.img_size_w+gap
            #4
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step1_flow_1_0[j])
            col+=self.img_size_w+gap
            #5
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=step1_imgout[j]
            col+=self.img_size_w+gap
            #6
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step2_flow_0_1[j])
            col+=self.img_size_w+gap
            #7
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=self.flow_bgr(step2_flow_1_0[j])
            col+=self.img_size_w+gap
            #8
            ret[row:row+self.img_size_h, col:col+self.img_size_w]=step2_outimg[j]
            col+=self.img_size_w+gap
        return ret
    
    
    def flow_bgr(self, flow):
        #flow:[h,w,2] no batch
        # Use Hue, Saturation, Value colour model 
        #色调（H），饱和度（S），明度（V）
        #这里色调<->方向
        #饱和度 =255
        #明度<->运动长度
        #黑色代表无运动
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), np.uint8)
        #print (hsv.shape)
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


if __name__ == '__main__':   
    with tf.Session(config=config) as sess:     
        gan = Step2_ConvLstm(sess)
        
        logwriter = tf.summary.FileWriter(logdir,   sess.graph)
        
        all_saver = tf.train.Saver(max_to_keep=2) 


        begin_t=time.time()
        for i in range(maxstep):     
            print ('\n',"{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
                   
            if i==0 or (i+1)%2000==0:#一次测试
                print ('begining to eval D:')
                ssim_mean, psnr_mean,contexloss,l1loss=gan.eval_once(i)
                
                #自己构建summary
                tsummary = tf.Summary()
                #tsummary.value.add(tag='mean prob of real1', simple_value=prob_T)
                #tsummary.value.add(tag='mean prob of fake1', simple_value=prob_F)
                tsummary.value.add(tag='mean L1_loss of G and GT', simple_value=l1loss)
                tsummary.value.add(tag='mean contexloss', simple_value=contexloss)
                tsummary.value.add(tag='mean ssim:', simple_value=ssim_mean)
                tsummary.value.add(tag='mean psnr:', simple_value=psnr_mean)
                #写入日志
                logwriter.add_summary(tsummary, i)
                
            if i==0 or (i+1)%1500==0:#保存一波图片
                pass#gan.eval_G_once(i)
                
                
            if (i+1)%2000==0:#保存模型
                print ('saving models...')
                pat=all_saver.save(sess, op.join(logdir,'model_keep'),global_step=i)
                print ('saved at:',pat)
            
            
            stt=time.time()
            print ('%d/%d  start train_once...'%(i,maxstep))
            #lost,sum_log=vgg.train_once(sess) #这里每次训练都run一个summary出来
            gan.train_once()
            #写入日志
            #logwriter.add_summary(sum_log, i)
            #print ('write summary done!')
            
                
            print ('time used:',time.time()-stt,' to be ',1.0/(time.time()-stt),' iters/s', ' left time:',(time.time()-stt)*(maxstep-i)/60/60,' hours')
            
        
        print ('Training done!!!-->time used:',(time.time()-begin_t),'s = ',(time.time()-begin_t)/60/60,' hours')

        
        
        
        
        
        
        
        
        
        
        
        
        
