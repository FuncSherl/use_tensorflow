#coding:utf-8
'''
Created on 2018年12月10日

@author: sherl
'''
import cv2,os,random,time
from datetime import datetime
import os.path as op
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from GAN_tools import *
import Dataset_adobe240fps_supply as datasupply


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())


train_size=datasupply.train_frames_sum #训练集规模 ：112064 
test_size=datasupply.test_frames_sum   #测试集规模：8508
batchsize=20

img_size_w=datasupply.target_imgw  #640
img_size_h=datasupply.target_imgh   #360
img_size=[img_size_h, img_size_w]

base_lr=0.0002 #基础学习率
beta1=0.5

maxstep=160000 #训练多少次
eval_step=int (test_size/batchsize)

decay_steps=10000
decay_rate=0.99

#incase_div_zero=1e-10  #这个值大一些可以避免d训得太好，也避免了g梯度

G_first_channel=12  #不是G的输入channel，而是g的输入经过一次卷积后的输出channel
D_first_channel=18

#G中unet的层数
G_unet_layercnt=5
G_filter_len=3
G_withbias=True


#两个D的用的D_block的层数，即缩小几回
D_1_layercnt=5
D_1_filterlen=3
D_1_withbias=True

D_2_layercnt=5
D_2_filterlen=3
D_2_withbias=True

#一次输入网络多少图片，这里设定为3帧，利用前后帧预测中间
G_group_img_num=3
img_channel=datasupply.img_channel  #3

logdir="./logs/GAN_"+TIMESTAMP+('_base_lr-%f_batchsize-%d_maxstep-%d'%(base_lr,batchsize, maxstep))

bigimgsdir=op.join(logdir, 'randomimgs')
if not op.exists(bigimgsdir): os.makedirs(bigimgsdir)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


class GAN_Net:
    def __init__(self, sess):
        self.sess=sess
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.G_para=[]
        self.D_para=[]       
        
        #for debug
        self.cnt_tep=0
        self.deb_kep=0
        self.deb_kep2=0
        
        #3个placeholder， img和noise,training 
        self.imgs_pla = tf.placeholder(tf.float32, [batchsize, img_size_h, img_size_w, G_group_img_num*img_channel], name='imgs_in')
        self.training=tf.placeholder(tf.bool, name='training_in')
        
        self.frame0=self.imgs_pla[:,:,:,:3]
        self.frame1=self.imgs_pla[:,:,:,3:6]
        self.frame2=self.imgs_pla[:,:,:,6:]
        
        frame0and2=tf.concat([self.frame0, self.frame2], 3)
        #print ('after concat:',frame0and2)
        self.G_net=self.Generator_net(frame0and2)
        
        #D_1的输出 
        frame0_False_2=tf.concat([self.frame0, self.G_net,self.frame2], 3)
        self.D_linear_net_F, self.D_linear_net_F_logit=self.Discriminator_net_linear(frame0_False_2)
        self.D_linear_net_T, self.D_linear_net_T_logit=self.Discriminator_net_linear(self.imgs_pla)
        self.D_linear_net_loss_sum, self.D_linear_net_loss_T, self.D_linear_net_loss_F=self.D_loss_TandF_logits(self.D_linear_net_T_logit, \
                                                                                                                self.D_linear_net_F_logit, "D_linear_net")
        
        #D_2的输出
        self.D_clear_net_F, self.D_clear_net_F_logit=self.Discriminator_net_clear(self.G_net)
        self.D_clear_net_T, self.D_clear_net_T_logit=self.Discriminator_net_clear(self.frame1)
        self.D_clear_net_loss_sum, self.D_clear_net_loss_T, self.D_clear_net_loss_F=self.D_loss_TandF_logits(self.D_clear_net_T_logit, \
                                                                                                                self.D_clear_net_F_logit, "D_clear_net")
        #这里对两个D的loss没有特殊处理，只是简单相加
        self.D_loss_all=self.D_clear_net_loss_sum + self.D_linear_net_loss_sum
        
        #下面是G的loss
        self.G_loss_mean_D1=self.G_loss_F_logits(self.D_linear_net_F_logit, 'G_loss_D1')
        self.G_loss_mean_D2=self.G_loss_F_logits(self.D_clear_net_F_logit, 'G_loss_D2')
        self.G_loss_all=self.G_loss_mean_D1 + self.G_loss_mean_D2
        
        #训练使用
        self.train_D=self.train_op_D(decay_steps, decay_rate)
        self.train_G=self.train_op_G(decay_steps, decay_rate)        
        
        #还是应该以tf.trainable_variables()为主
        t_vars=tf.trainable_variables()
        print ("trainable vars cnt:",len(t_vars))
        self.G_para=[var for var in t_vars if var.name.startswith('G')]
        self.D_para=[var for var in t_vars if var.name.startswith('D')]
        
        '''
        print ('\nshow all trainable vars:',len(tf.trainable_variables()))
        for i in tf.trainable_variables():
            print (i)
        '''
        print ('\nfirst show G params')
        for ind,i in enumerate(self.G_para): print (ind,i)
        
        print('\nnext is D:\n')
        for ind,i in enumerate(self.D_para): print (ind,i)
        
        print ('\nnext is tf.GraphKeys.UPDATE_OPS:')
        print (tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        
        self.summary_all=tf.summary.merge_all()
        init = tf.global_variables_initializer()#初始化tf.Variable
        self.sess.run(init)
        
        
    def img2tanh(self,img):
        #img=tf.cast(img,tf.float32)
        return img*2.0/255-1
    
    def tanh2img(self,tanhd):
        tep= (tanhd+1)*255//2
        return tep.astype(np.uint8)  
    
    def getbatch_train_imgs(self):
        tepimg=datasupply.get_train_batchdata(batchsize, G_group_img_num)
        return self.img2tanh(tepimg)
    
    def getbatch_test_imgs(self):
        tepimg=datasupply.get_test_batchdata(batchsize, G_group_img_num)
        return self.img2tanh(tepimg)
    
    def D_loss_TandF_logits(self, logits_t, logits_f, summaryname='default'):
        self.D_loss_fir=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_t, labels=tf.ones_like(logits_t))   #real
        
        
        self.D_loss_sec=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_f, labels=tf.zeros_like(logits_f))  #fake
        
       
        #testing target
        real_loss_mean=tf.reduce_mean(self.D_loss_fir)
        fake_loss_mean=tf.reduce_mean(self.D_loss_sec)
        tf.summary.scalar(summaryname+'_real_loss_mean',real_loss_mean)
        tf.summary.scalar(summaryname+'_fake_loss_mean',fake_loss_mean)
        
        loss_mean=real_loss_mean+fake_loss_mean
        
        tf.summary.scalar(summaryname+'_sum_loss_mean',loss_mean)        
        ############################################################
        
        return loss_mean,real_loss_mean,fake_loss_mean
        
    
    def G_loss_F_logits(self, logits, summaryname='default'):
        self.G_loss_fir=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
        loss_mean = tf.reduce_mean(self.G_loss_fir)
        tf.summary.scalar(summaryname+'_loss_mean',loss_mean)
        
        return loss_mean
    
    
    def train_op_G(self,decay_steps=8000, decay_rate=0.99, beta1=beta1):
        #self.lr_rate = base_lr
        self.lr_rate = tf.train.exponential_decay(base_lr,  global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate)
        
        print ('G: AdamOptimizer to maxmize %d vars..'%(len(self.G_para)))
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.G_optimizer=tf.train.AdamOptimizer(self.lr_rate  , beta1=beta1)
            
            #for i in optimizer.compute_gradients(self.G_loss_mean, var_list=self.G_para): print (i)
            
            train_op =self.G_optimizer.minimize(self.G_loss_all, global_step=self.global_step,var_list=self.G_para)
        
        return train_op
    
    def train_op_D(self,decay_steps=8000, decay_rate=0.99, beta1=beta1):
        #self.lr_rate = base_lr
        self.lr_rate = tf.train.exponential_decay(base_lr,  global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate)
        
        print ('D: AdamOptimizer to maxmize %d vars..'%(len(self.D_para)))
        
        #这里就不管globalstep了，否则一次迭代会加2次
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_optimizer= tf.train.AdamOptimizer(self.lr_rate  , beta1=beta1)
            train_op=self.D_optimizer.minimize(self.D_loss_all, var_list=self.D_para)   #global_step=self.global_step,
        return train_op
                                                                                                                                                       
    
    
    
    #tensor 范围外   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    def train_once_all(self):
        tepimgs=self.getbatch_train_imgs()
        
        lrrate,_,D1_T_prob, D1_F_prob, D2_T_prob, D2_F_prob, D1_loss, D2_loss, D_loss_sum_all =self.sess.run([self.lr_rate, self.train_D , \
                                                      self.D_linear_net_T, self.D_linear_net_F,\
                                                      self.D_clear_net_T, self.D_clear_net_F,  \
                                                      self.D_linear_net_loss_sum, self.D_clear_net_loss_sum, self.D_loss_all], \
                                                                            feed_dict={  self.imgs_pla:tepimgs , self.training:True})
        print ('trained D:')
        print ('lr:',lrrate)
        print ('D1(D_linear) prob T/F --> ',D1_T_prob,'/', D1_F_prob)
        print ('D1 loss_all:',D1_loss)
        print ('D2(D_clear) prob T/F --> ',D2_T_prob,'/', D2_F_prob)
        print ('D2 loss_all:',D2_loss)
        
        
        
        
        
        
        #print ('deb1',self.sess.run(self.debug))                   
        
        #debug2 d_fir  debug:g_last
        
        noise=self.get_noise()
        '''
        #train d first
        train_prob_t,debugout2,debugout1,lrr, deb_D, _,_,dloss=self.sess.run([self.D_net,
                                                                                self.debug2, self.debug, 
                                                                                self.lr_rate, 
                                                                                self.test_ori_loss_D , #测试自己的loss函数
                                                                                self.train_D_real, self.train_D_fake, self.D_loss_mean], 
                                                                                         feed_dict={  self.noise_pla: noise , self.training:True})
        '''
        #train d first
        train_prob_t,debugout2,debugout1,lrr, deb_D, _,dloss=self.sess.run([self.D_net,
                                                                                self.debug2, self.debug, 
                                                                                self.lr_rate, 
                                                                                self.test_ori_loss_D , #测试自己的loss函数
                                                                                self.train_D, self.D_loss_mean], 
                                                                                         feed_dict={  self.noise_pla: noise , self.training:True})
        
        print ('trained D:')
        print('D_first kernel[0,0,:,0]:\n',debugout2)
        print ('G_last kernel[0,0,:,0]:\n',debugout1)
        
        train_prob_f,debugout2,debugout1,deb_G, _,gloss,summary=self.sess.run([self.whole_net,
                                                                                self.debug2, self.debug,                                                 
                                                                                self.test_ori_loss_G, 
                                                                                self.train_G, self.G_loss_mean,
                                                                                self.summary_all], 
                                                                                     feed_dict={  self.noise_pla: noise , self.training:True})
        
        print ('trained G:')
        print('D_first kernel[0,0,:,0]:\n',debugout2,debugout2-self.deb_kep)
        self.deb_kep=debugout2
        print ('G_last kernel[0,0,:,0]:\n',debugout1,debugout1-self.deb_kep2)
        self.deb_kep2=debugout1
        
        '''
        #原版训练，有问题
        train_prob_f,train_prob_t,debugout2,debugout1,lrr, deb_D,deb_G, _,_,dloss,gloss,summary=self.sess.run([self.whole_net,self.D_net,
                                                                                self.debug2, self.debug, 
                                                                                self.lr_rate, 
                                                                                self.test_ori_loss_D ,self.test_ori_loss_G, 
                                                            self.train_D, self.train_G, self.D_loss_mean,self.G_loss_mean,
                                                            self.summary_all], 
                                                           feed_dict={  self.noise_pla: noise , self.training:True})
        '''
        
        
        print ('the lr_rate is:', lrr)
        print ('this train probs:', 'true:',np.mean(train_prob_t), '   false:',np.mean(train_prob_f))
        #print ('MyGloss:',deb_G, '  MyDloss:',deb_D)

        return summary,dloss,gloss    
       
    
    def Run_G(self, training=False):
        noise=self.get_noise()
        inerimg, outerprob=self.sess.run([self.G_net, self.whole_net], feed_dict={self.noise_pla: noise, self.training:training})
        return inerimg, outerprob
    
    def Run_WholeNet(self, training=False):
        '''
        training 为false时，bn会用学习的参数bn，因此在训练时的prob和测试时的prob又很大差异
        '''
        noise=self.get_noise()
        probs=self.sess.run(self.whole_net, feed_dict={self.noise_pla: noise, self.training:training})
        return probs
    
    def Run_D(self, training=False):
          
        '''
        #这里imgs要求是tanh化过的，即归一化到-1~1 
        training 为false时，bn会用学习的参数bn，因此在训练时的prob和测试时的prob又很大差异
        ''' 
        probs=self.sess.run(self.D_net, feed_dict={self.training:training})  #, feed_dict={ self.imgs_pla: self.img2tanh(self.tf_inimg) })
        #越接近真实图像越接近1
        return probs
    
    
    def eval_G_once(self, step=0):
        desdir=op.join(logdir, str(step))
        if not op.isdir(desdir): os.makedirs(desdir)
        
        #这里cnt不应该大于batchsize(64)
        cnt=4
        
        #中间用cnt像素的黑色线分隔图片
        bigimg_len=img_size*cnt+(cnt-1)*cnt
        bigimg_bests=np.zeros([bigimg_len[0],bigimg_len[1],3], dtype=np.uint8)
        bigimg_name='step-'+str(step)+'_cnt-'+str(cnt)+'_batchsize-'+str(batchsize)+'.png'
        
        for i in range(cnt):
            tepimgs,probs=self.Run_G()
            #保存原图
            for ind,j in enumerate(tepimgs[:cnt*3]):  
                #print (j[0][0][0])
                j=self.tanh2img(j)      
                #print (j[0][0][0])        
                im = Image.fromarray(j)
                imgname=str(i)+'_'+str(ind)+".jpg"
                im.save(op.join(desdir, imgname))
            
            #每个batch选随机的cnt个合成图片
            #print (probs.shape)
            tep=list(range(batchsize))
            random.shuffle(tep) #随机取cnt个图
            tep=tep[:cnt]  #np.argsort(probs[:,0])[-cnt:]
            #print (tep)
            for ind,j in enumerate(tep):
                st_x= ind*(img_size+cnt) #列
                st_y= i*(img_size+cnt) #行
                bigimg_bests[st_y:st_y+img_size, st_x:st_x+img_size,:]=self.tanh2img(tepimgs[j])
        
        bigimg_dir=op.join(bigimgsdir, bigimg_name)
        im = Image.fromarray(bigimg_bests)
        im.save(bigimg_dir)
            
        print ('eval_G_once,saved imgs to:',desdir, '\nbestimgs to:',bigimg_dir)
        
    def evla_D_once(self,eval_step=eval_step):
        cnt_real=0
        cnt_fake=0
        for i in range(eval_step):
            probs=self.Run_WholeNet()
            #print ('show prob shape:',probs.shape)  #[32,1]
            cnt_fake+=np.mean(probs)
        
        for i in range(eval_step):
            probs=self.Run_D()
            cnt_real+=np.mean(probs)
        return cnt_real/eval_step, cnt_fake/eval_step
        
    
    def Generator_net(self, inputdata, withbias=G_withbias, filterlen=G_filter_len):
        with tf.variable_scope("G_Net",  reuse=tf.AUTO_REUSE) as scope:
            tepimg=my_unet(inputdata,  layercnt=G_unet_layercnt,  filterlen=filterlen, withbias=withbias)
            #####################################################################################################################################
            #tanh
            self.G_tanh= tf.nn.tanh(tepimg, name='G_tanh')
        
        return self.G_tanh
            
                
    
    def Discriminator_net_linear(self, imgs_3, withbias=D_1_withbias, filterlen=D_1_filterlen):
        '''
        :用于判定视频帧连续性的D
        imgs_3:9 channel input imgs[batchsize, h, w, 9]
        '''
        #这里输入的imgs应该是tanh后的，位于-1~1之间
        #cast to float 
        self.imgs_float32=tf.cast(imgs_3, tf.float32)
        
        layer_cnt=D_1_layercnt
        inputshape=imgs_3.get_shape().as_list()
        initchannel=inputshape[-1]
        
            
        with tf.variable_scope("D_1_Net",  reuse=tf.AUTO_REUSE) as scopevar:
            scope=scopevar.name
            tep=my_conv(self.imgs_float32, filterlen+int(layer_cnt/2), initchannel*2, scope+'_start', stride=2, withbias=withbias)
    
            #tep=my_batchnorm( tep,self.training, scope)   #第一层不要bn试试
            tep=my_lrelu(tep, scope)
            
            for i in range(layer_cnt):
                tep=my_D_block(tep, initchannel*( 2**(i+2)), scope+'_Dblock'+str(i), filterlen=filterlen+int( (layer_cnt-i)/2 ), \
                               withbias=withbias, training=self.training)
            #######################################################################################################################################
            #fc
            tep=my_fc(tep, 1024, scope+'_fc1',  withbias=withbias)
            tep=my_lrelu(tep, scope)
            
            tep=my_fc(tep, 1, scope+'_fc2',  withbias=withbias)
            
            #sigmoid
            self.D_1_sigmoid=tf.nn.sigmoid(tep, name='D_1_sigmoid')
            print (self.D_1_sigmoid)
            
        return self.D_1_sigmoid, tep
    
    
    def Discriminator_net_clear(self, imgs, withbias=D_2_withbias, filterlen=D_2_filterlen):
        '''
        :用于判定生成视频帧的真实性的D
        imgs:input imgs [batchsize, h, w, 3]
        '''
        #这里输入的imgs应该是tanh后的，位于-1~1之间
        #cast to float 
        self.imgs_float32=tf.cast(imgs, tf.float32)
        
        layer_cnt=D_2_layercnt
        inputshape=imgs.get_shape().as_list()
        initchannel=inputshape[-1]
            
        with tf.variable_scope("D_2_Net",  reuse=tf.AUTO_REUSE) as scopevar:
            scope=scopevar.name
            tep=my_conv(self.imgs_float32, filterlen+int(layer_cnt/2), initchannel*2, scope+'_start', stride=2, withbias=withbias)
    
            #tep=my_batchnorm( tep,self.training, scope)   #第一层不要bn试试
            tep=my_lrelu(tep, scope)
            
            for i in range(layer_cnt):
                tep=my_D_block(tep, initchannel*( 2**(i+2)), scope+'_Dblock'+str(i), filterlen=filterlen+int( (layer_cnt-i)/2 ), \
                               withbias=withbias, training=self.training)
            #######################################################################################################################################
            #fc
            tep=my_fc(tep, 1024, scope+'_fc1',  withbias=withbias)
            tep=my_lrelu(tep, scope)
            
            tep=my_fc(tep, 1, scope+'_fc2',  withbias=withbias)
            
            
            #sigmoid
            self.D_2_sigmoid=tf.nn.sigmoid(tep, name='D_2_sigmoid')
            
        return self.D_2_sigmoid, tep
        
        




if __name__ == '__main__':   
    with tf.Session() as sess:      
        gan=GAN_Net(sess)
        
        logwriter = tf.summary.FileWriter(logdir,   sess.graph)
        
        all_saver = tf.train.Saver(max_to_keep=2) 


        begin_t=time.time()
        for i in range(maxstep):            
            if ((i+1)%500==0):#一次测试
                print ('\nbegining to eval D:')
                real,fake=gan.evla_D_once()
                print ('mean prob of real/fake:',real,fake)
                
                #自己构建summary
                tsummary = tf.Summary()
                tsummary.value.add(tag='mean prob of real', simple_value=real)
                tsummary.value.add(tag='mean prob of fake', simple_value=fake)
                #tsummary.value.add(tag='test epoch loss:', simple_value=tloss)
                #写入日志
                logwriter.add_summary(tsummary, i)
                
            if i==0 or (i+1)%1000==0:#保存一波图片
                gan.eval_G_once(i)
                
                
            if (i+1)%2000==0:#保存模型
                print ('saving models...')
                pat=all_saver.save(sess, op.join(logdir,'model_keep'),global_step=i)
                print ('saved at:',pat)
            
            
            stt=time.time()
            print ('\n%d/%d  start train_once...'%(i,maxstep))
            #lost,sum_log=vgg.train_once(sess) #这里每次训练都run一个summary出来
            sum_log,dloss,gloss=gan.train_once_all()
            #写入日志
            logwriter.add_summary(sum_log, i)
            #print ('write summary done!')
            
            #######################
            
            real,fake=gan.evla_D_once(1)
            print ('once prob of real/fake:',real,fake)
            
            print ('train once-->gloss:',gloss,'  dloss:',dloss)
            
            print ('time used:',time.time()-stt,' to be ',1.0/(time.time()-stt),' iters/s')
            
        
        print ('Training done!!!-->time used:',(time.time()-begin_t),'s = ',(time.time()-begin_t)/60,' min')








