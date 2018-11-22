#coding:utf-8
'''
Created on 2018年10月29日

@author:China
'''

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from datetime import datetime
import time,cv2
import os.path as op

import matplotlib.pyplot as plt

#当前路径下的文件
from  use_tensor.locate_feature.vgg16_ori import *
from use_tensor.locate_feature.proc_voc import classes


batch_select=1

modelpath='./logs/VOC_2018-11-02_20-45-58_base_lr-0.001000_batchsize-30_maxstep-30000'

'''
out: Tensor("conv5_3/BiasAdd:0", shape=(30, 14, 14, 512), dtype=float32)
self.conv5_3 Tensor("conv5_3:0", shape=(30, 14, 14, 512), dtype=float32)
input to first fc length: 25088
self.fc2 Tensor("fc2/Relu:0", shape=(30, 4096), dtype=float32)
GradientDescentOptimizer1 to minimize 30 vars..
GradientDescentOptimizer2 to minimize 2 vars..
dict in vgg16: imgs Tensor("Placeholder:0", shape=(30, 224, 224, 3), dtype=float32)
dict in vgg16: labs Tensor("Placeholder_1:0", shape=(30,), dtype=int32)
dict in vgg16: training Tensor("Placeholder_2:0", dtype=bool)
dict in vgg16: dropout 0.5
dict in vgg16: global_step <tf.Variable 'global_step:0' shape=() dtype=int32_ref>
dict in vgg16: parameters [<tf.Variable 'conv1_1/weights:0' shape=(3, 3, 3, 64) dtype=float32_ref>, <tf.Variable 'conv1_1/biases:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'conv1_2/weights:0' shape=(3, 3, 64, 64) dtype=float32_ref>, <tf.Variable 'conv1_2/biases:0' shape=(64,) dtype=float32_ref>, <tf.Variable 'conv2_1/weights:0' shape=(3, 3, 64, 128) dtype=float32_ref>, <tf.Variable 'conv2_1/biases:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'conv2_2/weights:0' shape=(3, 3, 128, 128) dtype=float32_ref>, <tf.Variable 'conv2_2/biases:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'conv3_1/weights:0' shape=(3, 3, 128, 256) dtype=float32_ref>, <tf.Variable 'conv3_1/biases:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'conv3_2/weights:0' shape=(3, 3, 256, 256) dtype=float32_ref>, <tf.Variable 'conv3_2/biases:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'conv3_3/weights:0' shape=(3, 3, 256, 256) dtype=float32_ref>, <tf.Variable 'conv3_3/biases:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'conv4_1/weights:0' shape=(3, 3, 256, 512) dtype=float32_ref>, <tf.Variable 'conv4_1/biases:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'conv4_2/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>, <tf.Variable 'conv4_2/biases:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'conv4_3/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>, <tf.Variable 'conv4_3/biases:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'conv5_1/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>, <tf.Variable 'conv5_1/biases:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'conv5_2/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>, <tf.Variable 'conv5_2/biases:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'conv5_3/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>, <tf.Variable 'conv5_3/biases:0' shape=(512,) dtype=float32_ref>, <tf.Variable 'fc1/weights:0' shape=(25088, 4096) dtype=float32_ref>, <tf.Variable 'fc1/biases:0' shape=(4096,) dtype=float32_ref>, <tf.Variable 'fc2/weights:0' shape=(4096, 4096) dtype=float32_ref>, <tf.Variable 'fc2/biases:0' shape=(4096,) dtype=float32_ref>]
dict in vgg16: parameters_last [<tf.Variable 'fc3/weights:0' shape=(4096, 20) dtype=float32_ref>, <tf.Variable 'fc3/biases:0' shape=(20,) dtype=float32_ref>]
dict in vgg16: conv1_1 Tensor("conv1_1:0", shape=(30, 224, 224, 64), dtype=float32)
dict in vgg16: conv1_2 Tensor("conv1_2:0", shape=(30, 224, 224, 64), dtype=float32)
dict in vgg16: pool1 Tensor("pool1:0", shape=(30, 112, 112, 64), dtype=float32)
dict in vgg16: conv2_1 Tensor("conv2_1:0", shape=(30, 112, 112, 128), dtype=float32)
dict in vgg16: conv2_2 Tensor("conv2_2:0", shape=(30, 112, 112, 128), dtype=float32)
dict in vgg16: pool2 Tensor("pool2:0", shape=(30, 56, 56, 128), dtype=float32)
dict in vgg16: conv3_1 Tensor("conv3_1:0", shape=(30, 56, 56, 256), dtype=float32)
dict in vgg16: conv3_2 Tensor("conv3_2:0", shape=(30, 56, 56, 256), dtype=float32)
dict in vgg16: conv3_3 Tensor("conv3_3:0", shape=(30, 56, 56, 256), dtype=float32)
dict in vgg16: pool3 Tensor("pool3:0", shape=(30, 28, 28, 256), dtype=float32)
dict in vgg16: conv4_1 Tensor("conv4_1:0", shape=(30, 28, 28, 512), dtype=float32)
dict in vgg16: conv4_2 Tensor("conv4_2:0", shape=(30, 28, 28, 512), dtype=float32)
dict in vgg16: conv4_3 Tensor("conv4_3:0", shape=(30, 28, 28, 512), dtype=float32)
dict in vgg16: pool4 Tensor("pool4:0", shape=(30, 14, 14, 512), dtype=float32)
dict in vgg16: conv5_1 Tensor("conv5_1:0", shape=(30, 14, 14, 512), dtype=float32)
dict in vgg16: conv5_2 Tensor("conv5_2:0", shape=(30, 14, 14, 512), dtype=float32)
dict in vgg16: conv5_3 Tensor("conv5_3:0", shape=(30, 14, 14, 512), dtype=float32)
dict in vgg16: pool5 Tensor("pool4_1:0", shape=(30, 7, 7, 512), dtype=float32)
dict in vgg16: fc1 Tensor("cond/Merge:0", shape=(30, 4096), dtype=float32)
dict in vgg16: fc2 Tensor("cond_1/Merge:0", shape=(30, 4096), dtype=float32)
dict in vgg16: fc3l Tensor("fc3/BiasAdd:0", shape=(30, 20), dtype=float32)
dict in vgg16: fc3 Tensor("fc3/Relu:0", shape=(30, 20), dtype=float32)
dict in vgg16: loss Tensor("Mean:0", shape=(), dtype=float32)
dict in vgg16: train_op name: "group_deps"
'''


class bp_model:
    def __init__(self, sess, modelpath=modelpath):
        self.sess=sess
        self.graph = tf.get_default_graph() 
        self.load_model(modelpath)   
        
        
        self.prob=self.graph.get_tensor_by_name('Softmax:0') 
        self.dat_place=self.graph.get_tensor_by_name('Placeholder:0') 
        self.label_place=self.graph.get_tensor_by_name('Placeholder_1:0') 
        self.training=self.graph.get_tensor_by_name('Placeholder_2:0') 
        
        for i in tf.trainable_variables():
            print(i)
        print (self.graph.get_all_collection_keys())       

        
        #待测试图片
        imgst,labst=self.sess.run([test_imgs, test_labs])
        feed_tep={self.dat_place: imgst,self.label_place: labst, self.training:False}#
        
        
        #///////////////////////////////////////////////////////////////////////////
        print ('batchselect:',batch_select)
        print("label_batchselect:",labst[batch_select],'-->',classes[labst[batch_select]])
        probs=sess.run(self.prob, feed_dict=feed_tep)[batch_select]
        print ('max prob index:',np.argmax(probs),"-->",np.max(probs),'-->',classes[np.argmax(probs)])
        print ('label prob:',probs[labst[batch_select]],'\n\n')
        #cv2.imshow('tets', imgst[batch_select][:,:,::-1])
        #cv2.waitKey()
        
        #////////////////////////////////////////////////////////////////////////////////////////////////////////

        self.get_onelayer_tensors('fc3')
        out3=self.sess.run([ self.fc3_out], feed_dict=feed_tep)[0]
        #print (out3.shape)
        tep=out3[batch_select]
        
        
        minmap={}
        maxmap={}
        selnum=2
        for i in np.argsort(tep)[:selnum]:
            minmap[i]=1
        for i in np.argsort(tep)[-selnum:]:
            maxmap[i]=1

        self.print_minmax(minmap, maxmap)
        
        
        
        
        minmap, maxmap=self.get_layerandlastlayer_fc2fc(minmap, maxmap, 'fc3', 'fc2', feed_tep)
        print ('fc3-->fc2:')
        self.print_minmax(minmap, maxmap)
            
            
        minmap, maxmap=self.get_layerandlastlayer_fc2fc(minmap, maxmap, 'fc2', 'fc1', feed_tep)
        print ('fc2-->fc1:')
        self.print_minmax(minmap, maxmap)
        
        
        
        
        minmap, maxmap=self.get_layerandlastlayer_fc2pool(minmap, maxmap, 'fc1', 'pool4_1:0', feed_tep)
        print ('fc1-->pool4:')
        self.print_minmax(minmap, maxmap)
        
        
        
        #这里是fc向卷积的过度，从这里开始元素有了位置信息，注意下标的转化，由一维到多维
        minmap, maxmap=self.get_layerandlastlayer_pool2cnn(minmap, maxmap,  'conv5_3', feed_tep)
        print ('pool4-->conv5_3:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2cnn(minmap, maxmap,  'conv5_3','conv5_2', feed_tep,np.array([1,1]))
        print ('conv5_3-->conv5_2:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2cnn(minmap, maxmap,  'conv5_2','conv5_1', feed_tep,np.array([1,1]))
        print ('conv5_2-->conv5_1:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2pool(minmap, maxmap,  'conv5_1','pool4:0', feed_tep,np.array([1,1]))
        print ('conv5_1-->pool4:0:')
        self.print_minmax(minmap, maxmap)
        
        
        
        
        
        minmap, maxmap=self.get_layerandlastlayer_pool2cnn(minmap, maxmap,  'conv4_3', feed_tep)
        print ('pool4:0-->conv4_3:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2cnn(minmap, maxmap,  'conv4_3','conv4_2', feed_tep,np.array([1,1]))
        print ('conv4_3-->conv4_2:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2cnn(minmap, maxmap,  'conv4_2','conv4_1', feed_tep,np.array([1,1]))
        print ('conv4_2-->conv4_1:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2pool(minmap, maxmap,  'conv4_1','pool3:0', feed_tep,np.array([1,1]))
        print ('conv4_1-->pool3:0:')
        self.print_minmax(minmap, maxmap)
        
        
        
        
        minmap, maxmap=self.get_layerandlastlayer_pool2cnn(minmap, maxmap,  'conv3_3', feed_tep)
        print ('pool3:0-->conv3_3:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2cnn(minmap, maxmap,  'conv3_3','conv3_2', feed_tep,np.array([1,1]))
        print ('conv3_3-->conv3_2:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2cnn(minmap, maxmap,  'conv3_2','conv3_1', feed_tep,np.array([1,1]))
        print ('conv3_2-->conv3_1:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2pool(minmap, maxmap,  'conv3_1','pool2:0', feed_tep,np.array([1,1]))
        print ('conv3_1-->pool2:0:')
        self.print_minmax(minmap, maxmap)
        
        
        
        
        
        minmap, maxmap=self.get_layerandlastlayer_pool2cnn(minmap, maxmap,  'conv2_2', feed_tep)
        print ('pool2:0-->conv2_2:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2cnn(minmap, maxmap,  'conv2_2','conv2_1', feed_tep,np.array([1,1]))
        print ('conv2_2-->conv2_1:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2pool(minmap, maxmap,  'conv2_1','pool1:0', feed_tep,np.array([1,1]))
        print ('conv2_1-->pool1:0:')
        self.print_minmax(minmap, maxmap)
        
        
        
        
        
        
        minmap, maxmap=self.get_layerandlastlayer_pool2cnn(minmap, maxmap,  'conv1_2', feed_tep)
        print ('pool1:0-->conv1_2:')
        self.print_minmax(minmap, maxmap)
        
        minmap, maxmap=self.get_layerandlastlayer_cnn2cnn(minmap, maxmap,  'conv1_2','conv1_1', feed_tep,np.array([1,1]))
        print ('conv1_2-->conv1_1:')
        self.print_minmax(minmap, maxmap)
        
        
        #finally
        minmap, maxmap=self.get_layerandlastlayer_cnn2img(minmap, maxmap, 'conv1_1',imgst[batch_select] ,feed_tep,np.array([1,1]))
        print ('conv1_1-->image:')
        self.print_minmax(minmap, maxmap)
        
        
        
        
        
        
        
        
        
        #print(mult[fc2_minindexs[0:3]])
        #/////////////////////////////////////////////////////////////////////////
        
        
        
        
        #self.show_a_vector(mult)
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        
        
        
        #print (w3,'\n',b3)
        #print (w3.T.shape)
        #self.show_weight(w3)
        
    def print_minmax(self,minmap, maxmap):
        print (len(minmap),'->',minmap)
        print (len(maxmap),'->',maxmap)
        print()
    
        
    def index2shape(self,index, targetsahpe):
        '''
        :主要用于由fc到pool时的下表转化，由一维到多维
        '''
        ret=[]
        mult=np.prod(targetsahpe)
        
        if index>=mult: return None
        
        for i in targetsahpe:
            mult=mult//i
            ret.append(index//mult)
            index=index%mult
        return ret
    
    def shape2index(self,shape, coordinate):
        '''
        :用于shape的数组中的坐标到一维index的转化
        '''
        ret=0
        mult=np.prod(shape)
        
        if len(shape)!= len(coordinate): return None
        
        for ind,i in enumerate(shape):
            mult//=i
            ret+=mult*coordinate[ind]
        return ret
    
    def get_layerandlastlayer_cnn2img(self, indexs_min, indexs_max,  cnnname_src,imagedata, feed_tep,cnn_stride ):
        now=self.get_onelayer_tensors(cnnname_src)#
        #last=self.get_onelayer_tensors(cnnname_des)#
        feat=imagedata
        print('finally :from shape:',now[-1].get_shape()[1:],'--> image:',feat.shape)
        
        w=self.sess.run([now[0]], feed_dict=feed_tep)[0]
        
        
        return self.cal_cnn(feat,w,indexs_min, indexs_max,cnn_stride,False,2)
        
    
    def get_layerandlastlayer_cnn2pool(self, indexs_min, indexs_max,  cnnname_src,poolname_des, feed_tep,cnn_stride ):
        now=self.get_onelayer_tensors(cnnname_src)#
        last=[self.graph.get_tensor_by_name(poolname_des)]#pool的name比较特殊
        
        print('from shape:',now[-1].get_shape()[1:],'-->',last[-1].get_shape()[1:])
        
        w,feat=self.sess.run([now[0], last[-1]], feed_dict=feed_tep)
        feat=feat[batch_select]
        
        return self.cal_cnn(feat,w,indexs_min, indexs_max,cnn_stride,False,2)
        
    def get_layerandlastlayer_pool2cnn(self, indexs_min, indexs_max,  cnnname, feed_tep,pool_stride=np.array([2,2]), pool_kernel=np.array([2,2,1,1])):
        cnn_tensors=self.get_onelayer_tensors(cnnname)
        
        print('des shape:',cnn_tensors[-1].get_shape()[1:],'pool kernel:',pool_kernel, 'pool stride:',pool_stride)
        
        feat=sess.run([cnn_tensors[-1]], feed_dict=feed_tep)[0][batch_select]
        
        mi,ma=self.cal_cnn(feat, np.ones(pool_kernel),indexs_min, indexs_max,pool_stride,True)
        
        return mi,ma
        #print (tep_feat.shape)
        
    def get_layerandlastlayer_cnn2cnn(self, indexs_min, indexs_max,  cnnname_src,cnnname_des, feed_tep,cnn_stride ):
        now=self.get_onelayer_tensors(cnnname_src)#
        last=self.get_onelayer_tensors(cnnname_des)#
        
        print('from shape:',now[-1].get_shape()[1:],'-->',last[-1].get_shape()[1:])
        
        w,feat=self.sess.run([now[0], last[-1]], feed_dict=feed_tep)
        feat=feat[batch_select]
        
        return self.cal_cnn(feat,w,indexs_min, indexs_max,cnn_stride,False,2)
           
        
    def get_layerandlastlayer_fc2pool(self, indexs_min, indexs_max, fcname, poolname, feed_tep):
        '''
        :当本层为fc，上层为pool时使用
        '''
        pool5_feat_tensor=self.graph.get_tensor_by_name(poolname)# 'pool4_1:0'
        w1_tensor=self.graph.get_tensor_by_name(fcname+'/weights:0')# 
        print('from shape:',w1_tensor.get_shape()[1:],'-->',pool5_feat_tensor.get_shape()[1:])
        w,feat=sess.run([w1_tensor, pool5_feat_tensor], feed_dict=feed_tep)
        tepw=w.T
        feat=feat[batch_select]
        
        shape=feat.shape
        tepfeat=feat.reshape(-1)
        
        #print (tepw.shape, tepfeat.shape)
        
        return self.cal_fc2fc(indexs_min, indexs_max,tepw, tepfeat)
        
        
    def get_layerandlastlayer_fc2fc(self,indexs_min, indexs_max, layername,lastlayername, feed_tep):
        '''
        indexs_min/max: a map, key is index, value is count
        :当本层为fc且上层也为fc时用
        '''
        now=self.get_onelayer_tensors(layername)#
        last=self.get_onelayer_tensors(lastlayername)#
        
        print('from shape:',now[-1].get_shape()[1:],'-->',last[-1].get_shape()[1:])
        
        w3,relu2=self.sess.run([now[0], last[-1]], feed_dict=feed_tep)
        w3=w3.T#默认下fc3->w 为4096*20 
        relu2=relu2[batch_select]
        #print(relu2.shape)           
        
        return self.cal_fc2fc(indexs_min, indexs_max,w3, relu2)
    #pool5 Tensor("pool4_1:0", shape=(30, 7, 7, 512)
        
    
    def cal_cnn(self,cnnfeature, kernel, indexs_min, indexs_max,  stride, pooling=False, selnum=1):
        '''
        :将maxpool 和  卷积  的反向合到一个函数里面
        cnnfeature:the 上一层输出的featuremap，被卷积的[width,height, channel]
        kernel：4维度的kernel weight， shape:[filter_height, filter_width, in_channels, out_channels]
        indexs:a map key is index, val is count, will trans index to coordinate by the shape cal use cnnfeature shape and stride shape
        stride : 2 [h,w]
        '''
        ret_min={}
        ret_max={}
        fshape=np.array(cnnfeature.shape)#feature shape
        fshape_wh=fshape[0:2]
        
        outshape_wh=np.ceil(fshape_wh/stride).astype(int)
        
        if not pooling:
            outshape=np.append(outshape_wh, kernel.shape[-1])#根据这里算出来的shape将index转化为坐标   默认SAME模式
        else:
            outshape=np.append(outshape_wh, fshape[-1])#根据这里算出来的shape将index转化为坐标   默认SAME模式
        
        print ('this layer\'s input shape:',outshape)
        
        kernel_wh=kernel.shape[0:2]
        
        pad=self.get_padding(fshape_wh, kernel_wh, stride)
        feature_withpad=np.append(pad, 0)+fshape
        tep_feat=np.zeros(feature_withpad)
        
        tep_feat[pad[0]//2:pad[0]//2+fshape_wh[0], pad[1]//2:pad[1]//2+fshape_wh[1], :]=cnnfeature
        
        for i in indexs_min.keys():
            
            coor=np.array(self.index2shape(i, outshape))
           # print (coor, stride)
            oricoor_hw=coor[0:2]*stride#padding 后坐标
            
            #print(oricoor)
            if not pooling:
                tepf=tep_feat[oricoor_hw[0]:oricoor_hw[0]+kernel_wh[0], \
                              oricoor_hw[1]:oricoor_hw[1]+kernel_wh[1], :kernel.shape[2]]
                tep_mult=tepf*kernel[:,:,:,coor[-1]]#3 -dim
            else:
                tepf=tep_feat[oricoor_hw[0]:oricoor_hw[0]+kernel_wh[0], \
                              oricoor_hw[1]:oricoor_hw[1]+kernel_wh[1], coor[-1]]
                tep_mult=tepf#2 dim
            
            
            
            #!!!!!!!!!!!!!!!!!!!
            min_sel=np.argsort(tep_mult.reshape(-1))[:selnum]
            for ti in min_sel:
                #print(ti)
                coor_inkernel=self.index2shape(ti, tep_mult.shape)
                global_coor=oricoor_hw+coor_inkernel[0:2]-pad//2
                
                #如果坐标小于0，说明都在补的pad上
                if (global_coor>=0).all() and (global_coor<fshape_wh).all():
                    if not pooling:
                        full_coor=np.append(global_coor, coor_inkernel[-1])
                    else:
                        full_coor=np.append(global_coor, coor[-1])
                    
                    
                    
                    ind=self.shape2index(fshape, full_coor)
                    if ind in ret_min.keys():
                        ret_min[ind]+=indexs_min[i]
                    else:
                        ret_min[ind]=indexs_min[i]
        #////////////////////////////////////////////////////////////////////////////////
        for i in indexs_max.keys():
            #print('\n',i,' index2coor:',outshape)
            coor=np.array(self.index2shape(i, outshape))
            oricoor_hw=coor[0:2]*stride#padding 后坐标
            
            #print(oricoor)
            if not pooling:
                tepf=tep_feat[oricoor_hw[0]:oricoor_hw[0]+kernel_wh[0], \
                              oricoor_hw[1]:oricoor_hw[1]+kernel_wh[1], :kernel.shape[2]]
                tep_mult=tepf*kernel[:,:,:,coor[-1]]#3 -dim
            else:
                tepf=tep_feat[oricoor_hw[0]:oricoor_hw[0]+kernel_wh[0], \
                              oricoor_hw[1]:oricoor_hw[1]+kernel_wh[1], coor[-1]]
                tep_mult=tepf#2 dim
            
            
            #!!!!!!!!!!!!!!!!!!!
            max_sel=np.argsort(tep_mult.reshape(-1))[-selnum:]
            for ti in max_sel:            
                coor_inkernel=self.index2shape(ti, tep_mult.shape)
                global_coor=oricoor_hw+coor_inkernel[0:2]-pad//2
                
                #如果坐标小于0，说明都在补的pad上
                if (global_coor>=0).all() and (global_coor<fshape_wh).all():
                    if not pooling:
                        full_coor=np.append(global_coor, coor_inkernel[-1])
                    else:
                        full_coor=np.append(global_coor, coor[-1])
                    
                    #print (full_coor)
                    #print (cnnfeature[full_coor[0], full_coor[1], full_coor[2]])    
                    
                    ind=self.shape2index(fshape, full_coor)
                    if ind in ret_max.keys():
                        ret_max[ind]+=indexs_max[i]
                    else:
                        ret_max[ind]=indexs_max[i]
        return ret_min, ret_max
            
             
        
    
    def get_padding(self, ni, k, s):
        '''
        :输入的形状（没有batch）   kernel形状（同ni对应）    stride形状  （维数要一致）
        https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
        '''
        ret=[]
        for i in range(len(ni)):#只在长宽上有padding
            if ni[i]%s[i]==0:
                ret.append( max(k[i]-s[i],0))
            else:
                ret.append( max(k[i]-ni[i]%s[i],0))
        return np.array(ret)
    

    
    def cal_fc2fc(self, indexs_min, indexs_max,now_weight, lastfeature):
        '''
        :根据本层fc需要向后bp的元素index，将每个元素向上bp，包括最大和最小两个方向，需求输入为本层需要bp的元素下标， 本层的weight，和上层输出的feature，返回上层feature中需要bp的元素下标
        :这里weight输入为本层所有weight，feature输入为选定的一张图片的feature，即去掉了batch维度
        '''
        
        ret_min={}
        ret_max={}
        
        for i in indexs_min:
            tepw=now_weight[i]
            teprelu=lastfeature
            #print (tepw.shape, teprelu.shape)
            tepvec=tepw*teprelu
            #print (tepvec.shape)
            tepminind=self.get_minindexs(tepvec)
            #print ("fc layer last feat--min:",tepminind, '\n',lastfeature[tepminind])
            for j in tepminind:
                if j in ret_min.keys():
                    ret_min[j]+=indexs_min[i]
                else:
                    ret_min[j]=indexs_min[i]
                    
        for i in indexs_max:
            tepw=now_weight[i]
            teprelu=lastfeature
            #print (tepw.shape, teprelu.shape)
            tepvec=tepw*teprelu
            #print (tepvec.shape)
            tepmaxind=self.get_maxindexs(tepvec)
            #print ("fc layer last feat--max:",tepmaxind, '\n',lastfeature[tepmaxind])
            for j in tepmaxind:
                if j in ret_max.keys():
                    ret_max[j]+=indexs_max[i]
                else:
                    ret_max[j]=indexs_max[i]
        return ret_min, ret_max
    
    
    def get_minindexs(self,vec,numselect=12):
        '''
        :用于取出给定的向量中的最小部分 这里可能需要一些套路
        '''
        
        return np.argsort(vec)[:numselect]
    
    def get_maxindexs(self,vec, numselect=12):
        '''
        :用于取出给定的向量中的最da部分 这里可能需要一些套路
        '''
        
        return np.argsort(vec)[-numselect:]
        
        
    def get_onelayer_tensors(self,name):
        '''
        get one namescope's tensors through name
        '''
        ret=[]
        classname = self.__dict__
        classname[name+'_w']=self.graph.get_tensor_by_name(name+'/weights:0')
        ret.append(classname[name+'_w'])
        
        classname[name+'_b']=self.graph.get_tensor_by_name(name+'/biases:0')
        ret.append(classname[name+'_b'])
        
        classname[name+'_out']=self.graph.get_tensor_by_name(name+'/BiasAdd:0')
        ret.append(classname[name+'_out'])
        
        if name!='fc3':
            if name.startswith('fc'):
                classname[name+'_relu']=self.graph.get_tensor_by_name(name+'/Relu:0')
                ret.append(classname[name+'_relu'])
            elif name.startswith('con'):#conv层的relu层名字比较特殊
                classname[name+'_relu']=self.graph.get_tensor_by_name(name+':0')
                ret.append(classname[name+'_relu'])
        
        return ret
        
         
    
    def load_model(self,modelpath=modelpath):
        saver = tf.train.import_meta_graph(op.join(modelpath,'model_keep-29999.meta'))
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        print ('restore weights done!')
        
        
        
        
        
    def show_feature_oneimg(self,fea):
        
        pass
    
    def show_a_vector(self,vec): 
        tep=np.sort(vec)
        #tep=vec
        #plt.scatter(range(len(dif)), dif, c=cValue[ind%len(cValue)],s=1,marker='.')
        plt.scatter(range(len(tep)), tep,s=1,marker='.')
        plt.show()
    
    def show_weight(self,w3):
        '''
        w3 shape:[last lauer's shape, out shape] like [4096,20(class num)]
        '''
        cValue = ['r','y','g','b','c','k','m']
        print (type(w3))
        w3=np.mat(w3)
        for ind,i in enumerate(w3.T):    
            tep=i.getA()[0]
            print (tep.shape)
            
            dif=[]
            for j in range(len(tep)-1):
                dif.append(tep[j]-tep[j+1])
            tep=np.sort(tep)
            #plt.scatter(range(len(dif)), dif, c=cValue[ind%len(cValue)],s=1,marker='.')
            plt.scatter(range(len(tep)), tep, c=cValue[(ind+1)%len(cValue)],s=1,marker='.')
            plt.show()
        
    


if __name__ == '__main__':
    with tf.Session() as sess:
        tep=bp_model(sess)
        




