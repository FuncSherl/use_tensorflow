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
        print ('label prob:',probs[labst[batch_select]])
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

        print (len(minmap),'->',minmap)
        print (len(maxmap),'->',maxmap)
        
        
        
        
        minmap, maxmap=self.get_layerandlastlayer_fc2fc(minmap, maxmap, 'fc3', 'fc2', feed_tep)
        print ('\nfc3-->fc2:')
        print (len(minmap),'->',minmap)
        print (len(maxmap),'->',maxmap)
            
            
        minmap, maxmap=self.get_layerandlastlayer_fc2fc(minmap, maxmap, 'fc2', 'fc1', feed_tep)
        print ('\nfc2-->fc1:')
        print (len(minmap),'->',minmap)
        print (len(maxmap),'->',maxmap)
        
        
        
        
        minmap, maxmap=self.get_layerandlastlayer_fc2pool(minmap, maxmap, 'fc1', 'pool4_1:0', feed_tep)
        print ('\nfc1-->pool4:')
        print (len(minmap),'->',minmap)
        print (len(maxmap),'->',maxmap)
        
        
        
        #这里是fc向卷积的过度，从这里开始元素有了位置信息，注意下标的转化，由一维到多维
        self.get_layerandlastlayer_pool2cnn(minmap, maxmap,  'conv5_3', feed_tep)
        
        
        
        
        
        
        
        
        
        #print(mult[fc2_minindexs[0:3]])
        #/////////////////////////////////////////////////////////////////////////
        
        
        
        
        #self.show_a_vector(mult)
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        
        
        
        #print (w3,'\n',b3)
        #print (w3.T.shape)
        #self.show_weight(w3)
    
        
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
        
        
    def get_layerandlastlayer_pool2cnn(self, indexs_min, indexs_max,  cnnname, feed_tep,pool_stride=np.array([2,2,1]), pool_kernel=np.array([2,2,1])):
        cnn_tensors=self.get_onelayer_tensors(cnnname)
        
        feat=sess.run([cnn_tensors[-1]], feed_dict=feed_tep)[0][batch_select]
        shape=np.array(feat.shape)#feature shape
        poolshape=np.ceil(shape/pool_stride).astype(int)
        
        
        
        
        
        pass
    
    def get_padding(self, ni, k, s):
        if ni%s==0:
            return max(k-s,0)
        else:
            return max(k-ni%s,0)
    
    

    
    
    def cal_cnn(self,cnnfeature, kernel, outindexs, kernelshape, stride):
        '''
        :需要 cnn的输入feature(去掉batch)  卷积后的坐标    卷积kernel  stride 计算对应上层featuremap中的坐标
        outindex:[ [[x,y,z],count], [[x2,y2,z2],count2] ...]
        '''
        pass
        
        
    def get_layerandlastlayer_fc2pool(self, indexs_min, indexs_max, fcname, poolname, feed_tep):
        '''
        :当本层为fc，上层为pool时使用
        '''
        pool5_feat_tensor=self.graph.get_tensor_by_name(poolname)# 'pool4_1:0'
        w1_tensor=self.graph.get_tensor_by_name(fcname+'/weights:0')# 
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
        
        w3,relu2=self.sess.run([now[0], last[-1]], feed_dict=feed_tep)
        w3=w3.T#默认下fc3->w 为4096*20 
        relu2=relu2[batch_select]
        #print(relu2.shape)           
        
        return self.cal_fc2fc(indexs_min, indexs_max,w3, relu2)
    #pool5 Tensor("pool4_1:0", shape=(30, 7, 7, 512)
    
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
            for j in tepmaxind:
                if j in ret_max.keys():
                    ret_max[j]+=indexs_max[i]
                else:
                    ret_max[j]=indexs_max[i]
        return ret_min, ret_max
    
    
    def get_minindexs(self,vec):
        '''
        :用于取出给定的向量中的最小部分 这里可能需要一些套路
        '''
        numselect=12
        return np.argsort(vec)[:numselect]
    
    def get_maxindexs(self,vec):
        '''
        :用于取出给定的向量中的最da部分 这里可能需要一些套路
        '''
        numselect=12
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
        




