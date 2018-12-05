# coding:utf-8
'''
Created on 2018��7��30��

@author: sherl
'''
import tensorflow as tf
import numpy as np
import math,random,time
import matplotlib.pyplot as plt
from datetime import datetime
import os.path as op
import  use_tensor.draw_sigmoid.test_draw_sigmoid as draw_sigmoid
import  use_tensor.draw_circle.test_draw_circle as draw_circle
from scipy.special import comb, perm

from mpl_toolkits.mplot3d import Axes3D

TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
batchsize=draw_sigmoid.batchsize

#s数据跨度，即上下范围
max_data=2
min_data=-2


num_step=300
div_step=(max_data-min_data)*1.0/num_step

num_derv=3   #泰勒展开到多少次

num_classes=2

modelpath='./logs/2018-12-05_18-52-26' #'./logs/2018-12-01_13-32-33' #'./logs/2018-12-04_21-16-10' #

class cal_tailor:
    def __init__(self,sess, modelpath=modelpath):
        self.sess=sess
        self.graph = tf.get_default_graph() 
        self.load_model(modelpath)  
        
        self.logit=self.graph.get_tensor_by_name('softmax_linear/add:0') 
        self.dat_place=self.graph.get_tensor_by_name('input_img:0') 
        self.label_place=self.graph.get_tensor_by_name('input_lab:0') 
        #self.training=self.graph.get_tensor_by_name('Placeholder_2:0') 
        
        #self.get_batch_data=draw_sigmoid.get_batch_data 
        self.get_batch_data=draw_circle.get_batch_data 
        
        for i in tf.trainable_variables():
            print(i)
        print (self.graph.get_all_collection_keys())
        
        
    def get_onedimval(self, point,dim, num=num_step, start=min_data, end=max_data):
        '''
        :跑模型得出结果
        point:在哪个点处展开
        dim:获取该点处哪个维度的网格点
        num:获取多少网格点，决定了每个点之间的间隔
        start：该维度数据取点的起点
        end：下界
        return:[num, num_classs]  2 dim list
        '''
        ret=[]
        point=np.array(point)
        step=1.0*(end-start)/num
        dat=np.zeros([batchsize, num_classes])
        print('point:',point)
        #print('dim:',dim,'  from:',start, '  to  ',end)
        for i in range(num):
            dat[i%batchsize]=point.copy()
            dat[i%batchsize][dim]=start+i*step
            if (i+1)%batchsize==0 or i==(num-1):
                lot=self.sess.run(self.logit, feed_dict={self.dat_place:dat})
                ret.extend(lot[0:i%batchsize+1])
                
        ret=np.array(ret)
        #print (ret.shape)
        return ret
    
    
    def get_values(self, dim_num=2, num=num_step, start=min_data, end=max_data, class_choose=0):
        '''
        :利用get_onedimval函数，这里先选定一个维度，针对该维度上的每一个step点，利用get_onedimval对该点对的另一个维度每个点取网络跑出来的logit值
        :最终获得一个2维的存储，里面对应网格状的点的对应class_choose的输出的值
        ,num是取多少点，step是数据间隔，注意有可能因为计算机精确度的原因加上step后结果并不改变
        '''
        step=1.0*(end-start)/num
        kep_val=np.zeros([num]*dim_num)
        
        #！！！！！！！！！！！！对于大于2维度的这里应该修改
        for i in range(num):
            x=i*step+start
            tp=[x,0]
            print('\n',i,'/',num,'-->',tp)
            tep=self.get_onedimval(tp, 1, num, start, end)
            kep_val[i]=tep[:,class_choose]
            #print (tep)
            
        return kep_val
            
        
    
        '''
        print (tep)
        #fig = plt.figure() 
        plt.scatter(list(range(len(tep))),tep[:,0],s=1,marker='.')
        plt.scatter(list(range(len(tep))),tep[:,1], color="orange",s=1,marker='.')
        plt.scatter(list(range(len(tep))),tep[:,1]+tep[:,0], color="red",s=1,marker='.')
        plt.show()
        '''
        
    def get_all_derivative(self,point, div_num=num_derv, num=num_step, start=min_data, end=max_data, class_choose=0):
        '''
        :这里以2位point为例,考虑利用递归
        :调用算导数的递归函数，并且分配一个空间存储对不同导数次数的x，y的导数值
        '''
        step=1.0*(end-start)/num
        all_der=np.zeros([div_num]*len(point))#这里行列分别代表x的和y的n次导数，应有f(x,y)dxy=f(x,y)dyx
        #print (all_der.shape)
        
        #这里选择class
        indata=self.get_values(len(point), num, start, end, class_choose)
        
        index=list(map(lambda x:int((x-start)//step ), point  ))
        all_der=self.recursive_cal_deribatice(indata, all_der, index, step)
        print (all_der[0][0])
        return all_der
    
    
    def recursive_cal_deribatice(self,indata,all_der,index=[num_step//2,num_step//2], step=div_step,cnt=0):
        '''
        :递归计算行列的导数，对第一行和列，是原数据的求导，对第二行和列，是以f(x,y)dx,y为基础计算，如此可以递归
        indata:输入的数据，一开始输入源数据
        all_der:记录导数的矩阵
        point:输入数据中以那个为中心，同上面的point不同，这里应为[len/2，len/2],注意分别代表x,y
        cnt：当前递归到那一行了
        '''
        #下面只适合二维
        print('recursive in :',cnt)
        #print (index)
        all_der[cnt,cnt]=indata[index[0],index[1]]
        
        tep_x=indata[index[0]].copy()#固定x的index那一行的data
        tep_y=indata[:,index[1]].copy()
        
        for i in range(cnt+1, all_der.shape[0]):#填充x方向的导数，应该用tep_y
            tep_y=self.cal_derivative(tep_y,step)
            print (i,tep_y[0:10])
            all_der[i,cnt]=tep_y[index[0]]
            
        for i in range(cnt+1, all_der.shape[1]):#填充y方向的导数，应该用tep_x
            tep_x=self.cal_derivative(tep_x,step)
            all_der[cnt,i]=tep_x[index[1]]
        
        if cnt<all_der.shape[0]-1:
            print ('calculating dxdy for next loop...')
            for i in range(indata.shape[0]):
                indata[i]=self.cal_derivative(indata[i],step)
            for i in range(indata.shape[1]):
                indata[:,i]=self.cal_derivative(indata[:,i],step)
            return self.recursive_cal_deribatice(indata, all_der, index,step, cnt+1)
        else: 
            return all_der
        
        
        
    def cal_derivative(self, l,  step=div_step, cnt=max(min(num_step//50, 12), 3)):
        '''
        :算一个list的数的导数
        l:list of data
        cnt: how many data to cal one derivative
        '''
        #print ('derivate from:',l)
        ret=[]
        tep=cnt//2
        for i in range(0,len(l)):
            st=max(0, i-tep)
            ed=min(len(l)-1, i+tep)
            cnt_all=0
            for j in range(st,ed):
                #print (j,l[j+1],l[j],(l[j+1]-l[j]),step)
                cnt_all+=(l[j+1]-l[j])/step
                #if math.isnan((l[j+1]-l[j])/step):print (j,l[j+1],l[j],(l[j+1]-l[j]),step), exit()
            ret.append(cnt_all/(ed-st))
        #print ('derivate to:',ret)
        return np.array(ret)
    
    
    def test_derivative(self):
        '''
        :对上面的求导函数测试
        '''
        tep=[]
        r=1000
        step=2.0*np.pi/r
        for i in range(r):
            tep.append(math.sin(i*step))
        ret=self.cal_derivative(tep, step)
        print(len(tep),len(ret))
        
        plt.grid(True, color = "b")
        plt.scatter(list(range(r)),tep, color="orange",s=1,marker='.')
        plt.scatter(list(range(r)),ret, color="red",s=1,marker='.')
        plt.show()
        
    
    
    def test_tailor(self,point_xk=[0,0],div_num=num_derv, num=num_step, start=min_data, end=max_data,class_choose=0):#excited
        step=1.0*(end-start)/num
        der=self.get_all_derivative(point_xk, div_num, num, start, end,class_choose)
        
        print (der)
        
        #上面获取的导数矩阵，下面对比模型输出的值与拟合的值对比
        point=[0.5,0.5]
        dimval=self.get_onedimval(point, 0, num, start, end)
        for i in range(num):
            tep=np.array(point, dtype=np.float32)
            tep[0]=start+i*step
            res=self.tailor_2(der,  point_xk, tep)
            print ('\npoint:',tep)
            print ('test taior-->ori:',dimval[i],'  tailor:', res)
            
        der2=self.get_all_derivative(point_xk,div_num, num, start, end, 1-class_choose)
        
        #再获取另一个类的导数矩阵，用于概率判别得到准确率
        cnt_true=0
        cnt_all=0
        for j in range(100):
            dat,lab=self.get_batch_data()#!!!!!!!!!!!!!!!!!!!
            for ind,i in enumerate(dat):
                p1=self.tailor_2(der,  point_xk, i)
                p2=self.tailor_2(der2,  point_xk, i)
                if np.argmax([p1,p2])==lab[ind]: cnt_true+=1
            cnt_all+=batchsize
        print ('the tailor accu:', cnt_true/cnt_all)
        
    def draw3d_ori_talior(self,point_xk=[0,0],div_num=num_derv,num=num_step, start=min_data, end=max_data, class_choose=0):
        step=1.0*(end-start)/num
        
        z=self.get_values(len(point_xk),num, start,end,class_choose)
        xa=np.arange(start,end, step)
        ya=np.arange(start,end, step)
        x,y=np.meshgrid(xa,ya)    # x-y 平面的网格
        
        fig = plt.figure()
        # 创建3d图形的两种方式
        # ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')
        # rstride:行之间的跨度  cstride:列之间的跨度
        # rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现
        #vmax和vmin  颜色的最大值和最小值
        #ax.plot_surface(x,y,z, rstride=1, cstride=1)#, cmap=plt.get_cmap('rainbow')
        #plt.show()
        
        
        der=self.get_all_derivative(point_xk,div_num, num, start, end, class_choose)
        for i in range(len(xa)):
            for j in range(len(ya)):
                z[i][j]=self.tailor_2(der,  point_xk, [xa[i],ya[j]])
                
        #print (x.shape, y.shape, z.shape)
        ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        plt.show()
        
            
        
    def tailor_2(self, all_der,  point_xk, point):
        resu=0
        
        point_xk=np.array(point_xk)
        point=np.array(point)
        x_xk=point-point_xk
        
        for i in range(all_der.shape[0]):
            for j in range(i+1):#x的导数数目
                '''
                print (math.factorial(i))
                print (all_der[j][i-j])
                print ( x_xk[0]**j, comb(i,j))
                '''
                resu+=1.0/math.factorial(i)*all_der[j][i-j]*(x_xk[0]**j)*(x_xk[1]**(i-j))*comb(i,j)
        return resu
                
        
        
        
    def eval_model(self):
        cnt_true=0
        cnt_all=0
    
        for i in range(100):
            dat,lab=self.get_batch_data()
            l=self.sess.run(self.logit, feed_dict={self.dat_place:dat})
            
            cnt_all+=batchsize
            tep=np.sum(np.argmax(l,axis=1)==lab)
            cnt_true+=tep
            #print ('eval one batch:',tep,'/',batchsize,'-->',tep/batchsize)
            
        print ('\neval once, accu:',cnt_true/cnt_all,'\n')
        
    
    def load_model(self,modelpath=modelpath):
        saver = tf.train.import_meta_graph(op.join(modelpath,'model_keep-49999.meta'))
        saver.restore(self.sess, tf.train.latest_checkpoint(modelpath))
        print ('restore weights done!')
        
        
        
        
        
if __name__ == '__main__':
    with tf.Session() as sess:
        tep=cal_tailor(sess)
        tep.eval_model()
        #tep.get_onedimval([0,1])
        #tep.get_values([5,0])
        #tep.test_derivative()
        tep.test_tailor()
        tep.draw3d_ori_talior()
        
        
        
        
        
        
    