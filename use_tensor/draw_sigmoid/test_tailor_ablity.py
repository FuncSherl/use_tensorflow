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

def function_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def function_tanh(x):
    ex=np.exp(x)
    nex=np.exp(-x)
    return (ex-nex)/(nex+ex)

def function_xn(x):
    return 6*(x**6)+10*(x**5)+6*(x**4)-3*(x**2)+x+2

def mystery_model(x):
    return function_xn(x)

def cal_derivative( l, cnt, step):
        '''
        :算一个list的数的导数
        l:list of data
        cnt: how many data to cal one derivative
        step:data gap
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

s_min=-5
s_max=5
num_step=1000
step=(s_max-s_min)/num_step

der_step=10

def cal_all_der(oridata, der_num=der_step, point=num_step//2):
    kep=np.zeros([der_num])
    for i in range(len(kep)):
        kep[i]=oridata[point]
        oridata=cal_derivative(oridata, 20, step)
        
    return kep

def tailor_test(derlist, pointxk, point):
    resu=0
    for ind,i in enumerate(derlist):
        resu+=1.0/math.factorial(ind)*i*((point-pointxk)**ind)
    return resu


def start():
    x=np.arange(s_min, s_max, step)
    oridata=np.array(list(map(mystery_model, x)))
    
    plt.grid(True, color = "b")
    plt.scatter(range(len(oridata)),oridata, color="g",s=1,marker='.')
    #plt.scatter(list(range(r)),ret, color="red",s=1,marker='.')
    plt.ylim(-5,5)
    
    derlist=cal_all_der(oridata)
    print (derlist)
    
    tailor_data=[]
    for i in x:
        tailor_data.append(tailor_test(derlist, 0, i))
        
    plt.scatter(range(len(tailor_data)),tailor_data, color="red",s=1,marker='.')
    
    plt.show()
    
    




if __name__ == '__main__':
    start()





















