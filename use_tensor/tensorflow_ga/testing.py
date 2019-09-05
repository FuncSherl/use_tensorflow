#coding:utf-8
'''
Created on Sep 3, 2019

@author: cloudin
'''
import tensorflow as tf
import numpy as np

order_properity_cnt=6  #color_depth,color_seris,coloe_group,deadline,client,material
dev_properity_cnt=6    #color_depth,color_series, last_color_group, last_plantime, all_colorcnt, lat_color_count
dnas_len=3          #how many dnas
poses_per_timegap=8    #每次运算时的时间间隔对应的几次染


wash_cost=200 #
wash_time=5*60  #5 hours for one wash

render_time=8*60  #每一缸耗时

delay_cost=1000 #

testing_orders=[[2, 2, 3, 2*24*60, 2, 4],
                [1, 2, 3, 1*24*60, 4, 4],
                [3, 2, 2, 3*24*60, 3, 5]]

testing_devs=[[0, 2, 2, 0, 1, 3],
              [3, 2, 3, 0, 1, 2]]
test_dnas=[[17, 4, 5],
           [2, 6, 8],
           [3, 9, 27]]
test_order_dev=[[1, 1],
               [0, 1],
               [1, 0]]


class Cal_all:
    def __init__(self, changegroup, changecnt):
        '''
        changegroup: color change cost between group
        cahngecnt:color series change cnt
        
        '''
        self.place_order=tf.placeholder(tf.int32, [None, order_properity_cnt], "order_placeholder")
        self.place_dev=tf.placeholder(tf.int32, [None, dev_properity_cnt], "dev_placeholder")
        self.place_dnas=tf.placeholder(tf.int32, [dnas_len, None], "dnas_placeholder")  #[dnas_len, order_len]
        self.place_order_dev=tf.placeholder(tf.int32, name="order_dev_placeholder")  #order-dev mat,0 or 1 each
        
        self.change_group=tf.convert_to_tensor(changegroup, tf.float32)
        self.change_cnt=tf.convert_to_tensor(changecnt, tf.int32)
        
        self.order_len=tf.shape(self.place_order)[0]
        self.dev_len=  tf.shape(self.place_dev)[0]
        
        self.dev_order_cnt=tf.reduce_sum(self.place_order_dev, 0)  #dev len
        self.order_dev_cnt=tf.reduce_sum(self.place_order_dev, 1)  #[order len]
        
    def one_dna2plan(self, dna):
        '''
        由传入负责dna的去重[dev_len, poses_per_timegap]
        '''
        tep=dna%(self.order_dev_cnt * poses_per_timegap)
        '''
        i=tf.constant(1, dtype=tf.int32)
        def cond(i):
            return tf.less(i,self.order_len)
        def body(i):
            tf.cond(tep[i]==tep[i-1])
            return tf.add(i, 1)
        r = tf.while_loop(cond, body, [i])
        '''
        
        dev=tf.floordiv(tep, poses_per_timegap)  #order len
        pos=tf.floormod(tep, poses_per_timegap)
        
        def devind2devid(x):
            tte=tf.where(tf.equal(self.place_order_dev[x[0]], 1))
            return tte[x[1]][0]
        elems = (tf.range(0, self.order_len, 1), dev)
        dev=tf.map_fn(devind2devid, elems, dtype=tf.int32)
        
        
        ret=tf.stack([dev, pos], -1)
        ret=tf.cast(ret, tf.int32)
        
        rett=tf.scatter_nd(ret, tf.range(1, self.order_len+1, 1),[self.dev_len, poses_per_timegap])
        #注意这里order id是从1开始的，因为默认值为0
        return rett
    
    def run_oneorder_on_dev(self, devinfo, orderid):
        #[last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt]
        #return tep_cose, last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt
        orderinfo=self.place_order[orderid]
        
        cost_wash,time_wash, last_colorcnt=self.compute_wash_pot(devinfo, orderinfo)
        last_plantime=tf.add(devinfo[3], time_wash)
        cost_render=self.compute_delay(last_plantime, orderinfo[3])
        
        return tf.add(cost_wash, cost_render), orderinfo[0], orderinfo[1], orderinfo[2], tf.add(last_plantime, render_time), last_colorcnt
        
    def compute_delay(self, devtime, ordertime):
        cost=tf.maximum(devtime+render_time-ordertime, 0)/24/60*delay_cost
        return cost
        
        
    def compute_wash_pot(self, devinfo, orderinfo):
        #color_depth,color_seris,coloe_group,deadline,client,material
        res1=tf.cond(tf.less(orderinfo[0], devinfo[0])   , lambda:True, lambda:False)
        res2=tf.cond(tf.equal(orderinfo[2], devinfo[2]), lambda:False, lambda:True)
        res3=tf.cond(tf.less(devinfo[4], self.change_cnt[  tf.cast(devinfo[1], tf.int32)  ]), lambda:False, lambda:True  )
        
        cost=tf.cond(res2, lambda:self.change_group[devinfo[2]][orderinfo[2]], lambda:wash_cost)
        cost,costtime, last_colorcnt=tf.cond(res1|res2|res3, lambda:[cost,wash_time, 1],  lambda:[0,0, tf.add(devinfo[4], 1)])
        return cost,costtime,last_colorcnt
    
    def testing_compute_wahs_pot(self):
        devinfo=[]
    
    
    def process_one_plan(self, devid, planlist):
        cost=tf.constant(0, tf.float32)
        #color_depth,color_series, last_color_group, last_plantime, all_colorcnt, lat_color_count
        i=tf.constant(0, dtype=tf.int32)
        last_color_depth=self.place_dev[devid][0]
        last_color_series=self.place_dev[devid][1]
        last_color_group=self.place_dev[devid][2]
        last_plan_time=self.place_dev[devid][3]
        last_color_cnt=self.place_dev[devid][5]
        loop_var=[i,cost,  last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt]
        
        def cond(i,cost,  last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt):
            return tf.less(i,poses_per_timegap)
        
        def body(i, cost,last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt):
            tep_cose, last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt\
            =tf.cond(tf.equal(planlist[i], 0), lambda:[0, last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt]\
                     , self.run_oneorder_on_dev(  [last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt], planlist[i]-1))
            
            cost=tf.add(cost, tep_cose)
            
            return tf.add(i, 1),cost, last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt
        
        i, cost,last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt \
        = tf.while_loop(cond, body, loop_var, parallel_iterations=1)
        
        return cost,[last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt]


elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
alternate = tf.map_fn(lambda x: x[0] * x[1], elems, dtype=tf.int64)

test_order_dev=tf.convert_to_tensor(test_order_dev, dtype=tf.int64)

def devind2devid(x):
    tte=tf.where(  tf.equal(test_order_dev[x[0]], 1)  )
    
    return tte[x[1]][0]
elems2 = (np.array([0,1,2]), np.array([1,0,0]))
dev=tf.map_fn(devind2devid, elems2, dtype=tf.int64)

with tf.Session() as sess:
    a=sess.run([dev, alternate])
    print (a)
        
        
        