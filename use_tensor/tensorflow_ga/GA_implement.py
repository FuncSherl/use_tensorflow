#coding:utf-8
'''
Created on Sep 3, 2019

@author: cloudin
'''
import tensorflow as tf
import numpy as np
from sqlalchemy.sql.base import _bind_or_error

order_properity_cnt=6  #color_depth,color_seris,coloe_group,deadline,client,material
dev_properity_cnt=6    #color_depth,color_series, last_color_group, last_plantime, all_colorcnt, lat_color_count
dnas_len=3          #how many dnas
poses_per_timegap=8    #每次运算时的时间间隔对应的几次染


wash_time=5*60  #5 hours for one wash

render_time=8*60  #每一缸耗时

delay_cost=1000 #每天延迟的cost
depthchange_cost=500

#group之间的颜色切换cost，同group的为depthchangec_cost
test_change_group=np.array([[depthchange_cost,450, 400, 450],
                            [450, depthchange_cost, 350, 400],
                            [400, 350, depthchange_cost, 600],
                            [450, 400, 600, depthchange_cost]], dtype=np.int32)

#需要多少次染色才洗缸
test_change_cnt=np.array([7, 6, 8, 5, 6], np.int32)

#测试用例，order列表
testing_orders=np.array([[2, 2, 3, 2*24*60, 2, 4],
                         [1, 2, 3, 1*24*60, 4, 4],
                         [3, 2, 2, 3*24*60, 3, 5],
                         [2, 2, 2, 1*24*60, 3, 5]], dtype=np.int32)

#测试用例，device列表
testing_devs=np.array([[0, 2, 2, 0, 1, 3],
                       [3, 2, 3, 0, 1, 2]], dtype=np.int32)

#测试用例，dna列表
testing_dnas=np.array([[13, 14, 15, 0],
                       [2, 3, 8, 18],
                       [3, 9, 27, 34]] , dtype=np.int32)

#测试用例，order与device的对应列表[order len, device len]
testing_order_dev=np.array([[1, 1],
                            [0, 1],
                            [1, 0],
                            [1, 1]]  ,dtype=np.int32)


class Cal_all:
    def __init__(self, sess, changegroup, changecnt):
        '''
        changegroup: color change cost between group
        cahngecnt:color series change cnt
        
        '''
        self.place_order=tf.placeholder(tf.int32, [None, order_properity_cnt], "order_placeholder")
        self.place_dev=tf.placeholder(tf.int32, [None, dev_properity_cnt], "dev_placeholder")
        self.place_dnas=tf.placeholder(tf.int32, [dnas_len, None], "dnas_placeholder")  #[dnas_len, order_len]
        self.place_order_dev=tf.placeholder(tf.int32, name="order_dev_placeholder")  #order-dev mat,0 or 1 each
        
        self.sess=sess
        
        self.change_group=tf.convert_to_tensor(changegroup, tf.int32)
        self.change_cnt=tf.convert_to_tensor(changecnt, tf.int32)
        
        self.order_len=tf.shape(self.place_order)[0]
        self.dev_len=  tf.shape(self.place_dev)[0]
        
        self.dev_order_cnt=tf.reduce_sum(self.place_order_dev, 0)  #dev len
        self.order_dev_cnt=tf.reduce_sum(self.place_order_dev, 1)  #[order len]
        
    def one_dna2plan(self, dna):
        '''
        由传入负责dna的去重
        根据dna生成plan，其中plan中的order id是下标从1开始，因为其他默认值为0
        return:[dev_len, poses_per_timegap]
        '''
        #tep=dna%(self.order_dev_cnt * poses_per_timegap)
        tep=dna%(self.dev_len * poses_per_timegap)
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
        dev=tf.cast(dev, tf.int32)
        pos=tf.floormod(tep, poses_per_timegap)
        pos=tf.cast(pos, tf.int32)
        '''
        def devind2devid(x):
            tte=tf.where(tf.equal(self.place_order_dev[x[0]], 1))
            return tf.cast( tte[x[1]][0], tf.int32)
        elems = (tf.range(self.order_len), dev)
        print (elems)
        dev=tf.map_fn(devind2devid, elems, dtype=tf.int32, parallel_iterations=10, back_prop=False)
        '''
        
        ret=tf.stack([dev, pos], -1)  #[4,2]
        ret=tf.cast(ret, tf.int32)
        
        rett=tf.scatter_nd(ret, tf.range(1, self.order_len+1),[self.dev_len, poses_per_timegap])
        print ("rett:",rett)
        
        #注意这里order id是从1开始的，因为默认值为0
        return rett
    
    def run_oneorder_on_dev(self, devinfo, orderid):
        '''
        根据device的状态：devinfo，运行一个订单：orderid
        return：更新后的cost和device的状态
        '''
        #devinfo的各个状态[last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt]
        #return tep_cose, last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt
        orderinfo=self.place_order[orderid]
        
        cost_wash,time_wash, last_colorcnt=self.compute_wash_pot(devinfo, orderinfo)
        last_plantime=tf.add(devinfo[3], time_wash)
        cost_render=self.compute_delay(last_plantime, orderinfo[3])
        
        return tf.add(cost_wash, cost_render), orderinfo[0], orderinfo[1], orderinfo[2], tf.add(last_plantime, render_time), last_colorcnt
        
    def compute_delay(self, devtime, ordertime):
        #延期的cost计算
        cost=tf.maximum(devtime+render_time-ordertime, 0)/24/60*delay_cost
        return tf.cast(cost, tf.int32)
        
        
    def compute_wash_pot(self, devinfo, orderinfo):
        '''
        计算洗缸的cost，根据现在dev的状态devinfo和order的要求orderinfo
        return：cost，花费时间（0 or wash_time），color染色计数
        '''
        #color_depth,color_seris,coloe_group,deadline,client,material
        res1=tf.cond(tf.less(orderinfo[0], devinfo[0])   , lambda:True, lambda:False)
        res2=tf.cond(tf.equal(orderinfo[2], devinfo[2]), lambda:False, lambda:True)
        res3=tf.cond(tf.less(devinfo[4], self.change_cnt[  tf.cast(devinfo[1], tf.int32)  ]), lambda:False, lambda:True  )
        
        cost=self.change_group[devinfo[2]][orderinfo[2]]
        cost,costtime, last_colorcnt=tf.cond(res1|res2|res3, lambda:[cost,wash_time, 1],  lambda:[0,0, tf.add(devinfo[4], 1)])
        return cost,costtime,last_colorcnt
    
    
    def process_one_plan(self, devid, planlist_i):
        '''
        利用循环顺序处理一个device上的plan，利用run_oneorder_on_dev，顺序运行一个机器上的所有订单
        devid:该device的id
        planlist_i:该机器上的订单
        return:处理的cost及运行后机器的状态
        '''
        print (planlist_i)
        cost=tf.constant(0, tf.int32)
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
            =tf.cond(tf.equal(planlist_i[i], 0), \
                     lambda:(0, last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt), \
                     lambda:self.run_oneorder_on_dev(  [last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt], planlist_i[i]-1))
            
            cost=tf.add(cost, tep_cose)
            
            return tf.add(i, 1),cost, last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt
        
        i, cost,last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt \
        = tf.while_loop(cond, body, loop_var, parallel_iterations=1)
        
        return (cost, last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt)

    def process_all_plans(self, plan_lists):
        '''
        根据一个dna生成的plan，在对应设备上逐行处理每个plan，多线程
        return：每行分别为[cost,last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt]
        同process_one_plan的返回值，注意是每行不是每列，这里tensor与普通的list填充不同
        '''
        #plan_lists:[dev len, poses_per_timegap] data is order id+1
        #kep_inds=tf.expand_dims(  tf.range(self.dev_len) , -1)
        #elems=( tf.concat([kep_inds, plan_lists], 1) )
        #print (elems)
        elems = (tf.range(self.dev_len), plan_lists)
        print (elems)
        def distribute_plan(x):
            devid=x[0]
            planlist=x[1]
            return self.process_one_plan(devid, planlist)
        
        resu=tf.map_fn(distribute_plan, elems, dtype=(tf.int32,)*6, parallel_iterations=10, back_prop=False)
        #each row is [cost,last_color_depth, last_color_series, last_color_group, last_plan_time, last_color_cnt]
        print ("resu:",resu)
        return resu
        
    def process_one_dna(self, dna):
        #利用上面的函数，将一条dna转换成plan然后运行plan
        plans_lists=self.one_dna2plan(dna)
        resu=self.process_all_plans(plans_lists)
        #resu[0]为所有执行返回的cost的集合
        return resu[0]  #tf.cast( tf.reduce_sum(resu[0]), tf.int32)  #cost sum
    
    def process_all_dna(self):
        #利用上面函数处理所有dna
        elems=self.place_dnas
        def distribute_dnas(x):
            return self.process_one_dna(x)
        
        resu=tf.map_fn(distribute_dnas, elems, dtype=tf.int32, parallel_iterations=10, back_prop=False)
        print (resu)
        return resu
    
    
    
    #tensorflow outside
    def repeat_mod_proc(self, dna, order_dev):
        #一条dna的去重，这里是全局的去重
        #order_dev:[order_len, dev_len]
        order_len=len(dna)
        dev_len=order_dev.shape[1]
        
        order_dev_cnt=np.sum(order_dev, 1)  #[order len]
        ind_orders=np.argsort(order_dev_cnt)
        print (ind_orders)
        kep=np.zeros([dev_len, poses_per_timegap], dtype=np.int8)
        
        cnt=0
        failcnt=0
        while cnt<order_len:
            orderid=ind_orders[cnt]
            tep=dna[orderid]%(dev_len*poses_per_timegap)
            dev_ind=int(tep/poses_per_timegap)
            poses_ind=tep%poses_per_timegap
            if order_dev[orderid][dev_ind] and kep[dev_ind][poses_ind]==0:
                cnt+=1
                kep[dev_ind][poses_ind]=1
                failcnt=0
            else:
                dna[orderid]+=1
                failcnt+=1
                if failcnt>dev_len*poses_per_timegap:
                    return None
        return dna
                
            
                    
        
        
            
            
        
    
    def run_one_batch(self, order_list, dev_list, dnas, order_dev):
        '''
        该类的入口函数，每次送进来一批次的数据，如一天的订单，注意订单时间是分钟的计数
        order_list:[None, order_properity_cnt]
        dev_lsit:[None, dev_properity_cnt]
        dnas:[dnas_len, None]
        order_dev:[order_len, dev_len]
        '''
        orderlen=order_list.shape[0]
        devlen=dev_list.shape[0]
        
        print (dnas)
        print (order_dev)
        for i in range(dnas_len):
            temp=self.repeat_mod_proc(dnas[i], order_dev)
            if temp is None:
                print ('error:dna:',dnas[i]," cann't be fitted")
                return None
            dnas[i]=temp
        print (dnas)
        
        resu=self.process_all_dna()
        
        print (  self.sess.run([self.order_len, self.dev_len], feed_dict={self.place_order:order_list,
                                         self.place_dev:dev_list,
                                         self.place_dnas:dnas,
                                         self.place_order_dev: order_dev})    )
        
        cost_all=self.sess.run([ resu], feed_dict={self.place_order:order_list,
                                         self.place_dev:dev_list,
                                         self.place_dnas:dnas,
                                         self.place_order_dev: order_dev})
        
        return cost_all
    
if __name__=="__main__":
    with tf.Session() as sess:
        tep=Cal_all(sess, test_change_group, test_change_cnt)
        cost_list=tep.run_one_batch(testing_orders, testing_devs, testing_dnas, testing_order_dev)
        print (cost_list)
        
        
        
        
        
        
        
        
        