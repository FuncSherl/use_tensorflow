# coding:utf-8
'''
Created on 2018��7��30��

@author: sherl
'''
import tensorflow as tf
import numpy as np
import math,random,time
import matplotlib.pyplot as plt
import datetime

NUM_CLASSES = 2
NUM_INPUTS=2
hidden1_units = 10
hidden2_units=10
batchsize = 100
RANGE_circle=4
draw_gap=20
max_step=50000
lr=0.04

today = datetime.date.today()   #datetime.date类型当前日期
str_today = str(today)   #字符串型当前日期,2016-10-09格式



def inference(points):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([2, hidden1_units],
                                stddev=1.0 / math.sqrt(float(2))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(points, weights) + biases)
        
        tf.summary.scalar('hid1_bias',tf.reduce_max(biases))
        tf.summary.histogram('hid1_bias',biases)
        
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(2))), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        
        tf.summary.scalar('hid2_bias',tf.reduce_max(biases))
        tf.summary.histogram('hid2_bias',biases)
        
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

      Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].

      Returns:
        loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar('loss',loss)
    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.
    
      Creates a summarizer to track the loss over time in TensorBoard.
    
      Creates an optimizer and applies the gradients to all trainable variables.
    
      The Op returned by this function is what must be passed to the
      `sess.run()` call to cause the model to train.
    
      Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
    
      Returns:
        train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    #tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def get_batch_data():
    '''
            这里数据的产生时标签决定了最终图形，如生成数据时在圆里面为1，则训出来的网络就按照这个圆取拟合
    '''
    dat=[]
    label=[]
    

    for i in range(batchsize):
        x=random.random()*RANGE_circle-RANGE_circle/2#in -2->2
        y=random.random()*RANGE_circle-RANGE_circle/2#in -2->2
        dat.append([x,y])
        #print (x,":",y)
        #在半径为1的圆里面为1 
        
        if x**2+y**2<=5 and x**2+y**2>1:#拟合一个圆
        #if abs(x)+abs(y)<=1:#拟合菱形
        #if y<=1/x:#拟合y=1/x
            #print (x,":",y,"in the circle")
            label.append(1)
        else: label.append(0)
    return dat,label
 
plt.ion()

fig = plt.figure() 


axes = fig.add_subplot(121)
axes.axis('equal')
plt.title('test fitness')


axes2 = fig.add_subplot(122)
#axes2.axis("equal")
plt.title("loss")



loss_list=[]
 
def evaluate(sess, logits, dat_place):
    kep_in=[]
    kep_out=[]
    for i in range(200):
        dat,lab=get_batch_data()
        l=sess.run(logits, feed_dict={dat_place:dat})
        for id,i in enumerate(l):
            if i.argmax()==0:
                kep_out.append(dat[id])
            else: kep_in.append(dat[id])
    
    

    
    #plt.cla()#清屏
    axes.cla()
    axes2.cla()
    
    axes2.grid(True, color = "r")
    axes2.plot(range(len(loss_list)), loss_list)
    
    axes.grid(True, color = "r")
    if len(kep_out)>0:
        tep=np.array(kep_out)
        axes.scatter(tep[:,0],tep[:,1],color='g',s=1,marker='.')#外面的是
        
    
    #print (kep_in)
    if len(kep_in)>0:#刚开始时weight都是随机的，所以前向的时候可能一个预测结果都不在圆里面，这时kep_in为空，要有一定判断
        tep2=np.array(kep_in)
        axes.scatter(tep2[:,0],tep2[:,1],color='b',s=1,marker='.')#里面的是blue
    
    plt.suptitle(u'test')   #对中文的支持很差！
    plt.pause(0.0001)
        

    

def start():
    dat_place = tf.placeholder(tf.float32, shape=(batchsize, NUM_INPUTS))
    label_place= tf.placeholder(tf.float32, shape=(batchsize))
    
    logits=inference(dat_place)
    los=loss(logits, label_place)
    
    train_op=training(los, lr)
    
    init = tf.global_variables_initializer()#初始化tf.Variable
    sess = tf.Session()
    
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    
    sess.run(init)
    stti=time.time()
    for step in range(max_step):
        dat,lab=get_batch_data()
        #print (dat)
        _, loss_value, summary_resu = sess.run([train_op, los, merged], feed_dict={dat_place:dat, label_place:lab})
        
        if (step+1)%100==0:
            loss_list.append(loss_value)
            print("step:",step," loss=",loss_value)
            
        if (step+1)%draw_gap==0:
            evaluate(sess, logits, dat_place)
        writer.add_summary(summary_resu, step)
    print ("done!!! time:",(time.time()-stti))
    
    writer.close()
    
    
    


if __name__ == '__main__':
    start()
    
    plt.ioff()
    plt.savefig("./"+str_today+'_lr'+str(lr)+'_max_step'+str(max_step)+'_hidden1_units'+str(hidden1_units)+".png")
    #plt.show()
    
    '''
    #b = tf.Variable([-.3], dtype=tf.float32)
    当你调用tf.constant时常量被初始化，它们的值是不可以改变的，而变量当你调用tf.Variable时没有被初始化，
在TensorFlow程序中要想初始化这些变量，你必须明确调用一个特定的操作

init = tf.global_variables_initializer()
sess.run(init)

https://blog.csdn.net/lengguoxing/article/details/78456279
    '''
    #sess = tf.Session()
    
    # sess.run(tf.global_variables_initializer())
    
