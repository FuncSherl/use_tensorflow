# coding:utf-8
'''
Created on 2018��7��30��

@author: sherl
'''
import tensorflow as tf
import math,random
from click.core import batch
from sklearn.svm.tests.test_svm import test_linear_svx_uppercase_loss_penality_raises_error

NUM_CLASSES = 2
NUM_INPUTS=2
hidden1_units = 10
batchsize = 100


def inference(points):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([2, hidden1_units],
                                stddev=1.0 / math.sqrt(float(2))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(points, weights) + biases)
        
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden1, weights) + biases
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
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


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
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def get_batch_data():
    dat=[]
    label=[]
    
    rang=4
    for i in range(batchsize):
        x=random.random()*rang-rang/2#in -2->2
        y=random.random()*rang-rang/2#in -2->2
        dat.append([x,y])
        
        #在半径为1的圆里面为1 
        if x**2+y**2<=1:
            print (x,":",y,"in the circle")
            label.append(1)
        else: label.append(0)
    return dat,label
    

def start():
    dat_place = tf.placeholder(tf.float32, shape=(batchsize, NUM_INPUTS))
    label_place= tf.placeholder(tf.float32, shape=(batchsize))
    
    logits=inference(dat_place)
    loss=loss(logits, label_place)
    
    train_op=training(loss, 0.01)
    
    init = tf.global_variables_initializer()#初始化tf.Variable
    sess = tf.Session()
    
    sess.run(init)
    for step in range(1000):
        dat,lab=get_batch_data()
        
        _, loss_value = sess.run([train_op, loss], feed_dict={dat_place:dat, label_place:lab})
        
        if step%100==0:
            print("step:",step," loss=",loss_value)
    
    
    
    
    


if __name__ == '__main__':
    
    
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
    
