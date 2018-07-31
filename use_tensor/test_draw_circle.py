#coding:utf-8
'''
Created on 2018��7��30��

@author: sherl
'''
import tensorflow as tf

if __name__ == '__main__':
    a=tf.placeholder(tf.int32)
    b=tf.placeholder(tf.int32)
    vb = tf.Variable([-.3], dtype=tf.float32)
    '''
    #b = tf.Variable([-.3], dtype=tf.float32)
    当你调用tf.constant时常量被初始化，它们的值是不可以改变的，而变量当你调用tf.Variable时没有被初始化，
在TensorFlow程序中要想初始化这些变量，你必须明确调用一个特定的操作

init = tf.global_variables_initializer()
sess.run(init)

https://blog.csdn.net/lengguoxing/article/details/78456279
    '''
    sess=tf.Session()
    
    resu=vb+0.2
    print (vb)
    
    #sess.run(tf.global_variables_initializer())
    
    print(sess.run(resu,{a:23,b:34}))
    print (vb)
    