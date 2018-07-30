#coding:utf-8
'''
Created on 2018��7��30��

@author: sherl
'''
import tensorflow as tf

if __name__ == '__main__':
    a=tf.placeholder(tf.int32,name='test')
    b=tf.placeholder(tf.int32)

    sess=tf.Session()
    
    resu=a+b
    print(sess.run(resu,{a:23,b:34}))
    