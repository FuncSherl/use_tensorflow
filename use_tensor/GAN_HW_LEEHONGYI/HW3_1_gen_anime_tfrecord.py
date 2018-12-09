#coding:utf-8
'''
Created on 2018年12月10日

@author: sherl
'''
from xml.dom import minidom
import cv2,os,random
from datetime import datetime
import os.path as op
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

oridatadir='F:\\DL_datasets\\LEEHongYi_HW3-1\\ori_data\\faces'
extradatadir='F:\\DL_datasets\\LEEHongYi_HW3-1\\extra_data\\images'

outname_ori='anime_tfrecord_ori'
outname_extra='anime_tfrecord_extra'
imgs_perfile=6000

def read_imglist(ddir):
    #prof=op.split(ddir)[-1]
    ret=[]
    for i in os.listdir(ddir):
        tep=op.join(ddir, i)
        ret.append(tep)
        
    print ('read dir:',ddir," done!",len(ret),' imgs..')
    return ret


def write_tfrec(imglist, outdirname):
    tfrecorddir='./'+outdirname
    if not op.isdir(tfrecorddir): os.makedirs(tfrecorddir)
    
    for ind,i in enumerate(imglist):
        img=Image.open(i)
        size = img.size
        
        print ('\nopening ',i, 'size:',size)
        
        if ind%imgs_perfile==0:
            ftrecordfilename = (outdirname+".tfrecords_%.3d" % int(ind/imgs_perfile))
            writer= tf.python_io.TFRecordWriter(op.join(tfrecorddir,ftrecordfilename))

        img_raw=img.tobytes()#将图片转化为二进制格式
        
        
        example = tf.train.Example(
            features=tf.train.Features(feature={
            #'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
            'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
        })) 
        writer.write(example.SerializeToString())  #序列化为字符串
        print (ind,'/',len(imglist),'  size:',size,'  to dir:',ftrecordfilename)
        
    writer.close()
    print ('for all: write to ',tfrecorddir,'->',ind,' images done!!')
    return ind
        
        
        

if __name__ == '__main__':
    ori=read_imglist(oridatadir)
    write_tfrec(ori, outname_ori)
    
    extra=read_imglist(extradatadir)
    write_tfrec(extra, outname_extra)
    
    
    