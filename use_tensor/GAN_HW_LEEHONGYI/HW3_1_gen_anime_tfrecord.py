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

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

oridatadir='F:\\DL_datasets\\LEEHongYi_HW3-1\\ori_data\\faces'
extradatadir='F:\\DL_datasets\\LEEHongYi_HW3-1\\extra_data\\images'

outname_ori='anime_tfrecord_ori'
outname_extra='anime_tfrecord_extra'
imgs_perfile=6000

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

def preprocess_img(image,outlen):
    #这里将图片
    ''''''
    image=tf.image.resize_images(image, (outlen,outlen))
    image=tf.cast(image, dtype=tf.uint8)
    
    #image = tf.image.resize_image_with_crop_or_pad(image, 230, 230)
    #image = tf.random_crop(image, [outlen, outlen, 3])
    
    
    
    image = tf.image.random_flip_left_right(image)
    
    
    return image


def read_tfrecord_batch(tfdir='./'+outname_ori,batchsize=32, imgsize=96):
    tep=os.listdir(tfdir)
    tep=random.shuffle(tep)
    tep=list(map(lambda x:op.join(tfdir, x), tep))
    print (tep)
    dataset = tf.data.TFRecordDataset(tep).repeat()
    
    
    def parse(one_element):
        feats = tf.parse_single_example(one_element, features={'data':tf.FixedLenFeature([], tf.string), 
                                                           #'label':tf.FixedLenFeature([],tf.int64), 
                                                           'width':tf.FixedLenFeature([], tf.int64),
                                                           'height':tf.FixedLenFeature([], tf.int64)})
        image = tf.decode_raw(feats['data'], tf.uint8)
        #label = tf.cast(feats['label'],tf.int32)
        width = tf.cast(feats['width'], tf.int32)
        height= tf.cast(feats['height'], tf.int32)
        
        image=tf.reshape(image,[height,width,3])
        image=preprocess_img(image, imgsize)
        
        return image
    
    dataset=dataset.map(parse,num_parallel_calls=4)#注意把值回赋给dataset
    
    dataset=dataset.batch(batchsize).shuffle(batchsize*10)
    #print("dataset.output_shapes",dataset.output_shapes)
    
    iterator = dataset.make_one_shot_iterator()

    image_batch = iterator.get_next()

    return image_batch

def test_showtfimgs(tfdir='./'+outname_ori):
    tep=read_tfrecord_batch(tfdir)
    with tf.Session() as sess:
        while True:
            images=sess.run(tep)
            plt.imshow(images[0])
            plt.show()

def gen_data_all():
    ori=read_imglist(oridatadir)
    write_tfrec(ori, outname_ori)
    
    extra=read_imglist(extradatadir)
    write_tfrec(extra, outname_extra)
    



if __name__ == '__main__':
    #gen_data_all()
    test_showtfimgs()
    
    
    