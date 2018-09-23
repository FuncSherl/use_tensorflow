#coding:utf-8
'''
Created on 2018��9��24��

@author: sherl
'''
from xml.dom import minidom
import cv2,os
import os.path as op


def show_objects(dir_xml):
    doc = minidom.parse(dir_xml)
    root = doc.documentElement
    
    filename=root.getElementsByTagName('filename')[0].childNodes[0].data
    
    file_dir=op.join(op.split(dir_xml)[0],'../JPEGImages/'+filename)
    print (filename)
    
    img=cv2.imread(file_dir)
    
    objects=root.getElementsByTagName('object')
    for i in objects:
        name=i.getElementsByTagName('name')[0].childNodes[0].data
        bndbox=i.getElementsByTagName('bndbox')[0]
        xmin=int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)
        ymin=int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)
        xmax=int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymax=int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [255,0,0], 3)
        
        cv2.putText(img, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
    
    cv2.imshow('test', img)
    cv2.waitKey()
    

if __name__ == '__main__':
    rootdir='F:\DL_datasets\PascalVOC2012\VOC_2012_all\VOC2012_trainval'
    annotation_dir=op.join(rootdir, 'Annotations')
    for i in os.listdir(annotation_dir):
        tep=op.join(annotation_dir, i)
        show_objects(tep)
    