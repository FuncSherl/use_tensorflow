#coding:utf-8
'''
Created on 2018��9��24��

@author: sherl
'''
from xml.dom import minidom
import cv2,os
import os.path as op

classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]



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
    
    
def get_label_file(dir_labels):
    '''
    input:dir to these label files
    out:a map with key->class name, val->a list of file names in this class,for example:
        aeroplane ['2008_000021', '2008_000033']
        bicycle ['2008_000036', '2008_000090']
        bird ['2008_000054', '2008_000095']
        boat ['2008_000007', '2008_000036']
    '''
    ret={}
    for label,i in enumerate(classes):
        ret[i]=[]
        fname=i+'_trainval.txt'
        tep=op.join(dir_labels, fname)
        with open(tep, 'r') as f:
            lin=f.readlines()
            for j in lin:
                t=j.strip().split()
                if int(t[-1])==1:
                    ret[i].append(t[0].strip())
    '''
    cnt=0
    for i in ret:               
        cnt+=len(ret[i])
    print (cnt)
    '''
    return ret
        
    

if __name__ == '__main__':
    rootdir='F:\DL_datasets\PascalVOC2012\VOC_2012_all\VOC2012_trainval'
    testdir='F:\DL_datasets\PascalVOC2012\VOC_2012_all\VOC2012_test'
    
    annotation_dir=op.join(rootdir, 'Annotations')
    label_dir=op.join(rootdir, 'ImageSets\Main')
    get_label_file(label_dir)
    
    for i in os.listdir(annotation_dir):
        tep=op.join(annotation_dir, i)
        show_objects(tep)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    