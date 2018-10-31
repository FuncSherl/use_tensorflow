#coding:utf-8
'''
Created on 2018��9��24��

@author: sherl
'''
from xml.dom import minidom
import cv2,os,random
import os.path as op
from PIL import Image
import matplotlib.pyplot as plt



import tensorflow as tf

#import tensorflow.data.Dataset as dataset
#print(tf.contrib.slim)


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
    
    
def get_label_file(dir_labels,classes=classes):
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

def gen_tfrecord(dir_labels,stage='train',classes=classes):
    '''
    stage in [train, val, test]
    :由于voc中test没有标注，只好用train和val数据集fenbie训练和测试，但是train和val数量都是8000多，训练不够，就把val生成的tfrecord中拷2个（4000数据）到train中
    '''
    ret=[]
    for label,i in enumerate(classes):
        
        fname=i+'_'+stage+'.txt'
        outname='voc_'+stage+'_data'

        tep=op.join(dir_labels, fname)
        with open(tep, 'r') as f:
            lin=f.readlines()
            for j in lin:
                t=j.strip().split()
                if int(t[-1])==1:
                    ret.append([t[0].strip(), label])
    
    if len(ret)<1:
        print ("error:no data in these classes")
        return
    
    random.shuffle(ret)
    
    datadir=op.join(dir_labels, '../../JPEGImages')
    tfrecorddir='./'+outname
    if not op.exists(tfrecorddir):
        os.makedirs(tfrecorddir)
    
    imgs_perfile=2000
    
    
    for ind,i in enumerate(ret):        
        tep=op.join(datadir, i[0]+'.jpg')
        label=i[-1]
        img=Image.open(tep)
        size = img.size
        
        if ind%imgs_perfile==0:
            ftrecordfilename = (outname+".tfrecords_%.3d" % int(ind/imgs_perfile))
            writer= tf.python_io.TFRecordWriter(op.join(tfrecorddir,ftrecordfilename))

        img_raw=img.tobytes()#将图片转化为二进制格式
        
        example = tf.train.Example(
            features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
            'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
        })) 
        writer.write(example.SerializeToString())  #序列化为字符串
        print (ind,'/',len(ret),'  size:',size,'  to dir:',ftrecordfilename)
        
    writer.close()

'''
TFRecord文件中的数据是通过tf.train.Example Protocol Buffer的格式存储的，下面是tf.train.Example的定义

message Example {
 Features features = 1;
};

message Features{
 map<string,Feature> featrue = 1;
};

message Feature{
    oneof kind{
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};

'''
def preprocess_img(image,outlen=224):
    #这里将图片变成224大小的，但是如果用crop可能导致切出来的图片里没有目标物体
    
    image=tf.image.resize_images(image, [230,230])
    #image = tf.image.resize_image_with_crop_or_pad(image, 230, 230)
    image = tf.random_crop(image, [outlen, outlen, 3])
    image = tf.image.random_flip_left_right(image)
    
    
    return image
    
    
def read_tfrecord_batch(tfdir,batchsize=32):
    tep=os.listdir(tfdir)
    tep=list(map(lambda x:op.join(tfdir, x), tep))
    print (tep)
    dataset = tf.data.TFRecordDataset(tep).repeat()
   
    
    def parse(one_element):
        feats = tf.parse_single_example(one_element, features={'data':tf.FixedLenFeature([], tf.string), 
                                                           'label':tf.FixedLenFeature([],tf.int64), 
                                                           'width':tf.FixedLenFeature([], tf.int64),
                                                           'height':tf.FixedLenFeature([], tf.int64)})
        image = tf.decode_raw(feats['data'], tf.uint8)
        label = tf.cast(feats['label'],tf.int32)
        width = tf.cast(feats['width'], tf.int32)
        height= tf.cast(feats['height'], tf.int32)
        
        image=tf.reshape(image,[height,width,3])
        image=preprocess_img(image)
        
        return image,label
    
    dataset=dataset.map(parse,num_parallel_calls=4)#注意把值回赋给dataset
    
    dataset=dataset.batch(batchsize).shuffle(batchsize*10)
    #print("dataset.output_shapes",dataset.output_shapes)
    
    iterator = dataset.make_one_shot_iterator()

    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch
    
    
    

if __name__ == '__main__':
    rootdir=r'F:\DL_datasets\PascalVOC2012\VOC_2012_all'
    #rootdir=r'D:\data_DL\PascalVOC2012\VOC2012_all'
    
    traindir=op.join(rootdir, 'VOC2012_trainval')
    testdir=op.join(rootdir, 'VOC2012_test')
    
    train_annotation_dir=op.join(traindir, 'Annotations')
    train_label_dir=op.join(traindir, 'ImageSets\Main')
    
    test_annotation_dir=op.join(testdir, 'Annotations')
    test_label_dir=op.join(testdir, 'ImageSets\Main')
    #get_label_file(label_dir)
    
    #gen_tfrecord(train_label_dir)
    #gen_tfrecord(train_label_dir,'val')
    
    '''
    for i in os.listdir(annotation_dir):
        tep=op.join(annotation_dir, i)
        show_objects(tep)
    '''
    with tf.Session() as sess:
        ims,las=read_tfrecord_batch('./voc_val_data')##'/media/sherl/本地磁盘1/workspaces/eclipse/use_tensorflow/use_tensor/locate_feature/voc_train_data'
        images,labels=sess.run([ims,las])
        print (images.shape)
        for ind,image in enumerate(images):
            print (classes[labels[ind]])
            plt.imshow(image)
            plt.show()
            

        
        
        
        
        
        
        
        
        
        
        
        
        
    