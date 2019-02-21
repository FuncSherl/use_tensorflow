import cv2
import os,random
import numpy as np
import os.path as op
import tensorflow as tf
import matplotlib.pyplot as plt

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

train_txt=r'./adobe240fps/train_list.txt'
test_txt=r'./adobe240fps/test_list.txt'

pc_id=2

if pc_id==0: videodir=r'E:\DL_datasets\DeepVideoDeblurring_Dataset_Original_High_FPS_Videos\original_high_fps_videos'  
elif pc_id==1: videodir=r'/media/sherl/本地磁盘/data_DL/Adobe240fps/original_high_fps_videos' #
elif pc_id==2: videodir=r'/media/ms/document/xvhao/use_tensorflow/use_tensor/GAN_slomo/data/original_high_fps_videos'

if pc_id==0: extratdir=r'E:\DL_datasets\DeepVideoDeblurring_Dataset_Original_High_FPS_Videos\extracted_videos' 
elif pc_id==1:extratdir=r'/media/sherl/本地磁盘/data_DL/Adobe240fps/extracted_videos' #
elif pc_id==2:extratdir=r'/media/ms/document/xvhao/use_tensorflow/use_tensor/GAN_slomo/data/extracted_videos'

extratdir_train=op.join(extratdir, 'train')
extratdir_test=op.join(extratdir, 'test')

if pc_id==0: tfrec_dir=r'E:\DL_datasets\DeepVideoDeblurring_Dataset_Original_High_FPS_Videos\tfrecords' 
elif pc_id==1: tfrec_dir=r'/media/sherl/本地磁盘/data_DL/Adobe240fps/tfrecords' #
elif pc_id==2: tfrec_dir=r'/media/ms/document/xvhao/use_tensorflow/use_tensor/GAN_slomo/data/tfrecords'

tfrec_dir_train=op.join(tfrec_dir, 'train')
tfrec_dir_test=op.join(tfrec_dir, 'test')
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
def get_trainframe_dirs():
    return list( map(lambda x:op.join(extratdir_train, x) ,os.listdir(extratdir_train)) )

def get_testframe_dirs():
    return list( map(lambda x:op.join(extratdir_test, x) ,os.listdir(extratdir_test)) )
'''

def video2frame(videop, outpath):
    videoCapture=cv2.VideoCapture(videop)
    if not videoCapture.isOpened():
        print ('failed to open:',videop)
        return 0
    videorate = videoCapture.get(cv2.CAP_PROP_FPS)
    allcnt = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    
    batcnt=0
    success, frame = videoCapture.read()  # 读取第一帧  
    print (frame.shape,'rate:',videorate,' framescnt:',allcnt)
    while success and batcnt <allcnt-1:  
        # frame = frame[0:1536,1200:1800]#截取画面  
        #videoCapture.set(cv2.CAP_PROP_POS_FRAMES, cnt)
                  
        
        tepd='%04d.jpg'%batcnt
        tepd=op.join(outpath, tepd)
        cv2.imwrite( tepd,frame)
        success, frame = videoCapture.read()  # 循环读取下一帧  
        batcnt+=1
        if not success: print ('read error:',batcnt,frame)
         
        
    videoCapture.release() 
    
    print ("done:",videop,':',str(batcnt))
    return batcnt
    

def txt2frames(txtpath, extratdir):
    cnt=0
    with open(txtpath,'r') as f:
        for i in f.readlines():
            tepi=i.strip()
            dirname=op.splitext(tepi)[0]
            movpath=op.join(videodir, tepi)
            outpath=op.join(extratdir, dirname)
            os.makedirs(outpath, exist_ok=True)
            resu=video2frame(movpath, outpath)
            cnt+=resu
        print (txtpath,'done:',cnt,' frames\n')
        
def frames2tfrec(frame_dir, tfrecdir, group_num=3, imgs_perfile=3000, img_shape_required=[640, 360]):#使用cv2.resize时，参数输入是 宽×高 ，与以往操作不同
    '''
    framedir:a root dir, contains many dirs which is one for a video
    group_num:3frame is a group
    '''
    os.makedirs(tfrecdir, exist_ok=True)
    
    cnt_num=0
    videodirs=os.listdir(frame_dir)
    for dirind,i in enumerate(videodirs):
        tepath=op.join(frame_dir, i) #path in video
        framelist=os.listdir(tepath)
        framelist.sort()
        
        imgdata=[]
        for j in range(len(framelist)):
            print (dirind,'/',len(videodirs),'  ',j,'/',len(framelist), cnt_num)
            
            rimg=cv2.imread(op.join(tepath, framelist[j]))
            rimg=cv2.resize(rimg, tuple(img_shape_required))
            imgdata.append(rimg)
            
            if len(imgdata)>group_num: imgdata.pop(0)
            elif len(imgdata)<group_num: continue
                
            #print (imgdata.dtype, type(imgdata))
            
            groupdata=np.concatenate(imgdata,axis=2)
            size=groupdata.shape  #(720, 1280, 9)
            groupdata_raw=groupdata.tobytes()#将图片转化为二进制格式
            
            print (size)
            if cnt_num%imgs_perfile==0:
                ftrecordfilename = (op.split(tfrecdir)[-1].strip()+".tfrecords_%.4d" % int(cnt_num/imgs_perfile))
                writer= tf.python_io.TFRecordWriter(op.join(tfrecdir,ftrecordfilename))
            
            example = tf.train.Example(
            features=tf.train.Features(feature={
            #'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[groupdata_raw])),
            'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]])),
            'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]]))
            })) 
            writer.write(example.SerializeToString())  #序列化为字符串
            cnt_num+=1
            
    writer.close()
    print ('for all: write to ',tfrecdir,'->',cnt_num,' groups done!!')
    return cnt_num

def preprocess_img(image,outlen):
    #这里将图片
    ''''''
    image=tf.image.resize_images(image, tuple(outlen))
    image=tf.cast(image, dtype=tf.float32)
    
    #image = tf.image.resize_image_with_crop_or_pad(image, 230, 230)
    #image = tf.random_crop(image, [outlen, outlen, 3])

    #image = tf.image.random_flip_left_right(image)
    return image

def read_tfrecord_batch(tfdir, imgsize, batchsize=12):
    '''
    imgsize:[new_height, new_width]
    '''
    tep=os.listdir(tfdir)
    random.shuffle(tep)
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
        
        image=tf.reshape(image,[height,width,-1])
        image=preprocess_img(image, imgsize)
        
        return image
    
    dataset=dataset.map(parse,num_parallel_calls=4)#注意把值回赋给dataset
    
    dataset=dataset.batch(batchsize).shuffle(batchsize*10)
    #print("dataset.output_shapes",dataset.output_shapes)
    
    iterator = dataset.make_one_shot_iterator()

    image_batch = iterator.get_next()

    return image_batch

def get_pipline_data_train(imgsize, batchsize):
    return read_tfrecord_batch(tfrec_dir_train, imgsize, batchsize)

def get_pipline_data_test(imgsize, batchsize):
    return read_tfrecord_batch(tfrec_dir_test, imgsize, batchsize)

def test_showtfimgs(tfdir, batchsize):
    tep=read_tfrecord_batch(tfdir, [360, 640], batchsize)
    with tf.Session() as sess:
        while True:
            images=sess.run(tep)
            cv2.imshow('test',images[0,:,:,:3].astype(np.uint8))
            cv2.waitKey(0)
            cv2.imshow('test',images[0,:,:,3:6].astype(np.uint8))
            cv2.waitKey(0)
            cv2.imshow('test',images[0,:,:,6:].astype(np.uint8))
            cv2.waitKey(0)
            #plt.show()

def gen_tfrecords():
    frames2tfrec(extratdir_train, tfrec_dir_train)
    frames2tfrec(extratdir_test, tfrec_dir_test)
    
    
if __name__ == '__main__':
    #txt2frames(train_txt ,extratdir_train)
    #txt2frames(test_txt, extratdir_test)
    #gen_tfrecords()
    test_showtfimgs(tfrec_dir_train, 5)
            
            
            
            
            
            
            
            