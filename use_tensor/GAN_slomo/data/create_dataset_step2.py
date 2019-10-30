#coding:utf-8
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

if pc_id==0: #
    videodir=r'E:\DL_datasets\DeepVideoDeblurring_Dataset_Original_High_FPS_Videos\original_high_fps_videos'  
    extratdir=r'E:\DL_datasets\DeepVideoDeblurring_Dataset_Original_High_FPS_Videos\extracted_videos' 
    tfrec_dir=r'E:\DL_datasets\DeepVideoDeblurring_Dataset_Original_High_FPS_Videos\tfrecords_step2'
    modelpath=r"E:\DL_models\use_tensorflow\v24\GAN_2019-08-20_21-49-57_base_lr-0.000200_batchsize-12_maxstep-240000_fix_a_bug_BigProgress"
    
elif pc_id==1: #lab pc
    videodir=r'/media/sherl/本地磁盘/data_DL/Adobe240fps/original_high_fps_videos' #
    extratdir=r'/media/sherl/本地磁盘/data_DL/Adobe240fps/extracted_videos' #
    tfrec_dir=r'/media/sherl/本地磁盘/data_DL/Adobe240fps/tfrecords_step2' #
    modelpath = "/home/sherl/Pictures/v24/GAN_2019-08-20_21-49-57_base_lr-0.000200_batchsize-12_maxstep-240000_fix_a_bug_BigProgress"
    modelpath="/home/sherl/Pictures/v25/GAN_2019-10-27_16-11-30_base_lr-0.000200_batchsize-12_maxstep-240000_rid_GAN_ssim_down"
    modelpath="/home/sherl/Pictures/v25/GAN_2019-10-29_17-35-34_base_lr-0.000200_batchsize-6_maxstep-240000"
    
elif pc_id==2: #server
    videodir=r'/home/vrlab/Documents/xvhao/use_tensorflow/use_tensor/GAN_slomo/data/original_high_fps_videos'
    extratdir=r'/home/vrlab/Documents/xvhao/use_tensorflow/use_tensor/GAN_slomo/data/extracted_videos'
    tfrec_dir=r'/home/vrlab/Documents/xvhao/use_tensorflow/use_tensor/GAN_slomo/data/tfrecords_step2'
    modelpath = "/home/vrlab/Documents/xvhao/use_tensorflow/use_tensor/GAN_slomo/logs_v24/GAN_2019-08-20_21-49-57_base_lr-0.000200_batchsize-12_maxstep-240000"
    modelpath = "/home/vrlab/Documents/xvhao/use_tensorflow/use_tensor/GAN_slomo/logs_v25/GAN_2019-10-29_17-35-34_base_lr-0.000200_batchsize-6_maxstep-240000"

extratdir_train=op.join(extratdir, 'train')
extratdir_test=op.join(extratdir, 'test')

tfrec_dir_train=op.join(tfrec_dir, 'train')
tfrec_dir_test=op.join(tfrec_dir, 'test')

group_cnt_images=12   #每个分组有多少图
mean_dataset=[102.1, 109.9, 110.0]  #0->1 is [0.4, 0.43, 0.43]
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
        
def frames2tfrec(frame_dir, tfrecdir, group_num=group_cnt_images, groups_perfile=300, img_shape_required=[640, 360]):#使用cv2.resize时，参数输入是 宽×高 ，与以往操作不同
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
            label_tep=str(dirind)+"_"+str(j)  #构建label
            print (dirind,'/',len(videodirs),'  ',j,'/',len(framelist), cnt_num,label_tep)
            
            rimg=cv2.imread(op.join(tepath, framelist[j]))
            rimg=cv2.resize(rimg, tuple(img_shape_required))
            imgdata.append(rimg)
            
            if len(imgdata)<group_num: continue
                
            #print (imgdata.dtype, type(imgdata))
            
            #从通道维度将一组12个图片连起来，变为(720, 1280, 3*12)
            groupdata=np.concatenate(imgdata,axis=2)
            size=groupdata.shape  #(720, 1280, 3*12)
            #print (size)
            groupdata_raw=groupdata.tobytes()#将图片转化为二进制格式
            
            print (size)
            if cnt_num%groups_perfile==0:
                ftrecordfilename = (op.split(tfrecdir)[-1].strip()+".tfrecords_%.4d" % int(cnt_num/groups_perfile))
                writer= tf.python_io.TFRecordWriter(op.join(tfrecdir,ftrecordfilename))
            
            example = tf.train.Example(
            features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_tep.encode("utf-8")])),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[groupdata_raw])),
            'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]])),
            'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]]))
            })) 
            writer.write(example.SerializeToString())  #序列化为字符串
            cnt_num+=1
            imgdata=[imgdata[-1]]
            
            
    writer.close()
    print ('for all: write to ',tfrecdir,'->',cnt_num,' groups done!!')
    return cnt_num

def preprocess_img(image,outlen, outchannel=9, training=True):
    #这里将图片
    ''''''
    #这里利用resize后的图片训练可能会导致在真实情况下效果由于与训练时的图片缩放比例不一样导致的问题
    #tf.image.resize_image_with_crop_or_pad()
    if training:
        image=tf.image.resize_images(image, tuple(  [outlen[0]+6, outlen[1]+6 ]   )  )
    else:
        image=tf.image.resize_images(image, tuple(  [outlen[0], outlen[1] ]   )  )
    #image=tf.image.random_flip_left_right(image)
    #image=tf.image.random_crop(image, [outlen[0], outlen[1], outchannel ])
    
    #image=tf.image.resize_images(image, tuple( outlen   )  )
    
    image=tf.cast(image, dtype=tf.float32)
    #print (image)  #Tensor("random_crop:0", shape=(180, 320, 9), dtype=float32)
    #image-=mean_dataset*int(outchannel/3)   #这里 不知道brocatcast是否好用
    
    return image

def read_tfrecord_batch(tfdir, imgsize, batchsize=12, img_channel=3, training=True):
    '''
    imgsize:[new_height, new_width]
    '''
    tep=os.listdir(tfdir)
    tep.sort()
    #random.shuffle(tep)
    tep=list(map(lambda x:op.join(tfdir, x), tep))
    print (tep)
    dataset = tf.data.TFRecordDataset(tep).repeat()
    
    
    def parse(one_element):
        feats = tf.parse_single_example(one_element, features={'data':tf.FixedLenFeature([], tf.string), 
                                                           'label':tf.FixedLenFeature([],tf.string), 
                                                           'width':tf.FixedLenFeature([], tf.int64),
                                                           'height':tf.FixedLenFeature([], tf.int64)})
        image = tf.decode_raw(feats['data'], tf.uint8)
        label = tf.cast(feats['label'],tf.string)   #feats['label'].decode()
        width = tf.cast(feats['width'], tf.int32)
        height= tf.cast(feats['height'], tf.int32)
        
        image=tf.reshape(image,[height,width,-1])
        
        # 这里需要随机取3帧，并且要求对称
        num_ori_group=group_cnt_images #tf.cast(image.get_shape().as_list()[-1]/img_channel, tf.int32)
        
        #生成的值遵循范围内的均匀分布 [minval, maxval)。下限minval包含在范围内，而上限maxval则被排除在外。
        #对于浮点数，默认范围是[0, 1)。对于int，至少maxval必须明确指定。
        #randnum_start=tf.random_uniform([1],minval=0, maxval=int(num_ori_group/3), dtype=tf.int64)[0]
        randnum_mid=tf.random_uniform([1],minval=int(1), maxval=int(num_ori_group-1), dtype=tf.int64)[0]
        #randnum_r=  tf.random_uniform([1],minval=2*int(num_ori_group/3), maxval=num_ori_group  , dtype=tf.int64)[0]
        
        frame0=image[:,:, :img_channel]
        frame1=image[:,:, randnum_mid*img_channel:(randnum_mid+1)*img_channel]
        frame2=image[:,:, (-1)*img_channel: ]
        tepframes=[frame0, frame1, frame2]
        image=tf.concat(tepframes, -1)
        
        ###############################################################################################
        image=preprocess_img(image, imgsize, len(tepframes)*img_channel, training)
        
        #这里t_rate指的是中间帧距离前帧的距离/后一帧与前一帧的距离，位于(0,1)
        t_rate= tf.cast(randnum_mid, tf.float32) / tf.cast(num_ori_group-1, tf.float32)
        return [image,t_rate, label]
    
    dataset=dataset.map(parse,num_parallel_calls=6)#注意把值回赋给dataset
    
    dataset=dataset.batch(batchsize)#.shuffle(batchsize*6)
    #print("dataset.output_shapes",dataset.output_shapes)
    
    iterator = dataset.make_one_shot_iterator()

    image_batch = iterator.get_next()
    #print(image_batch)
    return image_batch

def get_pipline_data_train(imgsize, batchsize):
    return read_tfrecord_batch(tfrec_dir_train, imgsize, batchsize, training=True)

def get_pipline_data_test(imgsize, batchsize):
    return read_tfrecord_batch(tfrec_dir_test, imgsize, batchsize, training=False)

def test_showtfimgs(tfdir, batchsize):
    tep=read_tfrecord_batch(tfdir, [360, 640], batchsize)
    with tf.Session() as sess:
        while True:
            images=sess.run(tep) 
            print ( images[0].shape)  #images这里是一个tuple，第一个是图片，第二个是timerate    (2, 360, 640, 9)
            images,rates,label=images[0],images[1],images[-1]
            print (rates)
            print (label)
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
    
def get_dataset_mean():
    '''
    该函数读取分解成帧的video的文件夹，所以要先生成tfrecord之后再计算mean，且存放中间分成帧的目录不要删除
    '''
    R=0.0
    G=0.0
    B=0.0
    cnt_frame=0
    
    for indi,i in enumerate(os.listdir(extratdir)):
        teplen=len(os.listdir(extratdir))
        teppath=op.join(extratdir, i)
        for indj,j in enumerate(os.listdir(teppath)):
            videolen=len(os.listdir(teppath))
            videopath=op.join(teppath, j)
            for indk,k in enumerate(os.listdir(videopath)):
                framelen=len(os.listdir(videopath))
                framepath=op.join(videopath, k)
                
                print ('\n',indi,"/",teplen," ",indj,"/",videolen," ",indk,"/",framelen,"->",framepath)
                
                img=cv2.imread(framepath)  #notice:cv2.imread is BGR mode
                h=img.shape[0]
                w=img.shape[1]
                #print (img.shape, img.dtype)  #(720, 1280, 3)
                
                B+=np.sum(img[:,:,0])*1.0/(h*w)
                G+=np.sum(img[:,:,1])*1.0/(h*w)
                R+=np.sum(img[:,:,2])*1.0/(h*w)
                cnt_frame+=1
                print ([B,G,R], cnt_frame)
    mean=[B/cnt_frame, G/cnt_frame, R/cnt_frame]
    print ("finally the mean is:", mean)  #[102.102762984439, 109.85845376864695, 110.03368690364164]
    return mean
            
                
        
    
    
    
if __name__ == '__main__':
    #txt2frames(train_txt ,extratdir_train)
    #txt2frames(test_txt, extratdir_test)
    gen_tfrecords()
    #test_showtfimgs(tfrec_dir_train, 2)
    #get_dataset_mean()
            
            
            
            
            
            
            
            