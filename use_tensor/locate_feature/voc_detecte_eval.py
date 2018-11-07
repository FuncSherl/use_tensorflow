#coding:utf-8
'''
Created on Thu Aug 30 10:43:27 2018

@author: lixiunan
'''

from __future__ import print_function
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle


def parse_voc_rec(filename):
    """
    parse pascal voc record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['name'] = obj.find('name').text
        obj_dict['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)]
        objects.append(obj_dict)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))  #拼接数组 加上头部0 和尾部 1
        mpre = np.concatenate(([0.], prec, [0.]))  #拼接数组 加上头部0 和尾部 0
       # print('mpre:',mpre)
       # print('mrec',mrec)
        # compute precision integration ladder 曲线值（也用了插值） 
        for i in range(mpre.size - 1, 0, -1):   # 倒序 10 9 8 7 .。。。1
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        #print('mpre:',mpre)
        #print('mrec',mrec)
        # look for recall value changes 找到变化的点或者说阶梯开始点索引
        i = np.where(mrec[1:] != mrec[:-1])[0]
        #print('i',i)
        # sum (\delta recall) * prec
        # 
        #print('tu',mpre[i+1])
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imageset_file, classname, cache_dir, ovthresh=0.5, use_07_metric=False):
    """
    pascal voc evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param classname: category name
    :param cache_dir: caching annotations
    :param ovthresh: overlap threshold
    :param use_07_metric: whether to use voc07's 11 point ap computation
    :return: rec, prec, ap
    """
  
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    cache_file = os.path.join(cache_dir, 'annotations.pkl')
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_filenames = [x.strip() for x in lines]
    #print(image_filenames)
    # load annotations from cache
    if not os.path.isfile(cache_file):
        recs = {}
        for ind, image_filename in enumerate(image_filenames):
            print('ind:',ind)
            print('image_filename',image_filename)
            recs[image_filename] = parse_voc_rec(annopath + '\\' + str(image_filename) +'.xml')
            if ind % 100 == 0:
                print('reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames)))
        print('saving annotations cache to {:s}'.format(cache_file))
        with open(cache_file, 'wb') as f:
            pickle.dump(recs, f)
    else:
        with open(cache_file, 'rb') as f:
            recs = pickle.load(f)

    # extract objects in :param classname:
    class_recs = {}
    npos = 0
    for image_filename in image_filenames:
        objects = [obj for obj in recs[image_filename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in objects])
        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        det = [False] * len(objects)  # stand for detected
        npos = npos + sum(~difficult)
        
        #GT保持的map
        class_recs[image_filename] = {'bbox': bbox,#为1个list，每个元素为4int的list
                                      'difficult': difficult,#每个框是否difficult
                                      'det': det}

    # read detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines[1:]] # 跳过第一行 表头
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_inds = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    bbox = bbox[sorted_inds, :]#这里bbox和imgid都是按照confidence由大到小拍好的
    image_ids = [image_ids[x] for x in sorted_inds]
    #print('image_ids',image_ids)
    # go down detections and mark true positives and false positives
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    fix_iou = False
    
#    #FP 计数 如果有图像本身没目标但是预测有目标 
#    count_FP = 0
#    
    #==============
    for d in range(nd):
        # 如果检测文件里面的图像未出现在 VOC检测文件里那么这个就是 误报，虚景
        if not image_ids[d] in class_recs.keys() :
            fp[d] = 1
            print('没有目标，但预测有目标',image_ids[d])
            continue
        
        
        r = class_recs[image_ids[d]]
        bb = bbox[d, :].astype(float)
        ovmax = -np.inf
        bbgt = r['bbox'].astype(float)

        if bbgt.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])#分别将每个bbgt中的与bb【0】做max操作，得到list
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            #print('inter:',inters)
            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +#为1个list，每个元素为4int的list
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)
            #
            #对于IOU一些修正，如果出现一个框完全包裹另一个框 这里仅设置了预测框大于真实框
            #
            #fix_iou = (bb[0] > bbgt[0] and bb[1] >bbgt[1] and bb[2] > bbgt[2] and bb[3] > bbgt[3]) or (bb[0] < bbgt[0] and bb[1] <bbgt[1] and bb[2] < bbgt[2] and bb[3]<bbgt[3])
           
           # print('bbgt',bbgt[:,2])
            fix_iou = bb[0]-bbgt[:,0] <0 and  bb[1]-bbgt[:,1] < 0 and  bb[2]-bbgt[:,2] > 0 and  bb[3]-bbgt[:,3] >0
           
            # fix_iou = False
#            if np.min(coord_sub ) > 0  or np.max(coord_sub) < 0 :
#                
#                fix_iou = True
#            else:
#                fix_iou = False
#            
            #
            #对于IOU一些修正，如果出现一个框完全包裹另一个框
            #
            
            overlaps = inters / uni
            #print('IOU',overlaps)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if (ovmax > ovthresh) or (fix_iou and ovmax > 0.3) :
            if not r['difficult'][jmax]:
                if not r['det'][jmax]:
                    tp[d] = 1.
                    r['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
    #print('tp:',tp)
    # compute precision recall
    fp = np.cumsum(fp) #计算累计fp
    
    # 如果测试样本没有目标却标注成了目标，那么这些都会算成  误报，也就是FP
   # print('FP:',fp)
        
    tp = np.cumsum(tp)
    #print('TP:',tp)
    rec = tp / float(npos)
    #print('rec:',rec)
    print('GT数量',npos)
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

detpath= r'F:\function_test\评估没有目标的样本\ship_det_val.txt'
annopath=r'F:\function_test\评估没有目标的样本\VOCdevkit\VOC2007\Annotations'
imageset_file=r'F:\function_test\评估没有目标的样本\VOCdevkit\VOC2007\ImageSets\Main\ship_val.txt'
classname='ship'
cache_dir = r'F:\function_test\评估没有目标的样本\temp'  # 每次修改数据集都要清空这个文件夹
rec, prec, ap = voc_eval(detpath, annopath, imageset_file, classname,cache_dir,ovthresh=0.5, use_07_metric=False)
print('AP',ap)
print('\n')
print('recall',rec[-1],'prec',prec[-1])