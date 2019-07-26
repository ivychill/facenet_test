# coding=utf-8
import numpy as np
import os
# import pandas as pd
import warnings
from itertools import combinations
from functools import partial
from datetime import datetime
# import matplotlib.pyplot as plt
from sys import argv
from getRep import getReptf
import face
model_name = face.facenet_model_checkpoint.split('/')[-1]
warnings.filterwarnings('ignore')
cwd = os.getcwd()


def getDistance(x1, x2):
    if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        d = x1.astype(np.float32)-x2.astype(np.float32)
    else:
        x1 = np.array(x1,dtype=np.float32)
        x2 = np.array(x2,dtype=np.float32)
        d = x1-x2
    return np.round(np.dot(d,d.T),3)   #d和d的转置相乘，2018.08.06,by xjxf

def baseImageRep_tf(imagesFileName):
    imagesFileName = imagesFileName
    personFileNameDir = os.path.join(cwd, imagesFileName)
    personFileName = os.listdir(personFileNameDir)

    labels = []
    dataSet = []
    imagename = []
    count=0
    for person in personFileName:
        personDir = os.path.join(personFileNameDir, person)
        imagesList = os.listdir(personDir)
        for image in imagesList:
            try:
                imagePath = os.path.join(personDir, image)  # 目标图片的绝对路径
                rep, _ = getReptf(imagePath)
                dataSet.append(rep)
                label = person  # 每个person的文件夹名称
                labels.append(label)
                imagename.append(image)

            except Exception as e:
                count+=1
                print '没提取到:   %s' % image

    assert len(dataSet) == len(labels) ,"raise a error, length of label isn't equal to length of dataset"   # 样本数和标签数要相等
    dataSet = np.array(dataSet)
    labels = np.array(labels).reshape(-1,1)
    imagename = np.array(imagename).reshape(-1,1)
    baseData = np.hstack((labels,dataSet))
    baseData = np.hstack((baseData,imagename))

    print '没有提取成功的图片数量：', count
    return baseData

def combineCompare(indata):   # 输入numpy数组，两两组合各种可能,返回一个包含各种可能组合之间的欧式距离的list
    result = []
    # print indata.shape
    if indata.shape[0] >1:
        res = combinations(indata,2)
        for pair in res:
            x1, x2 = pair
            distance = getDistance(x1,x2)
            print 'distance:',distance
            result.append(distance)
    return result

def zipCompare(data1,data2):
    result = []
    pairlist = []
    if data1.shape>0 and data2.shape>0:
        for index in data1:
            pairlist.extend(zip([index]*data2.shape[0],data2))
        for pair in pairlist:
            x1,x2 = pair
            dist = getDistance(x1, x2)
            result.append(dist)
            print 'dist:',dist
    return result


def calculate_accuracy(threshold, distList, actual_issameList):
    predict_issame = np.less(distList, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issameList))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issameList)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issameList)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issameList))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    recall = tpr
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / len(distList)
    precision =  0 if (tp + fp == 0) else float(tp) / float(tp + fp)
    return recall, precision, acc


if __name__ == '__main__':
    now = datetime.strftime(datetime.now(), '%Y-%m-%d-%H_%M_%S')
    # thresholds = np.arange(0.3, 1.8, 0.01)
    threshold = 0.88
    # P_valid_force = 6000
    # N_valid_force = 6000
    best_precison = 0
    best_accuracy = 0
    best_threshold_precison=None
    best_threshold_acc=None
    myRound = partial(round,ndigits=4)
    host_file = 'ptz'
    guest_file = 'zjz'


    savedir = os.path.join('./saveResults/',now)
    if not os.path.exists(savedir):
        os.makedirs(os.path.abspath(savedir))

    save_data = False
    npy_file = '2018-08-01-10_20_59_ptz.npy'
    with open(os.path.join(savedir,'TestReport%s.txt'% now),'wb',2) as f:
        f.write('-----测试日期:{}, 测试模型：{}, 库:{},  测试集:{}-----\n'.format(now,model_name,host_file,guest_file))
        print '-----测试日期:{}, 测试模型：{}, 库:{}, 测试集:{}-----\n'.format(now,model_name,host_file,guest_file)

        if save_data:
            baseData_host = baseImageRep_tf(host_file)  # 提取基础集的特征点和标签
            np.save(now+'_'+host_file, baseData_host)
        else:
            baseData_host = np.load(npy_file)
        baseData_guest = baseImageRep_tf(guest_file)  # 提取基础集的特征点和标签
        for guest in baseData_guest:
            guest_uid = guest[0]
            guest_name = guest[-1]
            ss_ = '---------输入:%s----%s--------\n' % (guest_uid,guest_name)
            print ss_
            f.write(ss_)
            got_uid = []
            for host in baseData_host:
                host_uid = host[0]
                host_name = host[-1]
                dist = getDistance(guest[1:-1],host[1:-1])
                if dist<threshold:
                    got_uid.append(host_uid)
                    str_ = '距离小于threshhold={}的uid为:{}，图片：{}，欧式距离:{}\n'.format(threshold,host_uid,host_name,dist)
                    print str_
                    f.writelines(str_)
            sss_ = '****识别的uid个数:%s*****\n' % len(set(got_uid))
            print sss_
            f.write(sss_)








