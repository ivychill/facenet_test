# coding=utf-8
import numpy as np
import os
# import pandas as pd
import warnings
from itertools import combinations
from getRep import getReptf
from functools import partial
from datetime import datetime
# import matplotlib.pyplot as plt
from sys import argv
import face
import xlwt
import mmd_represent
import only_align
import shutil

model_name = face.facenet_model_checkpoint.split('/')[-1]

warnings.filterwarnings('ignore')
cwd = os.getcwd()


def getDistance(x1, x2):
    # print 'get distance',x1.shape,type(x1)
    if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        d = x1.astype(np.float32)-x2.astype(np.float32)
    else:
        x1 = np.array(x1,dtype=np.float32)
        x2 = np.array(x2,dtype=np.float32)
        d = x1-x2
    return np.round(np.dot(d,d.T),3)

def baseImageRep_tf(data_dir):
    tmp_path='/opt/tmp/'
    only_align.align_img(data_dir,tmp_path)
    dataSet,labels = mmd_represent.get_rep(tmp_path,face.facenet_model_checkpoint)
    assert len(dataSet) == len(labels)  # 样本数和标签数要相等
    dataSet = np.array(dataSet)
    labels = np.array(labels).reshape(-1,1)
    baseData = np.hstack((labels,dataSet))
    shutil.rmtree(tmp_path)
    return baseData


def combinations_self(indata_ID,indata_camera):   # 输入numpy数组，两两组合各种可能,返回一个包含各种可能组合之间的欧式距离的list
    pairs_list_all=[]
    for ID in indata_ID:
        for camera in indata_camera:
            pairs_list_all.append((ID,camera))
    return  pairs_list_all


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

    tnr =  0 if (tn + fp == 0) else float(tn) / float(tn + fp)
    fnr =  0 if (tp + fn == 0) else float(fn) / float(tp + fn)
    return recall, precision, acc, tnr, fpr,fnr


if __name__ == '__main__':
    now = datetime.strftime(datetime.now(), '%Y-%m-%d-%H_%M_%S')
    thresholds = np.arange(0.01, 2, 0.01)
    P_valid_force = 100000
    N_valid_force = 1000000
    best_precison = 0
    best_accuracy = 0
    best_threshold_precison=None
    best_threshold_acc=None
    myRound = partial(round,ndigits=4)     #偏函数,2018.08.06，by xjxf
    test_file_ID =argv[1]
    test_file_camera=argv[2]

    savedir = os.path.join('./saveResults/',now+argv[3])
    if not os.path.exists(savedir):
        os.makedirs(os.path.abspath(savedir))

    f1 = open(os.path.join(savedir,'TestReport%s.txt'% now),'wb')
    f2 = open(os.path.join(savedir,'TestReport%s_backup.txt'% now),'wb')

    # 创建excel文件指针,2018.08.07,by xjxf  _start
    path_excel=os.path.join(savedir,'oush_distance%s_data.xls'% now)
    xls_3=xlwt.Workbook()
    sheet_1=xls_3.add_sheet('sheet_1',cell_overwrite_ok=True)
    row_count=0
    # 创建excel文件指针,2018.08.07,by xjxf  _end
    f1.write('-----测试日期:{}, 测试模型：{}, 测试数据:{},距离指标:欧式距离-----\n'.format(now,model_name,test_file_ID))
    f2.write('-----测试日期:{}, 测试模型：{}, 测试数据:{},距离指标:欧式距离-----\n'.format(now,model_name,test_file_ID))
    print '-----测试日期:{}, 测试模型：{}, 测试数据:{},,距离指标:欧式距离------\n'.format(now,model_name,test_file_ID)

    baseData_ID = baseImageRep_tf(test_file_ID)  # 提取基础集的特征点和标签
    baseData_camera = baseImageRep_tf(test_file_camera)  # 提取基础集的特征点和标签

    # pairs_list_all = list(combinations(baseData, 2))

    pairs_list_all=combinations_self(baseData_ID,baseData_camera)
    np.random.shuffle(pairs_list_all)
    P_num = 0
    N_num = 0
    pair_list =[]
    for pair in pairs_list_all:
        if pair[0][0]==pair[1][0] and P_num<P_valid_force:
            pair_list.append(pair)
            P_num+=1
        elif pair[0][0]!=pair[1][0] and N_num<N_valid_force:
            pair_list.append(pair)
            N_num+=1
        elif P_num==P_valid_force and N_num== N_valid_force:
            break
    s_ = '**正测试单元数量:{},  负测试单元数量:{}**\n'.format(P_num,N_num)
    print s_
    f1.write(s_)
    f2.write(s_)

    distList = []
    actual_issameList = []
    for pair in pair_list:
        actual_issame = True
        if pair[0][0] != pair[1][0]:
            actual_issame = False
        dist = getDistance(pair[0][1:], pair[1][1:])
        distList.append(dist)
        actual_issameList.append(actual_issame)


    # fig = plt.figure(figsize=(30,50))
    # dist_P = np.array(distList)[actual_issameList].tolist()
    # dist_N = np.array(distList)[np.logical_not(actual_issameList)].tolist()
    # plt.scatter(np.arange(len(dist_P)),dist_P,s=1000,marker='*',label='Positive')
    # plt.title(now,fontsize=70)
    # plt.scatter(np.arange(len(dist_N)), dist_N, s=1000, marker='.',label='Negative')
    # plt.yticks(fontsize=50)
    # plt.legend(loc='upper right',fontsize=50)
    # plt.savefig(os.path.join(savedir,'%s.png'% now))
    # plt.show()
    f2.write('threshold,recall, precision, acc, tnr, fpr,fnr\n')
    for threshold in thresholds:
        print '---------threshold:',round(threshold,2),'---------------\n'
        f1.write('---------threshold:'+str(round(threshold,2))+'---------------\n')

        recall, precision, acc, tnr, fpr,fnr = map(myRound,calculate_accuracy(threshold, distList, actual_issameList))

        if precision>best_precison:
            best_precison = precision
            best_threshold_precison = threshold

        if acc>best_accuracy:
            best_accuracy=acc
            best_threshold_acc = threshold

        ss = 'recall召回率:'+str(recall)+'\t'+'precison精准率:'+str(precision)+'\t'+' accuracy正确率:'+str(acc)+'\n'
        sss = 'TNR真负率:'+str(tnr)+'\t'+'FPR假正率:'+str(fpr)+'\t'+'FNR假负率:'+str(fnr)+'\t\n\n'
        print ss,sss
        f1.write(ss)
        f1.write(sss)
        f2.write('%f,%f,%f,%f,%f,%f,%f\n'%(threshold,recall, precision, acc, tnr, fpr,fnr))
        #写excel文件,2018.08.07, by xjxf  _start
        list_data=[threshold,tnr, fpr,fnr,recall, precision, acc]
        for i_1 in range(len(list_data)):
            sheet_1.write(row_count,i_1,list_data[i_1])
        row_count=row_count+1
    xls_3.save(path_excel)

        # 写excel文件,2018.08.07, by xjxf  _end

    foo =  '\n\n*********conclusion********\n'+'best_precison:{}   best_threshold_precision:{}\nbest_accuracy:{}   best_threshold_acc:{} '.format(best_precison, best_threshold_precison,best_accuracy,best_threshold_acc)
    print foo
    f1.write(foo)
    
    print("model_name",model_name)
    print("data path:",test_file_ID)
    print("excel path:",path_excel)



