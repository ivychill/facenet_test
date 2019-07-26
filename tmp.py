# coding=utf-8
import numpy as np
import os
# import pandas as pd
import warnings
from itertools import combinations
# from getRep import getReptf
from functools import partial
from datetime import datetime
# import matplotlib.pyplot as plt
from sys import argv
# import face
import xlwt
from sklearn import metrics

# model_name = face.facenet_model_checkpoint.split('/')[-1]
model_name = "tmp"



if __name__ == '__main__':
    now = datetime.strftime(datetime.now(), '%Y-%m-%d-%H_%M_%S')
    thresholds = np.arange(0.01, 1.8, 0.01)
    P_valid_force = 45000        # sample number need to change
    N_valid_force = 1000000       #100w
    best_precison = 0
    best_accuracy = 0
    best_threshold_precison=None
    best_threshold_acc=None
    myRound = partial(round,ndigits=4)     #偏函数,2018.08.06，by xjxf
    test_file =argv[1]

    savedir = os.path.join('./saveResults/',now)
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
    f1.write('-----测试日期:{}, 测试模型：{}, 测试数据:{},距离指标:欧式距离-----\n'.format(now,model_name,test_file))
    f2.write('-----测试日期:{}, 测试模型：{}, 测试数据:{},距离指标:欧式距离-----\n'.format(now,model_name,test_file))
    print '-----测试日期:{}, 测试模型：{}, 测试数据:{},,距离指标:欧式距离------\n'.format(now,model_name,test_file)

    baseData = baseImageRep_tf(test_file)  # 提取基础集的特征点和标签

    pairs_list_all = list(combinations(baseData, 2))
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

    print("P_valid_force is  " + str(P_valid_force))
    print("P_num in code is  " + str(P_num))
    print("N_valid_force is  " + str(N_valid_force))
    print("N_num in code is  " + str(N_num))


    # # fig = plt.figure(figsize=(30,50))
    # # dist_P = np.array(distList)[actual_issameList].tolist()
    # # dist_N = np.array(distList)[np.logical_not(actual_issameList)].tolist()
    # # plt.scatter(np.arange(len(dist_P)),dist_P,s=1000,marker='*',label='Positive')
    # # plt.title(now,fontsize=70)
    # # plt.scatter(np.arange(len(dist_N)), dist_N, s=1000, marker='.',label='Negative')
    # # plt.yticks(fontsize=50)
    # # plt.legend(loc='upper right',fontsize=50)
    # # plt.savefig(os.path.join(savedir,'%s.png'% now))
    # # plt.show()
    # f2.write('threshold,recall, precision, acc, tnr, fpr,fnr\n')
    # auc={"fpr_acu":[],"tpr_auc":[]}
    # for threshold in thresholds:
    #     print '---------threshold:',round(threshold,2),'---------------\n'
    #     f1.write('---------threshold:'+str(round(threshold,2))+'---------------\n')
    #
    #     recall, precision, acc, tnr, fpr,fnr = map(myRound,calculate_accuracy(threshold, distList, actual_issameList))
    #     auc['fpr_acu'].append(fpr)
    #     auc['tpr_auc'].append(recall)
    #     if precision>best_precison:
    #         best_precison = precision
    #         best_threshold_precison = threshold
    #
    #     if acc>best_accuracy:
    #         best_accuracy=acc
    #         best_threshold_acc = threshold
    #
    #     ss = 'recall召回率:'+str(recall)+'\t'+'precison精准率:'+str(precision)+'\t'+' accuracy正确率:'+str(acc)+'\n'
    #     sss = 'TNR真负率:'+str(tnr)+'\t'+'FPR假正率:'+str(fpr)+'\t'+'FNR假负率:'+str(fnr)+'\t\n\n'
    #     print ss,sss
    #     f1.write(ss)
    #     f1.write(sss)
    #     f2.write('%f,%f,%f,%f,%f,%f,%f\n'%(threshold,recall, precision, acc, tnr, fpr,fnr))
    #     #写excel文件,2018.08.07, by xjxf  _start
    #     list_data=[threshold,tnr, fpr,fnr,recall, precision, acc]
    #     for i_1 in range(len(list_data)):
    #         sheet_1.write(row_count,i_1,list_data[i_1])
    #     row_count=row_count+1
    # xls_3.save(path_excel)
    #
    #     # 写excel文件,2018.08.07, by xjxf  _end
    #
    # foo =  '\n\n*********conclusion********\n'+'best_precison:{}   best_threshold_precision:{}\nbest_accuracy:{}   best_threshold_acc:{} '.format(best_precison, best_threshold_precison,best_accuracy,best_threshold_acc)
    # print foo
    # f1.write(foo)
    #
    # print("model_name",model_name)
    # print("data path:",test_file)
    # print("excel path:",path_excel)
    # auc_result = metrics.auc(auc['fpr_acu'], auc['tpr_auc'])
    # print('Area Under Curve (AUC): %1.3f' % auc_result)