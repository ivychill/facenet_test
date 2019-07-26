# coding=utf-8
import numpy as np
import os
from getRep import getReptf
from datetime import datetime
import warnings
import face

model_name = face.facenet_model_checkpoint.split('/')[-1]
warnings.filterwarnings('ignore')
cwd = os.getcwd()

def baseImageRep_tf(imagesFileName):
    imagesFileName = imagesFileName
    personFileNameDir = os.path.join(cwd, imagesFileName)
    personFileName = os.listdir(personFileNameDir)

    labels = []
    dataSet = []
    count=0
    for person in personFileName:
        personDir = os.path.join(personFileNameDir, person)
        imagesList = os.listdir(personDir)
        image_num = 0
        for image in imagesList:
            try:
                imagePath = os.path.join(personDir, image)  # 目标图片的绝对路径
                rep, _ = getReptf(imagePath)
                # rep = rep.tolist()
                dataSet.append(rep)
                label = person  # 每个person的文件夹名称
                labels.append(label)
                image_num+=1
                print image_num
                if image_num==6:  # 限制识别库中每个人的特征值数量
                    break
                print 'done:',image

            except Exception as e:
                count+=1
                print '没提取到:   %s' % image

    assert len(dataSet) == len(labels)  # 样本数和标签数要相等
    dataSet = np.array(dataSet)
    labels = np.array(labels).reshape(-1,1)
    baseData = np.hstack((labels,dataSet))
    print '没有提取成功的图片数量：', count
    return baseData

def kNNClassify(inX, dataSet, labels, k=3):
    '''
    :param inX:  测试的样本128位特征值
    :param dataSet: 带标签人脸库数据，128列的numpy数组
    :param labels: 人脸库中的标签，numpy数组，List均可
    :param k: 参数K
    :return: KNN算法预识的label
    '''
    disTance = map(sum, np.power(dataSet.astype(np.float32)-inX.astype(np.float32), 2))
    data = np.vstack((disTance, labels))
    dataT = data.T
    dataT = dataT.tolist()

    for i in range(len(dataT)):
        dataT[i][0] = float(dataT[i][0])
    dataT.sort()

    count = dict()
    for i in range(k):
        if dataT[i][1] not in count:
            count[dataT[i][1]] = 1
        else:
            count[dataT[i][1]] +=1
    # 对字典项排序
    # res = sorted(count.items(),key=lambda x:x[1],reverse=True)  # 方法1
    # res=sorted(count.items(), key=operator.itemgetter(1),reverse=True)  # 方法2
    # print count.values(),count.keys()
    res =zip(count.values(), count.keys())
    res = sorted(res, reverse=True)
    label = res[0][1]
    nearest_dist = dataT[:metric_k]
    return label,nearest_dist

if __name__ == '__main__':
    basedatafile = '/data/liukang/face_data/ID_ID_226/id_dir/'
    testdatafile = '/data/liukang/face_data/ID_ID_226/cam_dir/'
    metric_k_list = [1,2,3]

    basedata = baseImageRep_tf(basedatafile)
    # # np.save('basedata',basedata)
    # basedata = np.load('basedata.npy')
    testdata = baseImageRep_tf(testdatafile)
    # np.save('testdata',testdata)
    # testdata = np.load('testdata.npy')

    base_num = basedata.shape[0]
    file_name = 'LK_test'

    now = datetime.strftime(datetime.now(), '%Y-%m-%d-%H_%M_%S')
    savedir = os.path.join('./saveResults/',file_name)
    # file_name = '0613_diku_dayu2_knn'
    if not os.path.exists(savedir):
        os.makedirs(os.path.abspath(savedir))

    f1 = open(os.path.join(savedir,'%s.txt'% file_name),'wb')
    for metric_k in metric_k_list:
        f1.write('-----测试日期:{}, 测试模型：{}, 人脸库:{},人脸库中特征值数量:{},metric_K:{}-----\n\n'.format(now, model_name,basedatafile,base_num,metric_k))

        total_result = []
        for testface in testdata:
            in_label = testface[0]
            res,nearest_dist = kNNClassify(testface[1:],dataSet=basedata[:,1:],labels=basedata[:,0],k=metric_k)
            result=False
            if res==in_label:
                result = True
            total_result.append(result)
            writeline = '输入:%s,\t KNN预测%s,\t结论:%s\n'%(in_label,res,result)
            f1.write(writeline)
            _ = '最近的K个距离分布:%s\n\n'%nearest_dist
            # f1.write(_)
        # print total_result.count(True)
        # print len(total_result)
        f_result = (float(total_result.count(True))/float(len(total_result)))

        accuracy = '\n统计测试准确率:%.5f\n\n'% f_result
        f1.write(accuracy)
    f1.close()