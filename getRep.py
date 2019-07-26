# coding=utf-8
# from face_recognition import get_embedding
from face import Encoder, Detection
# face_recognition = face.Recognition()
import tensorlayer as tl
# from align_extract_feature import  aligned_extract_feature

# # imagePath = '/home/chenwen/PycharmProjects/KC-facenet/yaoyu/multiface.jpg'
# imagePath = 'test.jpg'
# encoder = Encoder()
# detection = Detection()
#
# im= tl.vis.read_image(imagePath)
#
# findFace = detection.find_faces(im) # [<face.Face instance at 0x7f5330a30fc8>, <face.Face instance at 0x7f532f326050>, <face.Face instance at 0x7f532f326ef0>]  列表裏面爲Face類的實例
# print 'findFace----', findFace, type(findFace)
#
# # **********************************************************
# for face in findFace:   # 遍歷findFace的所有人臉
#     rep = encoder.generate_embedding(face) # 返回一個（128,）的numpy數組
#     faceimage = face.image  # Face類裏面的image屬性。人臉的numpy數組(160, 160, 3)
#     print type(faceimage), faceimage.shape
# # *************************************************************
encoder = Encoder()
detection = Detection()
def getReptf(imagePath):
    im = tl.vis.read_image(imagePath)   # 打开图片
    findFace = detection.find_faces(im)  # 找到定位并保存图片中所有的人脸，生成Face类的实例，作为一种数据结构存储
    replist = []
    flaglist=[]
    for face in findFace:             #在一张图像上可能找到多张脸  2018.08.06，by xjxf

        # print face.image

        rep = encoder.generate_embedding(face)  # 从face.image中提取人臉的矩陣，生成128位特征值numpy數組
        replist.append(rep)
        flaglist.append(face.direction)         # 正脸：0、3、4，左侧：1，右侧：2
    if len(replist)==0 or len(flaglist)==0:
        print 'getReptf:'
    else:
        return replist[0], flaglist[0]          # 返回列表中的一张人脸和返回人脸的flag

# # ***********************************************************************
# im = tl.vis.read_image(imagePath)   # 打开图片
# print im[:,:,0].shape
#
# import cv2
# # imCV = cv2.imread(imagePath)
#
# imCV = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# print imCV[:,:,0].shape
#
# print im[:,:,0]==imCV[:,:,0]
# 结论:opencv打开的格式为(250,250,3)的numpy数组BGR.   TF打开的是(250.250.3)的numpy数组,格式为RGB



if __name__ == '__main__':
    import os
    import numpy as np
    cwd = os.getcwd()
    imagesFileName = 'somepictures'
    personFileNameDir = os.path.join(cwd, imagesFileName)  # 某个人所有图片存放的文件夹路径
    personFileName = os.listdir(personFileNameDir)

    labels = []
    dataSet = []
    faceflags = []
    count = 0
    countall = 0
    for person in personFileName:
        personDir = os.path.join(personFileNameDir, person)
        imagesList = os.listdir(personDir)
        for image in imagesList:
            countall += 1
            try:
                imagePath = os.path.join(personDir, image)  # 目标图片的绝对路径
                rep, faceflag = aligned_extract_feature(imagePath)  # 正脸：0、3、4，左侧：1，右侧：2
                print rep.shape
                dataSet.append(rep)
                faceflags.append(faceflag)
                label = person  # 每个person的文件夹名称
                labels.append(label)

            except Exception as e:
                print '没提取到:   %s' % image, e
                count += 1
                # os.remove(imagePath)
    print '总共：', countall
    print '没有提取成功：', count
    dataSet = np.array(dataSet)  # 转化为numpy数组
    labels = np.array(labels)
    faceflags = np.array(faceflags)

