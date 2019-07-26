# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
import align.detect_face
import facenet
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
gpu_memory_fraction = 0.9
# facenet_model_checkpoint =  "/data/fengchen/facenet_1/models/20190603-101509"     # 130 epoch
# facenet_model_checkpoint =  "./models/facenet_lr_high_2"
# facenet_model_checkpoint =  "/data/fengchen/facenet_3/models/20190603-101511"     # 130 epoch, images_per_person=2
facenet_model_checkpoint =  "/data/fengchen/ensemble/model/20190220-111313"
# facenet_model_checkpoint = "../models/20180428-181544"   # 最初的模型
# facenet_model_checkpoint = "../models/20180519-210715"     # 接着在银星刘博发过来的模型
# facenet_model_checkpoint = "../models/20180518-220552"     # 自己训练的模型
# facenet_model_checkpoint = "../models/20180613-102209"     # 黄浦分局,刘博的模型
# facenet_model_checkpoint = "../models/20180624-172644"     # 黄浦分局,刘博6.24的模型

#facenet_model_checkpoint = "/home/gfs/minting/SITS/server/KC-facenet/model_checkpoints/20170512-110547"
#classifier_model = "/home/chenwen/PycharmProjects/KC-facenet/models/KNN.pkl"
#classifier_model = os.path.dirname(__file__) + "/../model_checkpoints/my_classifier_1.pkl"
debug = False


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.points = None
        self.direction = None

class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        faces = self.detect.find_faces(image)

        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face)
            face.name = self.identifier.identify(face)

        return faces


class Identifier:
    def __init__(self):
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def identify(self, face):
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            return self.class_names[best_class_indices[0]]


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input_ID:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings_ID:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=12):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []
        bounding_boxes, points = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        #numOfFace = bounding_boxes.shape[0]
        numOfPoint = points.shape
        #for bb in bounding_boxes:
        for i in range(len(bounding_boxes)):
            bb = bounding_boxes[i]
            pp = points[:,i]
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]

            #misc.imsave('margin12.jpg', cropped)
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            face.points = pp
            face.direction = 0



            v1 = [(pp[1] - pp[0]), (pp[6] - pp[5])] # left eye to right eye
            v2 = [(pp[2] - pp[0]), (pp[7] - pp[5])] # left
            v11 = [(pp[0] - pp[1]), (pp[5] - pp[6])]
            v22 = [(pp[2] - pp[1]), (pp[7] - pp[6])]

            left = cosine_similarity([v1], [v2])[0][0]
            right = cosine_similarity([v11], [v22])[0][0]

            faceFlag = 0

            if left - right > 0.15 and left > 0.70:
                faceFlag = 2
            elif left - right > 0.5:
                faceFlag = 2
            elif right - left > 0.5:
                faceFlag = 1
            elif right - left > 0.15 and right > 0.70:
                faceFlag = 1
            elif abs(left - right) < 0.10 and abs(left-right) > 0.05 and left > 0.70 and right > 0.70:
                faceFlag = 3
            elif abs(left - right) < 0.10 and abs(left - right) > 0.05 and left < 0.70 and right < 0.70:
                faceFlag = 4

            face.direction = faceFlag
            faces.append(face)

        return faces
