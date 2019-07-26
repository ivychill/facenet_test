#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Allows you to generate embeddings from a directory of images in the format:

Instructions:

Image data directory should look like the following figure:
person-1
├── image-1.jpg
├── image-2.png
...
└── image-p.png

...

person-m
├── image-1.png
├── image-2.jpg
...
└── image-q.png

Trained Model:
- Both the trained model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.

####
USAGE:
$ python batch_represent.py -d <YOUR IMAGE DATA DIRECTORY> -o <DIRECTORY TO STORE OUTPUT ARRAYS> --trained_model_dir <DIRECTORY CONTAINING PRETRAINED MODEL>
###
"""

"""
Attributions:
The code is heavily inspired by the code from by David Sandberg's ../src/validate_on_lfw.py
The concept is inspired by Brandon Amos' github.com/cmusatyalab/openface/blob/master/batch-represent/batch-represent.lua
"""

# ----------------------------------------------------
# MIT License
#
# Copyright (c) 2017 Rakshak Talwar
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
# ----------------------------------------------------

import os
import sys
import importlib
import time
import facenet
import numpy as np
from sklearn.datasets import load_files
import tensorflow as tf
from six.moves import xrange


def get_rep(data_dir,trained_model_dir,batch_size=50):
    with tf.Graph().as_default():

        with tf.Session() as sess:
            # load the model
            print("Loading trained model...\n")
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(trained_model_dir))
            facenet.load_model(trained_model_dir)

            # grab all image paths and labels
            print("Finding image paths and targets...\n")
            data = load_files(data_dir, load_content=False, shuffle=False)
            #print(data.keys())
            labels_array = data['target']
            paths = data['filenames']

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input_ID:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings_ID:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
            print("image_size is :", image_size)
            print("embedding size is :", embedding_size)
            # Run forward pass to calculate embeddings
            print('Generating embeddings from images...\n')
            start_time = time.time()
            nrof_images = len(paths)
            nrof_batches = int(np.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in xrange(nrof_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, do_random_crop=False, do_random_flip=False,
                                           image_size=image_size, do_prewhiten=True)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            time_avg_forward_pass = (time.time() - start_time) / float(nrof_images)
            print("Forward pass took avg of %.3f[seconds/image] for %d images\n" % (time_avg_forward_pass, nrof_images))
            labels_name_array = []
            for i in range(len(labels_array)):
                labels_name_array += [data['target_names'][labels_array[i]]]
    return emb_array,labels_name_array
