# -*- coding: utf-8 -*-

import tensorflow as tf
import utils
import numpy as np
import skimage.io

with open("vgg16.tfmodel", mode='rb') as f:
    fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
images = tf.placeholder("float", [None, 16, 16, 3])
tf.import_graph_def(graph_def, input_map={ "images": images })
print "graph loaded from disk"
graph = tf.get_default_graph()


data_path = './data/'
img_set = np.load(data_path + 'downsa_img.npy')
print img_set.shape


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    print "variables initialized"
    score_list = list()
    pool5_tensor = graph.get_operation_by_name('import/pool5').outputs[0]
    for batch in np.array_split(img_set, img_set.shape[0]/120):
        print 'batch shape',batch.shape
        assert batch.shape[1:] == (16, 16, 3)
        batch = batch / 255.00
        assert (np.max(batch) <= 1.0)
        feed_dict = {images: batch}
        tmp_s = sess.run(pool5_tensor, feed_dict=feed_dict)
        lens = tmp_s.shape[0]
        tmp_s = tmp_s.reshape(lens,-1)
        print 'tmp_s shape', tmp_s.shape
        score_list.append(tmp_s)

score = np.log1p(np.concatenate(score_list, axis=0))
print 'output shape:',score.shape
np.save(data_path + 'score',score)

