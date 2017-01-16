# -*- coding: utf-8 -*-

import tensorflow as tf
import utils
import numpy as np
import skimage.io
import gc
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('file_name', type=str)
args = parser.parse_args()
ori_path = args.file_name

with open("vgg16.tfmodel", mode='rb') as f:
  fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
images = tf.placeholder("float", [None, 16, 16, 3])
tf.import_graph_def(graph_def, input_map={ "images": images })
print "graph loaded from disk"
graph = tf.get_default_graph()


##split image##
ele_w = 16
ele_h = 16
#ori_img = skimage.img_as_ubyte(skimage.io.imread(ori_path))[1000:2500,1900:3400]
ori_img = skimage.img_as_ubyte(skimage.io.imread(ori_path))
S = ori_img.shape
ori_img = skimage.img_as_ubyte(skimage.transform.resize(ori_img, (int(S[0]/float(S[1])*224*12),224*12)))
S = ori_img.shape
ori_img = ori_img[:(S[0]/ele_h)*ele_h,:(S[1]/ele_w)*ele_w]
S = ori_img.shape
ori_img_split = ori_img.view()
ori_img_split = ori_img_split.reshape(S[0],S[1]/ele_w,ele_w,3)
ori_img_split = ori_img_split.swapaxes(1,0)
ori_img_split = ori_img_split.reshape(S[0]/ele_h*S[1]/ele_w, ele_h, ele_w, 3)
print ori_img_split.shape

###load element###
data_path = './data/'
ele_score = np.load(data_path + 'score.npy')
ele = np.load(data_path + 'downsa_img.npy')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print "variables initialized" 
    pool5_tensor = graph.get_operation_by_name('import/pool5').outputs[0]
    score_list = list()
    for batch in np.array_split(ori_img_split, ori_img_split.shape[0]/120+1):
        try:
            print 'batch shape', batch.shape
            assert batch.shape[1:] == (16, 16, 3)
        except:
            print 'error'
            continue
        batch = batch/255.0
        assert(np.max(batch) <= 1.0)
        print 'batch max',np.max(batch)
        feed_dict = { images: batch}
        tmp_s = sess.run(pool5_tensor, feed_dict=feed_dict)
        lens = tmp_s.shape[0]
        tmp_s = tmp_s.reshape(lens,-1)
        print 'tmp_s shape', tmp_s.shape
        score_list.append(tmp_s)
    
    score = np.log1p(np.concatenate(score_list, axis=0))
    del score_list
    gc.collect()
    '''
    for batch in ori_img_split:
        batch = skimage.img_as_ubyte(skimage.transform.resize(batch, (224, 224)))
        print np.max(batch)
        feed_dict = {images:np.expand_dims(batch, axis=0)/255.0}
        tmp_s = sess.run(fc1_tensor, feed_dict=feed_dict)
        score.append(tmp_s)
    '''

print 'output shape:', score.shape

for i in range(ori_img_split.shape[0]):
    idx = np.argmin(np.linalg.norm(score[i]-ele_score, axis=1))  
    print 'current block idx',idx
    print '#', i
    tmp = ele[idx]
    print np.max(tmp)
    ori_img_split[i] = tmp
    print ori_img_split[i].shape

ori_img_split = ori_img_split.reshape(S[1]/ele_w, S[0], ele_w, 3)
ori_img_split = ori_img_split.swapaxes(1,0)
ori_img_split = ori_img_split.reshape(S[0], S[1], 3)
skimage.io.imsave('mosaic.png', skimage.img_as_ubyte(ori_img_split))
