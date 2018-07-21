# coding: utf-8

import tensorflow as tf
import os
import random
import sys
from PIL import Image
import numpy as np


#训练样本和测试样本数分界点，先random样本，然后总样本数的前_NUM_TEST个是测试样本，后面的是训练样本
_NUM_TEST = 500

#调试方便，固定下随机种子
_RANDOM_SEED = 666

#数据集路径
IMAGES_DIR = 'D:/ideaworkspace/TensorFlow/Captcha_MultiTask/captcha/images/'

#tfrecord文件存放路径
TFRECORDS_DIR = "D:/ideaworkspace/TensorFlow/Captcha_MultiTask/captcha/tfrecords/"


#判断tfrecord文件是否存在
def _tfrecords_exists(tfrecords_dir):
    for train_or_test in ['train', 'test']:
        tfrecord_filename = os.path.join(tfrecords_dir, train_or_test + '.tfrecords')
        if not tf.gfile.Exists(tfrecord_filename):
            return False
    return True

#获取所有验证码图片全路径名+文件名存放在list中
def _get_path_imagename_list(IMAGES_DIR):
    path_imagename_list = []
    for filename in os.listdir(IMAGES_DIR):
        #获取文件路径
        path = os.path.join(IMAGES_DIR, filename)
        path_imagename_list.append(path)
    return path_imagename_list


#################################################################################
#4位验证码图片转为tfrecord, protobuf格式，标准写法
def image_to_tfrecord(image_data, label0, label1, label2, label3):
    #Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label0': int64_feature(label0),
        'label1': int64_feature(label1),
        'label2': int64_feature(label2),
        'label3': int64_feature(label3),
    }))

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
#################################################################################

# 遍历，把所有图片转为tfrecord文件
def _convert_images_to_tfrecords(train_or_test, path_imagename_list, tfrecords_dir):
    assert train_or_test in ['train', 'test']

    with tf.Session() as sess:
        #定义tfrecord文件的路径+名字
        path_tfrecord_name = os.path.join(tfrecords_dir, train_or_test + '.tfrecords')
        with tf.python_io.TFRecordWriter(path_tfrecord_name) as tfrecord_writer:
            for i,path_imagename in enumerate(path_imagename_list):
                try:
                    sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(path_imagename_list)))
                    sys.stdout.flush()

                    #读取图片
                    image_data = Image.open(path_imagename)
                    #准备使用alexnet模型训练，模型要求输入图片高和宽都为224，根据模型的结构resize
                    image_data = image_data.resize((224, 224))
                    #不需要彩色的，灰度化
                    image_data = np.array(image_data.convert('L'))
                    #将图片转化为bytes
                    image_data = image_data.tobytes()

                    #获取label ,[-1]得到的是XXXX.jpg,取前4个
                    labels = path_imagename.split('/')[-1][0:4]
                    num_labels = []
                    for j in range(4):
                        num_labels.append(int(labels[j]))

                    #生成protocol数据类型
                    tfrecord = image_to_tfrecord(image_data, num_labels[0], num_labels[1], num_labels[2], num_labels[3])
                    tfrecord_writer.write(tfrecord.SerializeToString())

                except IOError as e:
                    print('Could not read:',path_imagename)
                    print('Error:',e)
                    print('Skip it\n')
    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
#判断tfrecord文件是否存在
    #判断tfrecord文件是否存在
    if _tfrecords_exists(TFRECORDS_DIR):
        print('tfcecord文件已存在')
    else:
        #获得所有图片全路径+文件名列表
        path_imagename_list = _get_path_imagename_list(IMAGES_DIR)

        #把数据切分为训练集和测试集,并打乱
        random.seed(_RANDOM_SEED)
        random.shuffle(path_imagename_list)
        train_path_imagename_list = path_imagename_list[_NUM_TEST:]
        test_path_imagename_list = path_imagename_list[:_NUM_TEST]

        #把图片转换成tfrecord
        _convert_images_to_tfrecords('train', train_path_imagename_list, TFRECORDS_DIR)
        _convert_images_to_tfrecords('test', test_path_imagename_list, TFRECORDS_DIR)
        print('已生成tfcecord文件')

