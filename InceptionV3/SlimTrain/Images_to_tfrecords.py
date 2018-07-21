# coding: utf-8

import tensorflow as tf
import os
import random
import sys


#训练样本和测试样本数分界点，先random样本，然后总样本数的前_NUM_TEST个是测试样本，后面的是训练样本
_NUM_TEST = 10
#调试方便，固定下随机种子
_RANDOM_SEED = 666
#如果图片较多，转换为tfrecord时，需考虑分成多个分片数据块--借机练习下
_NUM_SHARDS = 2
#图片所在根路径
IMAGES_DIR = "D:/trainimages/"
#输出tfrecords文件路径
TFRECORDS_DIR = "D:/tfrecords/"
#标签文件名字
LABELS_DIR_FILENAME = "D:/tfrecords/labels.txt"

#IMAGES_DIR目录下的是子目录，子目录名称subdir_name就是分类名称，子目录内存放该类图片，例如
#D:\trainimages\animal\，IMAGES_DIR是D:\trainimages，分类名称就是子目录名称(subdir_name):animal
# 1.获取所有图片全路径名+文件名，以及其分类名称(IMAGES_DIR下的子目录名称subdir_name)
def _get_pathimagenames_and_classes(IMAGES_DIR):
    #用于存放IMAGES_DIR+subdir_name的目录列表
    path_list = []
    #分类名称
    classname_list = []
    for subdir_name in os.listdir(IMAGES_DIR):
        #合并文件路径
        path = os.path.join(IMAGES_DIR, subdir_name)
        #判断该路径是否为目录
        if os.path.isdir(path):
            path_list.append(path)
            classname_list.append(subdir_name)

    #用于存放每张图片的全路径名+文件名
    path_imagename_list = []
    #循环每个分类的文件夹
    for path in path_list:
        for image_name in os.listdir(path):
            path_image_name = os.path.join(path, image_name)
            #把图片加入图片列表
            path_imagename_list.append(path_image_name)

    return path_imagename_list, classname_list

# 2.遍历，把所有图片转为tfrecord文件
def _convert_images_to_tfrecords(train_or_test, path_imagename_list, classname_id_dict, tfrecord_dir):
    #确认数据名称是训练数据集或测试数据集
    assert train_or_test in ['train', 'test']
    #图片较多，转换为tfrecord时，需考虑分成多个分片数据块，做个练习
    #计算每个数据块平均存放多少数据量 文件数量/分片数取整
    num_per_shard = len(path_imagename_list) // _NUM_SHARDS
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                #定义输出的tfrecord文件的路径+名字
                path_tfrecord_name = _get_path_tfrecord_name(tfrecord_dir, train_or_test, shard_id)
                #写tfrecord固定用法tf.python_io.TFRecordWriter(tfrecord文件名)
                with tf.python_io.TFRecordWriter(path_tfrecord_name) as tfrecord_writer:
                    #每一个数据块开始的位置
                    start_ndx = shard_id * num_per_shard
                    #每一个数据块最后的位置
                    end_ndx = min((shard_id+1) * num_per_shard, len(path_imagename_list))
                    for i in range(start_ndx, end_ndx):
                        try:
                            #打印当前处理到第几张图片，在写第几个shard
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, len(path_imagename_list), shard_id))
                            sys.stdout.flush()
                            #读取图片
                            image_data = tf.gfile.FastGFile(path_imagename_list[i], 'rb').read()
                            #获取最后一个文件夹的名称，就是图片的类别名称
                            class_name = os.path.basename(os.path.dirname(path_imagename_list[i]))
                            #找到类别名称对应的id
                            class_id = classname_id_dict[class_name]
                            #生成tfrecord文件
                            tfrecord = image_to_tfrecord(image_data, b'jpg', class_id)
                            tfrecord_writer.write(tfrecord.SerializeToString())
                        except IOError as e:#考虑某些图片文件损坏，无法读取的情况
                            print("Could not read:", path_imagename_list[i])
                            print("Error:",e)
                            print("Skip it\n")

    sys.stdout.write('\n')
    sys.stdout.flush()


#定义输出tfrecord文件的路径+名字
def _get_path_tfrecord_name(tfrecord_dir, train_or_test, shard_id):
    tfrecord_filename = 'image_%s_%05d-of-%05d.tfrecord' % (train_or_test, shard_id, _NUM_SHARDS)
    return os.path.join(tfrecord_dir, tfrecord_filename)

#判断tfrecord文件是否存在
def _tfrecords_exists(tfrecords_dir):
    for train_or_test in ['train', 'test']:
        for shard_id in range(_NUM_SHARDS):
            #定义tfrecord文件的路径+名字
            tfrecord_filename = _get_path_tfrecord_name(tfrecords_dir, train_or_test, shard_id)
        if not tf.gfile.Exists(tfrecord_filename):
            return False
    return True

#################################################################################
#图片转为tfrecord, protobuf格式，标准写法
def image_to_tfrecord(image_data, image_format, class_id):
    #Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
    }))

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
#################################################################################

#输出labels文件函数
def write_label_file(id_classname_dict, labels_dir_filename=LABELS_DIR_FILENAME):
    with tf.gfile.Open(labels_dir_filename, 'w') as f:
        for id in id_classname_dict:
            class_name = id_classname_dict[id]
            f.write('%d:%s\n' % (id, class_name))


if __name__ == '__main__':
    #判断tfrecord文件是否存在
    if _tfrecords_exists(TFRECORDS_DIR):
        print('tfcecord文件已存在')
    else:
        #获得所有图片以及分类
        path_imagename_list, class_name_list = _get_pathimagenames_and_classes(IMAGES_DIR)
        #把分类转为字典格式，类似于{'house': 3, 'flower': 1, 'plane': 4, 'guitar': 2, 'animal': 0}
        #zip是打元组包的函数(class_names[0],0),(class_names[1],1)....
        classname_id_dict = dict(zip(class_name_list, range(len(class_name_list))))

        #把数据切分为训练集和测试集
        random.seed(_RANDOM_SEED)
        random.shuffle(path_imagename_list)
        train_path_imagename_list = path_imagename_list[_NUM_TEST:]
        test_path_imagename_list = path_imagename_list[:_NUM_TEST]

        #数据转换
        _convert_images_to_tfrecords('train', train_path_imagename_list, classname_id_dict, TFRECORDS_DIR)
        _convert_images_to_tfrecords('test', test_path_imagename_list, classname_id_dict, TFRECORDS_DIR)

        #输出labels文件
        id_classname_dict = dict(zip(range(len(class_name_list)), class_name_list))
        write_label_file(id_classname_dict)






