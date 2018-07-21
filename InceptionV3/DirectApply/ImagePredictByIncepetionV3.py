# coding: utf-8

import os  #文件操作使用
from PIL import Image  #读取图片给matplotlib显示使用
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#读取自己编写的Id_Desc_MapToFile.py生成的id_desc.txt文件，存储到字典id_desc_dict中
id_desc_dict = {}
id_desc_path_file = "..\\InceptionV3_data\\id_desc.txt"
id_desc_fileHandle = open(id_desc_path_file, mode='r')
id_desc_dict = eval(id_desc_fileHandle.read())
id_desc_fileHandle.close()

#创建一个图引入google训练好的模型classify_image_graph_def.pb
with tf.gfile.FastGFile("..\\InceptionV3_Model\\classify_image_graph_def.pb", 'rb') as f:
    #创建一个图来存放谷歌训练好的InceptionV3模型
    graph_def = tf.GraphDef()
    #读取InceptionV3模型文件，存放到一个图中
    graph_def.ParseFromString(f.read())
    #注意这个地方name为空，否则下方的softmax:0会找不到
    tf.import_graph_def(graph_def, name='')



#服务端执行
with tf.Session() as sess:
    # 常用的tensors:
    # 'softmax:0': A tensor containing the normalized prediction across 1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    #遍历图片所在目录,os.walk(path)返回三个值：parent,dirnames,filenames，
    # 分别表示path的路径、path路径下的文件夹的名字、path路径下文件夹以外的其他文件
    for parent, dirnames, filenames in os.walk('images\\'):
        for file in filenames:
            #每张图片的路径+文件名
            image_path = os.path.join(parent,file)

            #读取图片
            image_data_fileHandle = open(image_path, 'rb')
            image_data = image_data_fileHandle.read()
            #网上下载的刚好都是JPEG图片
            prediction2d = sess.run(fetches=softmax_tensor, feed_dict={'DecodeJpeg/contents:0': image_data})
            #squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
            # 1000个概率值
            prediction1d = np.squeeze(prediction2d)
            #排序找到最大的概率值的索引:将prediction1d降序，取前5个值的index
            top_index = np.argsort(-prediction1d)[:5]
            image_data_fileHandle.close()

            for index in top_index:
                #获取分类描述
                class_desc = id_desc_dict[index]
                #获取该分类的置信度
                score = prediction1d[index]
                print("分类描述：", class_desc, "置信度：", score)

            print(image_path)
            #显示图片
            plt.imshow(Image.open(image_path))
            plt.axis('off')
            plt.show()







