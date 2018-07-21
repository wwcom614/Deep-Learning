# coding: utf-8

import os  #文件操作使用
from PIL import Image  #读取图片给matplotlib显示使用
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#1. 读取训练结果labels文件output_labels.txt，转换成行id和名称的键值对字典
id_name_fileHandle = open("output_labels.txt", mode='r')
# 最终要存储到字典labelId_to_labelName中：{labelId: labelName}
labelId_to_labelName = {}
for labelId, line in enumerate(id_name_fileHandle.readlines()):
    #去掉换行符
    line = line.strip('\n')
    labelId_to_labelName[labelId] = line

#3.测试图片预测结果转换函数，通过字典labelId_to_labelName，将预测labelId转换为labelName
def id_to_name_func(id):
    if id not in labelId_to_labelName:
        return ''
    return labelId_to_labelName[id]


#2. 创建一个图，引入自己训练好的模型output_graph.pb
with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
    #创建一个图来存放谷歌训练好的模型
    graph_def = tf.GraphDef()
    #读取模型文件，存放到一个图中
    graph_def.ParseFromString(f.read())
    #注意这个地方name为空，否则下方的softmax:0会找不到
    tf.import_graph_def(graph_def, name='')


#服务端执行
with tf.Session() as sess:
    # 常用的tensors:
    # 'softmax:0': A tensor containing the normalized prediction across 1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image.
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #遍历图片所在目录,os.walk(path)返回三个值：parent,dirnames,filenames，
    # 分别表示path的路径、path路径下的文件夹的名字、path路径下文件夹以外的其他文件
    for parent, dirnames, filenames in os.walk('D:\\testimages\\'):
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
            #排序找到最大的概率值的索引:将prediction1d降序，取前1个值的index
            top_index = np.argsort(-prediction1d)[:1]

            for index in top_index:
                #获取分类描述
                class_desc = id_to_name_func(index)
                #获取该分类的置信度
                score = prediction1d[index]
                print("分类描述：", class_desc, "置信度：", score)

            print(image_path)
            #显示图片
            plt.imshow(Image.open(image_path))
            plt.axis('off')
            plt.show()

    image_data_fileHandle.close()





