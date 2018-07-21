# coding: utf-8
import tensorflow as tf

#classify_image_graph_def.pb是网上下载的谷歌训练好的InceptionV3模型
with tf.Session() as sess:
    #classify_image_graph_def.pb文件位置
    with tf.gfile.FastGFile("..\\Model\\classify_image_graph_def.pb", 'rb') as f:
        #创建一个图来存放谷歌训练好的InceptionV3模型
        graph_def = tf.GraphDef()
        #读取InceptionV3模型文件，存放到一个图中
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='InceptionV3')
    #保存图到TensorBoard文件
    writer = tf.summary.FileWriter("InceptionV3Log", sess.graph)
    writer.close()


    # CMD下 cd D:\ideaworkspace\TensorFlow\TensorBoard\

    #查看上述代码生成的TensorBoard
    #tensorboard --logdir=InceptionV3Log