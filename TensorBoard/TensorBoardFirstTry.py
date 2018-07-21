# -*-  coding: UTF-8 -*-
import tensorflow as tf


W = tf.Variable(2.0, dtype=tf.float32, name="Weight")
b = tf.Variable(1.0, dtype=tf.float32, name="Bias")
x = tf.placeholder(dtype=tf.float32, name="Input")

with tf.name_scope("Output"):  #输出命名空间
    y = W * x +b


#重要：初始化所有变量，这样变量才可以使用。常量不需要初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #注意Windows环境下的的路径为了方便可以使用r''，或把\都替换成/或\\
    writer = tf.summary.FileWriter(logdir=r'..\TensorBoard\FirstTry_Log', graph=sess.graph)
    result = sess.run(y, {x: 3.0})
    print("y:", result)


# CMD下 cd D:\ideaworkspace\TensorFlow\TensorBoard\

#查看上述代码生成的TensorBoard
#tensorboard --logdir=FirstTry_Log

# 查看Linear包下的Linear_Mnist.py生成的TensorBoard
# tensorboard --logdir=Linear_Mnist_Log

# 查看Linear包下的Linear_Mnist.py生成的TensorBoard
# tensorboard --logdir=CNN_NN_Conv2d_Mnist_Log