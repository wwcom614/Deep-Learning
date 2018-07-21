# -*-  coding: UTF-8 -*-

#自己动手实现cnn卷积神经网络，拟合手写数字识别数据集

import numpy as np
import tensorflow as tf

#载入手写数字识别数据集(55000 * 28 *28) 55000张 28*28像素的灰度图片集
from tensorflow.examples.tutorials.mnist import input_data
#one_hot独热码，有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制，例如
# 0：1000000000
# 1：0100000000
# 2：0010000000
mnist = input_data.read_data_sets('..\\mnist_data', one_hot=True)

# 变量空间占位，先把量张开，呵呵。后续待session初始化
# None表示张量(Tensor)的第一个维度可以是任意长度
# 图片是28*28像素的，灰度取值0~255，除以255.归一化
input_x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])/ 255.
#10个数字的真实标签
output_y = tf.placeholder(dtype=tf.int32, shape=[None, 10])
#将卷积神经网络的input_x变成28*28*1的矩阵image。
# 注：非彩色图片，灰度图片，所以最后一个维度是1，如果是彩色要是RGB值
input_x_image = tf.reshape(input_x, [-1, 28, 28, 1])

#从mnist测试数据集中选取0~2999,前3000个手写数字的图片和对应标签作为自己的测试数据集
test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]

#构建卷积神经网络
#第1层卷积，使用32个filters。
# 因为图片像素是2维的，所以使用conv2d。TensorFlow提供了有2种卷积方法，都行。
# 区别是：tf.layers.conv2d是基于tf.nn.conv2d封装后的方法，参数更多一些，两者参数也不同
conv1 = tf.layers.conv2d(inputs=input_x_image,  #形状28*28*1,
                         filters=32,            #过滤器：会生成深度为filters个原有tensor输出
                         kernel_size=[5,5],     #卷积核横向大小，在原tensor的一次采样大小，例如5*5
                         strides=1,             #步长。每隔多少步，进行一次采样。strides=1是完全采样
                         padding='same',        #补零。padding=‘same’采样之后如果想还是与原tensor维度保持一样，需要补0(本例中会补2圈0)。padding=‘valid'是不补0
                        activation=tf.nn.relu   #激活函数使用Relu
                        )                       #第1层卷积后，输出的conv1形状是[28, 28, 32]

#第1层池化(亚采样，目的是降维)
pool1 = tf.layers.max_pooling2d(inputs=conv1,   #第1层卷积输出conv1作为池化层输入，形状是[28, 28, 32]
                                pool_size=[2,2],#采样大小，例如2*2
                                strides=2,      #步长。每隔多少步，进行一次采样。strides=1是完全采样
                                )               #第1层池化后，输出的pool1形状是[14, 14, 32]

#第2层卷积，使用64个filters。
conv2 = tf.layers.conv2d(inputs=pool1,          #第1层池化后，输出的pool1形状是[14, 14, 32]
                         filters=64,            #过滤器：会生成深度为filters个原有tensor输出
                         kernel_size=[5,5],     #卷积核横向大小，在原tensor的一次采样大小，例如5*5
                         strides=1,             #步长。每隔多少步，进行一次采样。strides=1是完全采样
                         padding='same',        #补零。padding=‘same’采样之后如果想还是与原tensor维度保持一样，需要补0(本例中会补2圈0)。padding=‘valid'是不补0
                         activation=tf.nn.relu   #激活函数使用Relu
                         )                       #第2层卷积后，输出的conv1形状是[14, 14, 64]

#第2层池化(亚采样，目的是降维)
pool2 = tf.layers.max_pooling2d(inputs=conv2,   #第2层卷积输出conv2作为池化层输入，形状是[14, 14, 64]
                                pool_size=[2,2],#采样大小，例如2*2
                                strides=2,      #步长。每隔多少步，进行一次采样。strides=1是完全采样
                                )               #第1层池化后，输出的pool1形状是[7, 7, 64]

#为输出做准备，扁平化flat
flat = tf.reshape(pool2, [-1, 7*7*64])   #扁平化后的flat为[7*7*64,]

#生成units=1024个神经元的全连接层
dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

#输出继续降维，丢弃掉rate=50%的dense数据
dropout = tf.layers.dropout(inputs=dense1, rate=0.5)

#在dropout基础上，最终输出units=10个神经元的全连接层作为最终预测值, 形状[1, 1, 10]
dense_predict = tf.layers.dense(inputs=dropout, units=10)

#计算误差：先计算交叉熵(Cross Entropy，再用激活函数Softmax计算百分比概率)
#onehot_labels是真实结果，logits是算法预测结果
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=dense_predict)

#使用Adam优化器来最小化误差，尝试学习率0.01(机器搓，
# 好机器可以再将学习率调小一个数量级做尝试，从经验来看这是最有效果的超参数)
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

#效果评估。计算预测值和实际标签真实值的匹配程度--精度
#tf.metrics.accuracy会创建并返回2个局部变量：accuracy和update_op,
#labels是真实结果，predictions是预测值
#tf.argmax，在某个维度上寻找最大值，返回该最大值的索引
accuracy = tf.metrics.accuracy(labels=tf.argmax(output_y, axis=1),
                               predictions=tf.argmax(dense_predict, axis=1))[1]

#上面是客户端模型，下面到执行层tf.Session
with tf.Session() as sess:
    #重要！！：初始化数据流图中的所有全局变量和局部变量(accuracy中的accuracy和update_op)
    #2个操作形成一组操作tf.group。tf.global_variables_initializer()无法初始化局部变量
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    for i in range(1000):
        #依次从mnist的训练数据集中取下一个50个样本,每次训练50个样本
        batch = mnist.train.next_batch(50)
        #{}字典，key是训练所需参数，value是该参数从哪获取
        train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
        if i % 100 == 0:
            test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
            print("Step=%d, train_loss=%.4f, test_accuracy=%.2f" %(i, train_loss, test_accuracy))
'''
Step=0, train_loss=2.2787, test_accuracy=0.13
Step=100, train_loss=0.3697, test_accuracy=0.52
Step=200, train_loss=0.0806, test_accuracy=0.66
Step=300, train_loss=0.2820, test_accuracy=0.73
Step=400, train_loss=0.2306, test_accuracy=0.77
Step=500, train_loss=0.1488, test_accuracy=0.80
Step=600, train_loss=0.0825, test_accuracy=0.83
Step=700, train_loss=0.0433, test_accuracy=0.84
Step=800, train_loss=0.1324, test_accuracy=0.86
Step=900, train_loss=0.1053, test_accuracy=0.87
'''

#看看预测结果如何
test_output = sess.run(dense_predict, {input_x: test_x[:20]})
predict_y = np.argmax(test_output, axis=1)
print('Predict:', predict_y)
print('Real:', np.argmax(test_y, axis=1))