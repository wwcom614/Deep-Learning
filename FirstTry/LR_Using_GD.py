# -*-  coding: UTF-8 -*-

#用梯度下降法求解线性回归问题

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#构造加噪声的线性数据作为数据集Orignal Data
points_num = 100  #构造这么多个点
data = []
for i in range(points_num):
    #生成均值为loc,标准差为scale的高斯随机数
    x_tmp = np.random.normal(loc=0.0, scale=0.66)
    #用这些x点对应线性方程生成 y = 0.1 * x + 0.2
    #权重Weight=0.1 , 偏差Bias=0.2
    y_tmp = 0.1 * x_tmp + 0.2 + np.random.normal(loc=0.0, scale=0.08)
    data.append([x_tmp, y_tmp])
x_data = [v[0] for v in data]
y_data = [v[1] for v in data]


#构造线性回归模型--神经网络中间层
#初始化Weight向量，在minval和maxval之间均匀分布
W = tf.Variable(tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0))
#初始化Bias向量
b = tf.Variable(tf.zeros(shape=[1]))
#构造线性回归模型W * x_data + b，预测y_predict
y_predict = W * x_data + b

#定义损失函数(loss function),也叫二次代价函数(cost function)，(真实值-预测值)的平方/样本数
loss = tf.reduce_mean(tf.square(y_predict - y_data))

#用梯度下降法优化损失函数loss，让其最小
#定义一个优化器：使用了tf.train中的基于梯度下降法的优化器。
# 机器比较挫，步长--学习率(0~1之间)使用了较大的值，但这个参数从使用经验来看，是最影响拟合效果的
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3)
train = optimizer.minimize(loss)

#重要！！初始化数据流图中的所有全局变量
init = tf.global_variables_initializer()

#上面是客户端模型，下面到执行层tf.Session
with tf.Session() as sess:
    sess.run(init)
    #训练30次
    for step in range(30):
        sess.run(train)
        print("Step= %d, Loss= %f, Weight= %f, Bias= %f" %(step, sess.run(loss), sess.run(W), sess.run(b)))


    #绘制拟合情况
    plt.plot(x_data, y_data, 'r*', label="Orignal Data")
    plt.plot(x_data, sess.run(y_predict), label='Fitted Line')
    plt.legend()
    plt.title("Linear Regression using Gradient Descent")
    plt.show()

