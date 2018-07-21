# -*-  coding: UTF-8 -*-

#组建1层隐藏层的线性模型+激活函数神经网络，梯度下降求解最小损失函数，
#解决回归问题。画图查看拟合效果

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#构造样本数据集
#下述[:, np.newaxis]，增加一列，相当于reshape(200,1)，但更灵活
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  #[200,1]
#样本加噪声，生成均值为loc,标准差为scale的高斯随机数
noise = np.random.normal(loc=0, scale=0.02, size=x_data.shape)
y_data = np.square(x_data) + noise  #输入层形状：[200,1]

#定义输入输出变量，先用占位符定义好张量，待session初始化
x = tf.placeholder(dtype=tf.float32, shape=[None,1]) #N行样本，1列
y = tf.placeholder(dtype=tf.float32, shape=[None,1]) #N行样本，1列

#定义神经网络中间层
Weight_L1 = tf.Variable(tf.random_normal(shape=[1,10]))
Bias_L1 = tf.Variable(tf.zeros(shape=[200,10]))
#Weight_L1 * x + Bias_L1。注意矩阵乘法的顺序
y_L1 = tf.matmul(x, Weight_L1) + Bias_L1
#tanh是一种激活函数(Activation Function)：目的是向模型中加入非线性
output_L1 = tf.nn.tanh(y_L1)
#神经网络中间层输出形状：[200,1] * [1,10] = [200,10]


#定义神经网络输出层
Weight_L2 = tf.Variable(tf.random_normal(shape=[10,1]))
Bias_L2 = tf.Variable(tf.zeros(shape=[200,1]))
#Weight_L2 * output_L1 + Bias_L2。注意矩阵乘法的顺序
y_L2 = tf.matmul(output_L1, Weight_L2) + Bias_L2
#tanh是一种激活函数(Activation Function)：目的是向模型中加入非线性
prediction = tf.nn.tanh(y_L2)
#神经网络输出层输出形状：[200,10]] * [10,1] = [200,1]

#损失函数，二次代价函数 (真实值-预测值)的平方/样本数
loss = tf.reduce_mean(tf.square(y - prediction))
#用梯度下降法优化损失函数loss，让其最小。
# 步长--学习率(0~1之间)，但这个参数从使用经验来看，是最影响拟合效果的
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

#重要！！初始化数据流图中的所有全局变量
init = tf.global_variables_initializer()

#上面是客户端模型，下面到执行层tf.Session
with tf.Session() as sess:
    #变量初始化
    sess.run(init)

    #训练200次
    for _ in range(30):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    #训练完成，获取最终预测值
    y_predict = sess.run(prediction, feed_dict={x: x_data})

#绘图查看
plt.scatter(x_data, y_data)
plt.plot(x_data, y_predict)
plt.show()