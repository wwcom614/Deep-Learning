# -*-  coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#激活函数(Activation Function)：目的是向模型中加入非线性

#创建X轴输入数据
x = np.linspace(start=-10, stop=10, num=200)

#实现tf.nn中，几种常见的激活函数。维基百科有公式
def sigmoid(inputs):
    outputs = [1 / float(1 + np.exp(-x)) for x in inputs]
    return outputs

def relu(inputs):
    outputs = [x * (x > 0) for x in inputs]
    return outputs

def tanh(inputs):
    outputs = [(np.exp(x) - np.exp(-x)) / float(np.exp(x) + np.exp(-x)) for x in inputs]
    return outputs

def softplus(inputs):
    outputs = [np.log(1 + np.exp(x)) for x in inputs]
    return outputs

#Y轴输出数据
y_sigmoid = tf.nn.sigmoid(x)
y_relu = tf.nn.relu(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

#上面是客户端模型，下面到执行层tf.Session
with tf.Session() as sess:
    y_sigmoid, y_relu, y_tanh, y_softplus = sess.run([y_sigmoid, y_relu, y_tanh, y_softplus])

    #绘图
    plt.title("Activation Function")

    plt.subplot(221)
    plt.plot(x, y_sigmoid, label='sigmoid')
    plt.legend(loc='best')

    plt.subplot(222)
    plt.plot(x, y_relu, label='relu')
    plt.legend(loc='best')

    plt.subplot(223)
    plt.plot(x, y_tanh, label='tanh')
    plt.legend(loc='best')

    plt.subplot(224)
    plt.plot(x, y_softplus, label='softplus')
    plt.legend(loc='best')

    plt.show()