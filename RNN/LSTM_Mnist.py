# coding: utf-8
import tensorflow as tf

#载入手写数字识别数据集
from tensorflow.examples.tutorials.mnist import input_data
#载入手写识别数据集。one_hot独热码，有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制，例如
# 0：1000000000
# 1：0100000000
# 2：0010000000
mnist = input_data.read_data_sets('..\\mnist_data', one_hot=True)

#载入图片的像素是28*28
input_rows = 28 #输入的图片，有28个像素行
input_cols = 28 #输入的图片，有28个像素列
lstm_size = 100 # LSTM的隐藏层单元数
n_classes = 10  # 10种数字分类
batch_size = 50 #每次取batch_size个样本
n_batch = mnist.train.num_examples // batch_size  # 需要取n_batch次可以取完训练集

# 将像素值打成列，作为特征值，28*28=784。 横行是样本数，用None定义，可以是任意个样本
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
# 列是特征值，10个数字。横行是样本数，用None定义，可以是任意个样本
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

#初始化权重值
weight = tf.Variable(tf.truncated_normal(shape=[lstm_size, n_classes],mean=0, stddev=0.1))
#初始化偏置值
bias = tf.Variable(tf.constant(value=0.1, shape=[n_classes]))

#自定义一个RNN网络函数
def RNN(X, weight, bias):
    #将输入图片样本变形为 [样本数，图片行像素数，图片列像素数]
    inputs = tf.reshape(tensor=X, shape=[-1, input_rows, input_cols])
    #定义LSTM的cell
    from tensorflow.contrib import rnn

    lstm_cell = rnn.BasicLSTMCell(lstm_size)
    # final_state[0或者1，batch_size, LSTM的隐藏层单元数lstm_size
    # final_state[0]是cell state--中间cell的计算结果, 该单元输入+input gate + forget gate组合输出结果
    # final_state[1]是hidden state--最终最后一次输出的计算结果
    #outputs，是函数输出的结果 。也就是final_state[0] + output gate组合输出结果
    #outputs是每次的final_state[1]，最后一行outputs，就是final_state[1]
    #默认time_major==False, outputs = [batch_size, input_rows, LSTM的隐藏层单元数lstm_size]
    #time_major==True, outputs = [input_rows, batch_size, LSTM的隐藏层单元数lstm_size]
    outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=inputs, dtype=tf.float32)
    return tf.nn.softmax(logits=tf.matmul(final_state[1], weight) + bias)

#预测
prediction = RNN(x, weight, bias)

#交叉熵损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
#使用Adam优化器最小化交叉熵损失函数
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

#预测结果与真实值的对比放入一个bool列表中
correct_prediction = tf.equal(tf.argmax(prediction,axis=1), tf.argmax(y, axis=1))
#计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#尝试将模型保存为文件
saver = tf.train.Saver()

#上面是客户端模型，下面到执行层tf.Session
with tf.Session() as sess:
    #初始化全局变量
    sess.run(tf.global_variables_initializer())
    for epoch in range(20):
        for batch in range(batch_size):
            train_batch_x, train_batch_y = mnist.train.next_batch(batch_size=batch_size)
            sess.run(train_step, feed_dict={x:train_batch_x, y:train_batch_y})

        #保存模型到文件
        saver.save(sess=sess, save_path='Model\\LSTM_Mnist_Model.ckpt')

        #从文件加载到内存模型
        saver.restore(sess=sess, save_path='Model\\LSTM_Mnist_Model.ckpt')
        test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("current iter:", epoch, ", test_accuracy:", test_accuracy)





