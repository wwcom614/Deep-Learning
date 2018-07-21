# -*-  coding: UTF-8 -*-

import tensorflow as tf

#载入手写数字识别数据集60000行训练数据集(mnist.train)和10000行测试数据集(mnist.test) 28*28像素的灰度图片集
from tensorflow.examples.tutorials.mnist import input_data

#载入手写识别数据集。one_hot独热码，有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制，例如
# 0：1000000000
# 1：0100000000
# 2：0010000000
mnist = input_data.read_data_sets('..\\mnist_data', one_hot=True)

#定义每次取的训练样本数量
batch_size = 100
#计算全量训练样本一共需要取多少个批次能全部遍历训练样本集。 //是整除
n_batch = mnist.train.num_examples // batch_size

with tf.name_scope("input"):#TesorBoard使用
    #定义变量为tf空张量
    # 将像素值打成列，作为特征值，28*28=784。 横行是样本数，用None定义，可以是任意个样本
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x-input')
    # 列是特征值，10个数字。横行是样本数，用None定义，可以是任意个样本
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y-input')

with tf.name_scope("param"): #TesorBoard使用
    #为降低过拟合的可能性，使用Dropout(每次训练只使用指定百分比的神经元),此处定义保留数据的比例
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    # 随遍历梯度下降，不断调小学习率步长
    lr = tf.Variable(0.001, dtype=tf.float32, name='learning_rate')

from TensorBoard.VarAnalysis import variable_summaries
# 使用线性神经网络+激活函数模型拟合
with tf.name_scope("layer1"): #TesorBoard使用
    #第1层，对接输入
    # 初始化W1采用如下截断高斯随机数方式而不是初始化为全0，这样训练拟合效果更好
    with tf.name_scope("weight"):#TesorBoard使用
        W1 = tf.Variable(tf.truncated_normal(shape=[784, 500], mean=0, stddev=0.1), name='W1')
        variable_summaries(W1) #TensorBoard在Scalar中分析均值标准差等，在Distribution和hisgram中分析直方图
    with tf.name_scope("bias"):#TesorBoard使用
        # 初始化b1采用全0.1不是初始化为全0，这样训练拟合效果更好
        b1 = tf.Variable(tf.zeros([500]) + 0.1, name='b1')
        variable_summaries(b1) #TensorBoard在Scalar中分析均值标准差等，在Distribution和hisgram中分析直方图
    with tf.name_scope("Wx_plus_b"):#TesorBoard使用
        #线性模型
        Wx_plus_b = tf.matmul(x, W1) + b1
    with tf.name_scope("L1"):#TesorBoard使用
        #激活函数增加非线性
        L1 = tf.tanh(Wx_plus_b)
        #为降低过拟合的可能性，使用Dropout(每次训练只使用指定百分比的神经元)。注：使用Dropout后模型拟合收敛速度会变慢
    with tf.name_scope("L1_drop"):#TesorBoard使用
        # 测试的时候使用所有神经元
        L1_drop = tf.nn.dropout(L1,keep_prob=keep_prob)

with tf.name_scope("layer2"): #TesorBoard使用
    #第2层隐藏层
    with tf.name_scope("weight"):#TesorBoard使用
        # 初始化W2采用如下截断高斯随机数方式而不是初始化为全0，这样训练拟合效果更好
        W2 = tf.Variable(tf.truncated_normal(shape=[500, 300], mean=0, stddev=0.1),name='W2')
        variable_summaries(W2) #TensorBoard在Scalar中分析均值标准差等，在Distribution和hisgram中分析直方图
    with tf.name_scope("bias"):#TesorBoard使用
        # 初始化b2采用全0.1不是初始化为全0，这样训练拟合效果更好
        b2 = tf.Variable(tf.zeros([300]) + 0.1, name='b2')
        variable_summaries(b2) #TensorBoard在Scalar中分析均值标准差等，在Distribution和hisgram中分析直方图
    with tf.name_scope("Wx_plus_b"):#TesorBoard使用
        #线性模型
        Wx_plus_b = tf.matmul(L1_drop, W2) + b2
    with tf.name_scope("L2"):#TesorBoard使用
        #激活函数增加非线性
        L2 = tf.tanh(Wx_plus_b)
    with tf.name_scope("L2_drop"):#TesorBoard使用
        #为降低过拟合的可能性，使用Dropout(每次训练只使用指定百分比的神经元)。注：使用Dropout后模型拟合收敛速度会变慢
        # 测试的时候使用所有神经元
        L2_drop = tf.nn.dropout(L2,keep_prob=keep_prob)

with tf.name_scope("layer3"): #TesorBoard使用
    #第3层输出层
    with tf.name_scope("weight"):#TesorBoard使用
        # 初始化W3采用如下截断高斯随机数方式而不是初始化为全0，这样训练拟合效果更好
        W3 = tf.Variable(tf.truncated_normal(shape=[300, 10], mean=0, stddev=0.1))
        variable_summaries(W3) #TensorBoard在Scalar中分析均值标准差等，在Distribution和hisgram中分析直方图
    with tf.name_scope("bias"):#TesorBoard使用
        # 初始化b3采用全0.1不是初始化为全0，这样训练拟合效果更好
        b3 = tf.Variable(tf.zeros([10]) + 0.1)
        variable_summaries(b3) #TensorBoard在Scalar中分析均值标准差等，在Distribution和hisgram中分析直方图
    with tf.name_scope("Wx_plus_b"):#TesorBoard使用
        #线性模型
        Wx_plus_b = tf.matmul(L2_drop, W3) + b3
    with tf.name_scope("prediction"):#TesorBoard使用
        #最后一层的激活函数一般使用Softmax：将预测结果转换为概率显示
        prediction = tf.nn.softmax(Wx_plus_b)

#损失函数，也叫代价函数，使用与Softmax搭配使用的交叉熵(对数释然代价函数)
#labels是真实值，logits是预测值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
tf.summary.scalar('loss', loss) # 在TensorBoard中查看损失函数的变化

#实际应用：优化器需要尝试多种，看哪种最好
'''
SGD最慢，因为没有任何加速度
Momentum：当前权值的改变会受到上次权值改变的影响，有上次加速度在，特点是训练速度快；但路线最绕，因为没有提前一步计算下一步可能去哪，最后还会冲过头
NAG是聪明版本的Momentum，会提前计算下个位置的梯度。但也略绕
选路优先:Adadelta、Adagrad、Rmsprop
Adagrad：适用于数据分类自身样本量极不均衡的样本集。比较常见的数据给予较小的学习率调整参数，比较罕见的的数据给予较大的学习率调整参数。
优势在于不需要人为调节学习率，可自动调节；缺点：随迭代次数增多，因为分母是计算所有次梯度的平方均值会越来越大，导致学习率越来越小，最终学习率趋近与0
Rmsprop：借鉴了Adagrad的思想并改良，主要是分母是只计算最近t次梯度平方的均值，所以分母不会越来越大。
Adadelta：甚至不需要设置默认学习率，不需要使用学习率也可以达到好的效果
Adam：像Adadelta和Rmsprop一样，Adam会存储之前衰减的平方梯度，同时，它也会保存之前衰减的梯度。经过一些处理后，再使用类似Adadelta和Rmsprop的方式更新参数
'''
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# 最终的结果是10行2列的10个值，看第2列哪个值的概率最大，那么其位置argmax就是这个数字
# 真实值y和预测值prediction做equal对比，如果位置一样，也就是都是同一数字，那么结果是true，否则为false，结果放入一个布尔列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 将布尔列表correct_prediction转换为数字张量，true的值是1，false的值是0，然后求reduce_mean(和/所有数量),就是准确率了
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
tf.summary.scalar('accuracy', accuracy) # 在TensorBoard中查看准确率的变化

#重要！！初始化数据流图中的所有全局变量
init = tf.global_variables_initializer()

#合并所有在TensorBoard要监控的tf.summary
merged = tf.summary.merge_all()

#上面是客户端模型，下面到执行层tf.Session
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('..\\TensorBoard\\Linear_Mnist_Log', sess.graph)
    #基于全量训练样本做多少次拟合训练
    for epoch in range(10):
        #随循环执行次数epoch的增大，让学习率lr步长减小：开始下降步长大些，接近目标时步子放小，避免跑过头
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        #每次从训练样本中拿出batch_size的数据做训练，总共循环n_batch次，保证能使用过所有训练样本数据
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            #因数据过小，实际不进行dropout，只要配置keep_prob:1.0，意思是100%保留
            summary, _ = sess.run(fetches=[merged, train_step], feed_dict={x:batch_x, y:batch_y, keep_prob:1.0})

        writer.add_summary(summary, epoch) #将每次epoch最终的merged监控结果，添加到TensorBoard中
        learning_rate = sess.run(lr)
        test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        print("Current Iter:", epoch, ",Test Accuracy:", test_accuracy, ",Learning Rate:",learning_rate)
