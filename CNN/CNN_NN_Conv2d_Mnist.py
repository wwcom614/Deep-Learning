import tensorflow as tf

#载入手写数字识别数据集60000行训练数据集(mnist.train)和10000行测试数据集(mnist.test) 28*28像素的灰度图片集
from tensorflow.examples.tutorials.mnist import input_data

#载入手写识别数据集。one_hot独热码，有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制，例如
# 0：1000000000
# 1：0100000000
# 2：0010000000
mnist = input_data.read_data_sets('..\\mnist_data', one_hot=True)

#定义每次取的样本数量
batch_size = 100

#自定义一个初始化权重值的函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.1) #截断正态分布，这种初始化方法比全0拟合效果好
    return tf.Variable(initial)

#自定义一个初始化偏置值的函数
def bias_variable(shape):
    initial = tf.constant(shape=shape, value=0.1)
    return tf.Variable(initial)

#自定义一个2维卷积函数
def conv2d(input, filter):
    # input 是个4维Tensor [每次使用多少个样本batch_size, 图片高in_height，图片宽in_width, 黑白图片in_channels=1]
    # filter是卷积核(滤波器) [filter_height，filter_width, in_channels, out_channels]
    # strides是步长 strides[0]=strides[3]=1, strides[1]表示x方向步长，strides[2]表示y方向步长
    # padding， SAME补0， VALID不补0
    return tf.nn.conv2d(input=input, filter=filter, strides=[1,1,1,1], padding='SAME')

#自定义一个池化层(使用max类型的，还有mean类型和随机类型，可以尝试)
def max_pool_2x2(value):
    # ksize是池化窗口大小，ksize[0]=ksize[3]=1, ksize[1]表示x方向窗口大小，ksize[2]表示y方向窗口大小
    # strides是步长 strides[0]=strides[3]=1, strides[1]表示x方向步长，strides[2]表示y方向步长
    # padding， SAME补0， VALID不补0
    return tf.nn.max_pool(value=value, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#TesorBoard  输入input
with tf.name_scope('input'):
    #定义变量为tf空张量
    # 将像素值打成列，作为特征值，28*28=784。 横行是样本数，用None定义，可以是任意个样本
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    # 列是特征值，10个数字。横行是样本数，用None定义，可以是任意个样本
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    with tf.name_scope('x_image'): #TesorBoard使用
        #改变输入x为4D张量[每次使用多少个样本batch_size, 图片高in_height，图片宽in_width, 黑白图片in_channels=1]
        x_image = tf.reshape(x, [-1,28,28,1])

#TesorBoard  第1层卷积conv1
with tf.name_scope('conv1'):
    with tf.name_scope('W_conv1'): #TesorBoard使用
        #初始化第1个卷积层的权重值和偏置值
        # 5*5的卷积采样窗口，从1个输入平面(黑白图片，彩色是3)，使用32个卷积核抽取特征，输出32个特征平面
        W_conv1 = weight_variable([5,5,1,32])
    with tf.name_scope('b_conv1'): #TesorBoard使用
        # 每个卷积核对应一个偏置值
        b_conv1 = bias_variable([32])

    with tf.name_scope('conv2d_1'): #TesorBoard使用
        #第1层卷积  输出[-1,28,28,32]
        conv2d_1 = conv2d(input=x_image, filter=W_conv1) + b_conv1
    with tf.name_scope('act_conv1'): #TesorBoard使用
        #第1层卷积后，使用激活函数relu增加非线性  [-1,28,28,32]
        act_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('pool1'): #TesorBoard使用
        #第1层池化降维  输出[-1,14,14,32]
        pool1 = max_pool_2x2(value=act_conv1)
        #28*28的图片，第1次卷积后还是28*28的图片(因为卷积使用的是samepadding)。第1次池化后变为14*14

#TesorBoard  第2层卷积conv2
with tf.name_scope('conv2'):
    with tf.name_scope('W_conv2'): #TesorBoard使用
        #初始化第2个卷积层的权重值和偏置值
        # 5*5的卷积采样窗口，从1个输入平面(黑白图片，彩色是3)，使用32个卷积核抽取特征，输出32个特征平面
        W_conv2 = weight_variable([5,5,32,64])
    with tf.name_scope('b_conv1'): #TesorBoard使用
        # 每个卷积核对应一个偏置值
        b_conv2 = bias_variable([64])
    with tf.name_scope('conv2d_2'): #TesorBoard使用
        #第2层卷积
        conv2d_2 = conv2d(input=pool1,filter=W_conv2) + b_conv2
    with tf.name_scope('act_conv2'): #TesorBoard使用
        #第2层卷积后，使用激活函数relu增加非线性
        act_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('pool2'): #TesorBoard使用
        #第2层池化降维
        pool2 = max_pool_2x2(value=act_conv2)
        #第1层池化后14*14的图片，第2次卷积后还是14*14的图片(因为卷积使用的是samepadding)。第2次池化后变为7*7
        #那么pool2是64个特征的7*7的平面

#TesorBoard  第1层全连接fc1
with tf.name_scope('fc1'):
    with tf.name_scope('W_fc1'): #TesorBoard使用
        #初始化第1个全连接层的权重值
        #pool2是64个特征的7*7的平面, 下一步想全连接到1024个神经元的全连接层
        W_fc1 = weight_variable([64*7*7, 1024])
    with tf.name_scope('b_fc1'): #TesorBoard使用
        # 每个神经元对应一个偏置值
        b_fc1 = bias_variable([1024])

    with tf.name_scope('pool2_flat'): #TesorBoard使用
        #第2层池化降维结果扁平化为一维
        pool2_flat = tf.reshape(pool2, [-1, 64*7*7], name='pool2_flat')

    with tf.name_scope('wx_plus_b1'): #TesorBoard使用
        #第1个全连接层的输出
        wx_plus_b1 = tf.matmul(pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('fc1'): #TesorBoard使用
        #第1个全连接层的输出后，使用激活函数relu增加非线性
        fc1 = tf.nn.relu(wx_plus_b1)

    with tf.name_scope('keep_prob'): #TesorBoard使用
        #dropout亚采样防止过拟合
        keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    with tf.name_scope('fc1_drop'): #TesorBoard使用
        fc1_drop = tf.nn.dropout(fc1, keep_prob=keep_prob, name='fc1_drop')

#TesorBoard  第2层全连接fc2
with tf.name_scope('fc2'):
    with tf.name_scope('W_fc2'): #TesorBoard使用
        #初始化第2个全连接层的权重值
        W_fc2 = weight_variable([1024, 10])
    with tf.name_scope('b_fc2'): #TesorBoard使用
        # 每个神经元对应一个偏置值
        b_fc2 = bias_variable([10])

    with tf.name_scope('wx_plus_b2'): #TesorBoard使用
        #第2个全连接层的输出
        wx_plus_b2 = tf.matmul(fc1_drop, W_fc2) + b_fc2

    with tf.name_scope('prediction'): #TesorBoard使用
        #预测输出
        prediction = tf.nn.softmax(wx_plus_b2)

with tf.name_scope('cross_entropy'): #TesorBoard使用
    #计算交叉熵代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction, name='cross_entropy'))
    tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('train'): #TesorBoard使用
    #使用Adam优化器最小化交叉熵代价函数
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'): #TesorBoard使用
    #计算准确率
    with tf.name_scope('correct_prediction'): #TesorBoard使用
        #预测结果与真实值的对比放入一个bool列表中
        correct_prediction = tf.equal(tf.argmax(prediction,axis=1), tf.argmax(y, axis=1))
    with tf.name_scope('accuracy'): #TesorBoard使用
        #计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

#合并TensorBoard所有的tf.summary
merged = tf.summary.merge_all()

#上面是客户端模型，下面到执行层tf.Session
with tf.Session() as sess:
    #初始化全局变量
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('..\\TensorBoard\\CNN_NN_Conv2d_Mnist_Log\\Train', sess.graph)
    test_writer = tf.summary.FileWriter('..\\TensorBoard\\CNN_NN_Conv2d_Mnist_Log\\Test', sess.graph)
    #机器搓，仅遍历1遍样本
    for i in range(101):
        train_batch_x, train_batch_y = mnist.train.next_batch(batch_size=batch_size)
        sess.run(train_step, feed_dict={x:train_batch_x, y:train_batch_y, keep_prob:0.5})
        #记录训练集计算的参数到TensorBoard的log中
        train_summary = sess.run(merged, feed_dict={x:train_batch_x, y:train_batch_y, keep_prob:1.0})
        train_writer.add_summary(train_summary, i)

        #记录测试集计算的参数到TensorBoard的log中
        test_batch_x, test_batch_y = mnist.test.next_batch(batch_size=batch_size)
        test_summary = sess.run(merged, feed_dict={x:test_batch_x, y:test_batch_y, keep_prob:1.0})
        test_writer.add_summary(test_summary, i)

        train_accuracy = sess.run(accuracy, feed_dict={x:mnist.train.images[:10000], y:mnist.train.labels[:10000], keep_prob:1.0})
        test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        print("current iter:", i, ", train_accuracy:", train_accuracy, ", test_accuracy:", test_accuracy)