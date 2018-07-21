# coding: utf-8

import tensorflow as tf

from Captcha_MultiTask.nets import nets_factory


# 不同字符数量，总共有10种数字待识别训练
Captcha_Kinds = 10
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 训练时，每批次取几张图片
BATCH_SIZE = 200
# tfrecord文件存放路径
TFRECORDS_DIR = "D:/ideaworkspace/TensorFlow/Captcha_MultiTask/captcha/tfrecords/train.tfrecords"

# placeholder
# input data
x = tf.placeholder(tf.float32, [None, 224, 224])
#output labels
y0 = tf.placeholder(tf.float32, [None])
y1 = tf.placeholder(tf.float32, [None])
y2 = tf.placeholder(tf.float32, [None])
y3 = tf.placeholder(tf.float32, [None])

# 学习率
lr = tf.Variable(0.05, dtype=tf.float32)

# 从tfrecord目录读取tfrecord文件解析函数
# tensorflow有两种常用数据输入方法。
# 方法1比较简单：使用placeholder来占位，在真正run的时候通过feed字典把真实的输入传进去
# 方法1很方便，但是遇到大型数据的时候就会很吃力，由于在单线程环境下我们的IO操作一般都是同步阻塞的，势必会在一定程度上导致学习时间的增加，
# 尤其是相同的数据需要重复多次读取的情况下Feeding。同时，中间环节数据类型转换等操作也是不小的开销。
# 方法2是在Graph定义好文件读取的方法，让TF自己去从文件中读取数据，并解码成可使用的样本集TF直接从文件中读取数据，
# 由于使用了多线程，使得IO操作不再阻塞模型训练，同时为了实现线程间的数据传输引入了Queues
# 方法2流程：使用input pipeline读取tfrecords文件，然后随机乱序，生成文件序列，读取并解码数据，输入模型训练
# 首先用tf.train.string_input_producer读取tfrecords文件的list建立FIFO序列，可以声明num_epoches表示需要读取数据的次数
# 可以声明shuffle参数将tfrecords文件读入顺序打乱，然后定义TFRecordReader读取上面的序列返回下一个record，
# 用tf.parse_single_example对读取到TFRecords文件进行解码，根据保存的serialize example和feature字典返回feature所对应的值。
# 此时获得的值都是string，需要进一步解码为所需的数据类型。把图像数据的string reshape成原始图像后可以进行preprocessing操作。
# 此外，还可以通过tf.train.batch(测试集)或者tf.train.shuffle_batch(训练集)将图像生成batch序列。
# 方法2：
def read_and_decode(tfrecord_dir):
    # 使用tf.train.string_input_producer函数，入参是要读取的文件的名字，输出会将全部tf文件打包为一个tf内部的queue类型
    # 注意：tf.train.string_input_producer函数的shuffle参数默认是True，会打乱队列；此外，num_epochs参数是指的过几遍训练数据
    filename_queue = tf.train.string_input_producer([tfrecord_dir], shuffle=False, num_epochs=1)
    # 接下来，使用某种文件读取器reader，读取文件名队列并解码，输入 tf.train.shuffle_batch 函数中，生成 batch 队列，传递给下一层
    # 不同的文件结构使用不同的文件读取器reader：
    # 如果读取的文件是像 CSV 那样的文本文件，用的文件读取器和解码器就是 TextLineReader 和 decode_csv
    # 如果读取的数据是像 cifar10 那样的 .bin 格式的二进制文件，就用 tf.FixedLengthRecordReader 和 tf.decode_raw 读取固定长度的文件读取器和解码器
    # 如果读取的数据是图片，或者是其他类型的格式，那么先把数据转换成 TensorFlow 的标准支持格式 tfrecords(谷歌protobuf二进制文件)：
    # 修改 tf.train.Example 的Features，将protobuf序列化为一个字符串，再通过 tf.python_io.TFRecordWriter 将序列化的字符串写入tfrecords，
    # 然后再用跟上面一样的方式读取tfrecords，只是读取器变成了tf.TFRecordReader，之后通过一个解析器tf.parse_single_example ，图片接下来用解码器tf.decode_raw解码
    reader = tf.TFRecordReader()
    # 用reader的read方法，这个方法需要一个IO类型的参数，就是我们上边string_input_producer输出的那个queue了，
    # reader从这个queue中取一个文件目录，然后打开它经行一次读取，reader的返回是一个tensor
    _, serialized_file = reader.read(filename_queue)
    #对这个tensor做些数据与处理,解析每列存入features
    features = tf.parse_single_example(serialized_file,
                                       features={
                                           'image' : tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    # 从features['image']获取图片数据
    image = tf.decode_raw(features['image'], tf.uint8)
    # tf.train.shuffle_batch必须确定如下shape。
    # 这样tf.train.shuffle_batch会返回[batch_size,[224, 224]]
    image = tf.reshape(image, [224, 224])
    # 图片预处理，将数值转为-1~1之间的值
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # 获取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, label0, label1, label2, label3


# 调用read_and_decode函数，获取图片数据和标签
image, label0, label1, label2, label3 = read_and_decode(TFRECORDS_DIR)

# 上面输出的tensor代表的是一个样本（[高宽管道]），但是训练网络的时候的输入一般都是一推样本（[样本数高宽*管道]），
# 我们就要用tf.train.batch或者tf.train.shuffle_batch这个函数把一个一个小样本的tensor打包成一个高一维度的样本batch，
# 这些函数的输入是单个样本，输出就是4D的样本batch了，其内部原理是创建一个queue，然后不断调用你的单样本tensor获得样本，
# 直到queue里边有足够的样本，然后一次返回一堆样本，组成样本batch
# 如果不使用tf.train.shuffle_batch，会导致生成的样本和label之间对应不上，乱序
# tf.train.batch与tf.train.shuffle_batch函数是单个Reader读取，但是可以多线程，2个线程就达到了速度的极限
image_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
    [image, label0, label1, label2, label3], batch_size = BATCH_SIZE,
    # capacity是队列的最大容量
    # min_after_dequeue是出队后，队列至少剩下min_after_dequeue个数据
    # 使用num_threads个线程处理。我就生成了1个tfrecord文件，所以使用一个线程处理
    capacity=200, min_after_dequeue=100, num_threads=1)

#定义网络结构，使用alexnet_v2模型训练
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=Captcha_Kinds,
    weight_decay=0.005,
    is_training=True)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


with tf.Session() as sess:
    # 初始化全局变量和局部变量
    sess.run(init)

    # alexnet模型的要求输入格式inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    # 数据输入网络得到输出值。end_points没用到，只用到了4个logits，预测值
    logits0,logits1,logits2,logits3,end_points = train_network_fn(X)

    # 把标签转成one_hot的形式，真实值
    one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=Captcha_Kinds)
    one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=Captcha_Kinds)
    one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=Captcha_Kinds)
    one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=Captcha_Kinds)

    # 计算真实值和预测值的交叉熵损失函数loss
    loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits0,labels=one_hot_labels0))
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1,labels=one_hot_labels1))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2,labels=one_hot_labels2))
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits3,labels=one_hot_labels3))
    # 计算总的loss
    total_loss = (loss0+loss1+loss2+loss3)/4.0
    # 优化total_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

    # 计算准确率
    correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0,1),tf.argmax(logits0,1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0,tf.float32))

    correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1,1),tf.argmax(logits1,1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1,tf.float32))

    correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2,1),tf.argmax(logits2,1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2,tf.float32))

    correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3,1),tf.argmax(logits3,1))
    accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3,tf.float32))

    # 保存模型
    saver = tf.train.Saver()


    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()


    # 启动QueueRunner, 此时文件名队列已经进队
    # tf.train.start_queue_runners一定要运行，且其位置要在定义好读取graph之后，在真正run之前，
    # 其作用是把queue里边的内容初始化，不跑这句的话，最开始的string_input_producer那里就没用，整个读取流水线都没用了。
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    for i in range(6001):
        # 随机获取一个批次的数据和标签
        b_image, b_label0, b_label1 ,b_label2 ,b_label3 = sess.run([image_batch, label_batch0, label_batch1, label_batch2, label_batch3])
        # 优化模型
        # image和label一定要一起run，使得image和label是在一张graph里边的，
        # 跑一次graph，这两个tensor都会出结果，且同一次跑出来的image和label才是对应的
        sess.run(optimizer, feed_dict={x: b_image, y0:b_label0, y1: b_label1, y2: b_label2, y3: b_label3})

        # 每迭代20次计算一次loss和准确率
        if i % 20 == 0:
            # 每迭代2000次降低一次学习率，简单起见，步长缩减为原来的1/3
            if i%2000 == 0:
                sess.run(tf.assign(lr, lr/3))
            acc0,acc1,acc2,acc3,loss_ = sess.run(fetches=[accuracy0,accuracy1,accuracy2,accuracy3,total_loss],feed_dict={x: b_image,
                                                                                                                 y0: b_label0,
                                                                                                                 y1: b_label1,
                                                                                                                 y2: b_label2,
                                                                                                                 y3: b_label3})
            learning_rate = sess.run(lr)
            print ("Iter:%d  Loss:%.3f  Accuracy:%.2f,%.2f,%.2f,%.2f  Learning_rate:%.4f" % (i,loss_,acc0,acc1,acc2,acc3,learning_rate))

            # 保存模型
            #训练停止条件，可以对准确率达到多少，或者loss小于多少，或者直接指定训练次数
            # if acc0 > 0.90 and acc1 > 0.90 and acc2 > 0.90 and acc3 > 0.90:
            if i==6000:
                #global_step是把训练次数加到模型名称里
                saver.save(sess, "./models/captcha_predict.model", global_step=i)
                break

                # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)






