# -*-  coding: UTF-8 -*-

import tensorflow as tf



input1 = tf.constant(1.0)
input2 = tf.constant(2.0)
input3 = tf.constant(3.0)

#创建占位符，feed时再赋值
input4 = tf.placeholder(dtype=tf.float32)
input5 = tf.placeholder(dtype=tf.float32)
output = tf.multiply(input4, input5)

add = tf.add(input1, input2)
mul = tf.multiply(add, input3)

with tf.Session() as sess:
    #Fetch:在一个会话session里同时执行多个操作operation
    print("fetch:", sess.run(fetches=[add, mul]))

    #变量定义时用占位符，可以run的时候再Feed数据，以字典形式传入
    print("feed:", sess.run(output, feed_dict={input4:[7.], input5:[8.]}))