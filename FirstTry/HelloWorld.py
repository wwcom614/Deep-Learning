# -*-  coding: UTF-8 -*-

import tensorflow as tf

#创建一个常量 Operation操作
hw = tf.constant("Hello World!")

# 启动一个TensorFlow的Session会话
sess1 = tf.Session()

#运行 Graph计算图
print(sess1.run(hw))

#关闭Session会话
sess1.close()
################################################

#另外一种更好更安全的创建和关闭Session的方法，推荐使用Python的with管理Session上下文
with tf.Session() as sess2:

    const1 = tf.constant([[1, 2]])
    const2 = tf.constant([[3],
                          [4]])
    #矩阵点乘
    multiple = tf.matmul(const1, const2)
    print("multiple的Tensor表示法:", multiple)
    #multiple的Tensor表示法: Tensor("MatMul:0", shape=(1, 1), dtype=int32)

    print("multiple:",sess2.run(multiple))
    #multiple: [[11]]

    if const1.graph is tf.get_default_graph():
        print("const1所在的图是当前上下文默认的图")
        #const1所在的图是当前上下文默认的图

    if const2.graph is tf.get_default_graph():
        print("const2所在的图是当前上下文默认的图")
        #const2所在的图是当前上下文默认的图

    if multiple.graph is tf.get_default_graph():
        print("multiple所在的图是当前上下文默认的图")
        #multiple所在的图是当前上下文默认的图


################################################
#变量自增尝试
current = tf.Variable(0, name='counter')
update = tf.assign(current, current + 1)

with tf.Session() as sess:
    #重要！！变量使用前需要初始化
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(5):
        sess.run(update)
        print(sess.run(current))
