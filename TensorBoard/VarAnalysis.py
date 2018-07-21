import tensorflow as tf

#用于对一个矩阵或向量，在TensorBoard中的scalar标签查看网络运行状态
def variable_summaries(var):
    with tf.name_scope("summaries"):
        tf.summary.scalar('mean', tf.reduce_mean(var)) #均值
        with tf.name_scope("stddev"):
            #标准差 = tf.sqrt(方差)； 方差 = (var-均值)平方的和 / N
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))
        tf.summary.scalar('stddev', stddev)  #标准差
        tf.summary.scalar('max', tf.reduce_max(var)) #最大值
        tf.summary.scalar('min', tf.reduce_min(var)) #最小值
        tf.summary.histogram('hisgram', var)  #直方图
