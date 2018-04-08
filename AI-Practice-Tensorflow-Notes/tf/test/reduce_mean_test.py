import tensorflow as tf

x1 = [[1, 2],
      [3, 4]]
x2 = [[1., 2.],
     [3., 4.]]
x_3d = [[[1., 2., 3.], [2., 3., 4.]],[[3., 4., 5.], [2., 3., 4.]],
        [[1., 2., 3.], [2., 3., 4.]],[[1., 2., 3.], [2., 3., 4.]]]
rm1 = tf.reduce_mean(x1)
rm2 = tf.reduce_mean(x2)
rm3 = tf.reduce_mean(x2, 0) # 按列求平均值
rm4 = tf.reduce_mean(x2, 1) # 按行求平均值
rm5 = tf.reduce_mean(x_3d)

with tf.Session() as sess:
    print(sess.run(rm1))
    print(sess.run(rm2))
    print(sess.run(rm3))
    print(sess.run(rm4))
    print(sess.run(rm5))