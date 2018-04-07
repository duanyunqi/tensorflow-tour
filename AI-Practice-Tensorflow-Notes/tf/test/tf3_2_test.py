import tensorflow as tf
x1 = tf.constant([[1.0, 2.0]])
x2 = tf.constant([[2.0, 3.0]])
x = x1 + x2
w = tf.constant([[3.0], [4.0]])
y=tf.matmul(x,w)
print(y)
with tf.Session() as sess:
    print(sess.run(y))


