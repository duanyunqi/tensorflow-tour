# coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward

TEST_INTERVAL_SECS = 5


def ttest(mnist):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        y_value = tf.arg_max(y, 1)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    xs = mnist.test.images

                    xs[[xs < 0.45]] = 0

                    ys = sess.run(y_value, feed_dict={x: xs})
                    accuracy_score = sess.run(correct_prediction, feed_dict={x: xs, y_: mnist.test.labels})
                    return accuracy_score, ys
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    accuracy_score, ys = ttest(mnist)
    for i in range(len(accuracy_score)):
        if(accuracy_score[i] == 0):
            print(i, ':', ys[i])


if __name__ == '__main__':
    main()