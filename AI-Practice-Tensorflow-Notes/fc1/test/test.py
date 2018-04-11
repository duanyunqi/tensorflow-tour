from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/', one_hot=True)
print("train data size:", mnist.train.num_examples)
print("validation data size:", mnist.validation.num_examples)
print('test data size:', mnist.test.num_examples)

print(mnist.train.labels[0])
print(mnist.train.images[0])
print(mnist.train.images[0].shape)