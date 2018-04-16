import tensorflow as tf

ckpt = tf.train.get_checkpoint_state('../model/')
ckpt_path = ckpt.model_checkpoint_path
print(ckpt)
print(ckpt_path)
