
"""
DropBlock impl from https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74
"""
import tensorflow as tf
def droppath(x,drop_prob=0.1):
    keep_prob = 1.-drop_prob
    batch_size = tf.concat([tf.shape(x)[0:1],tf.tile([1],[tf.size(tf.shape(x)[1:])])],axis=-1)
    enable_drop = tf.math.floor(tf.random.uniform(batch_size) + keep_prob)
    return x/keep_prob*enable_drop
