
import numpy as np
import tensorflow as tf
def normalize(x, mode='caffe'):
    x = x.astype(np.float32)
    # x = tf.cast(x,tf.dtypes.float32)
    if mode == 'tf':
        # x /= 127.5
        # x -= 1.
        # return x
        return x/255.
    elif mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x -= mean
        x /= std
        return x
    elif mode == 'caffe':
        x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        x -= mean
        return x
    return x
