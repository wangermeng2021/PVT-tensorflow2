
import tensorflow as tf
def concat_max_average_pool(pool_size=(2, 2),name="last_conv"):
    def layer_func(x):
        x1 = tf.keras.layers.MaxPooling2D(pool_size)(x)
        x2 = tf.keras.layers.AveragePooling2D(pool_size)(x)
        return tf.concat([x1, x2], axis=-1)
    return tf.keras.layers.Lambda(layer_func, name=name)
def rename_layer(name="last_conv"):
    def layer_func(x):
        return x
    return tf.keras.layers.Lambda(layer_func, name=name)