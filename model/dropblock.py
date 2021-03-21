
"""
DropBlock impl from https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py#L74
"""
import tensorflow as tf
def dropblock(net, keep_prob, dropblock_size,data_format='channels_last'):
  """DropBlock: a regularization method for convolutional neural networks.
  DropBlock is a form of structured dropout, where units in a contiguous
  region of a feature map are dropped together. DropBlock works better than
  dropout on convolutional layers due to the fact that activation units in
  convolutional layers are spatially correlated.
  See https://arxiv.org/pdf/1810.12890.pdf for details.
  Args:
    net: `Tensor` input tensor.
    keep_prob: `float` or `Tensor` keep_prob parameter of DropBlock. "None"
        means no DropBlock.
    dropblock_size: `int` size of blocks to be dropped by DropBlock.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
  Returns:
      A version of input tensor with DropBlock applied.
  Raises:
      if width and height of the input tensor are not equal.
  """

  if keep_prob is None:
    return net

  tf.logging.info('Applying DropBlock: dropblock_size {}, net.shape {}'.format(
      dropblock_size, net.shape))

  if data_format == 'channels_last':
    _, width, height, _ = net.get_shape().as_list()
  else:
    _, _, width, height = net.get_shape().as_list()
  if width != height:
    raise ValueError('Input tensor with width!=height is not supported.')

  dropblock_size = min(dropblock_size, width)
  # seed_drop_rate is the gamma parameter of DropBlcok.
  seed_drop_rate = (1.0 - keep_prob) * width**2 / dropblock_size**2 / (
      width - dropblock_size + 1)**2

  # Forces the block to be inside the feature map.
  w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
  valid_block_center = tf.logical_and(
      tf.logical_and(w_i >= int(dropblock_size // 2),
                     w_i < width - (dropblock_size - 1) // 2),
      tf.logical_and(h_i >= int(dropblock_size // 2),
                     h_i < width - (dropblock_size - 1) // 2))

  valid_block_center = tf.expand_dims(valid_block_center, 0)
  valid_block_center = tf.expand_dims(
      valid_block_center, -1 if data_format == 'channels_last' else 0)

  randnoise = tf.random_uniform(net.shape, dtype=tf.float32)
  block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
      (1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
  block_pattern = tf.cast(block_pattern, dtype=tf.float32)

  if dropblock_size == width:
    block_pattern = tf.reduce_min(
        block_pattern,
        axis=[1, 2] if data_format == 'channels_last' else [2, 3],
        keepdims=True)
  else:
    if data_format == 'channels_last':
      ksize = [1, dropblock_size, dropblock_size, 1]
    else:
      ksize = [1, 1, dropblock_size, dropblock_size]
    block_pattern = -tf.nn.max_pool(
        -block_pattern, ksize=ksize, strides=[1, 1, 1, 1], padding='SAME',
        data_format='NHWC' if data_format == 'channels_last' else 'NCHW')

  percent_ones = tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / tf.cast(
      tf.size(block_pattern), tf.float32)

  net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
      block_pattern, net.dtype)
  return net