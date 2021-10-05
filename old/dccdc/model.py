"""Model creation."""

import tensorflow as tf


class _TemporalConv3(tf.keras.layers.Layer):

  def __init__(self, input_shape, output_multiplier=1, dilation_rate=1,
               bias=False, **kwargs):
    super(_TemporalConv3, self).__init__(**kwargs)

    self._dconv2d = tf.keras.layers.DepthwiseConv2D(
        (1, 3), strides=1, padding='same', depth_multiplier=output_multiplier,
        data_format='channels_last', dilation_rate=(1, dilation_rate),
        use_bias=bias)

    self._input_shape = input_shape
    self._output_multiplier = output_multiplier
    self._dilation_rate = dilation_rate

  def call(self, inputs, **kwargs):
    x, mask = inputs
    mask = tf.expand_dims(mask, axis=-1)
    conv_mask_sum = tf.nn.conv1d(
        mask, filters=tf.ones([3, 1, 1]), stride=1, padding='SAME',
        data_format='NWC', dilations=[1, self._dilation_rate, 1])
    conv_rescale = tf.math.divide_no_nan(3.0, conv_mask_sum)
    conv_mask = tf.cast(tf.greater(conv_mask_sum, 0.0), tf.float32)

    conv_x2 = self._dconv2d(tf.reshape(x, [-1, 1, self._input_shape[0],
                                           self._input_shape[1]]), **kwargs)
    conv_x = tf.reshape(
        conv_x2,
        [-1, self._input_shape[0],
         self._input_shape[1] * self._output_multiplier])

    return conv_x * conv_rescale, tf.squeeze(conv_mask, axis=[-1])


class _TemporalLocalPool(tf.keras.layers.Layer):

  def __init__(self, input_shape, radius, **kwargs):
    super(_TemporalLocalPool, self).__init__(**kwargs)

    self._window_l = 1 + 2 * radius
    self._input_shape = input_shape

  def call(self, inputs, **kwargs):
    x, mask = inputs

    mask = tf.expand_dims(mask, axis=-1)
    conv_mask_sum = tf.nn.conv1d(
        mask, filters=tf.ones([self._window_l, 1, 1]), stride=1, padding='SAME',
        data_format='NWC')
    conv_mask = tf.cast(tf.greater(conv_mask_sum, 0.0), tf.float32)

    x_sum = tf.nn.depthwise_conv2d(
        tf.reshape(x, [-1, 1, self._input_shape[0], self._input_shape[1]]),
        filter=tf.ones([1, self._window_l, self._input_shape[1], 1]),
        strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
    x_sum = tf.reshape(x_sum, [-1, self._input_shape[0], self._input_shape[1]])
    x_mean = tf.math.divide_no_nan(x_sum, conv_mask_sum)

    return x_mean, tf.squeeze(conv_mask, axis=[-1])


class _Dense(tf.keras.layers.Layer):

  def __init__(self, input_shape, output_channels, bias=False, **kwargs):
    super(_Dense, self).__init__(**kwargs)

    self._input_shape = input_shape
    self._output_channels = output_channels
    self._dense = tf.keras.layers.Dense(output_channels, use_bias=bias)

  def call(self, inputs, **kwargs):
    x, masks = inputs

    in_mask_channels = tf.cast(len(masks), tf.float32)
    all_masks = sum(masks)
    mask_rescale = tf.math.divide_no_nan(in_mask_channels, all_masks)
    mask = tf.cast(tf.greater(all_masks, 0.0), tf.float32)

    x = tf.reshape(x, [-1] + self._input_shape)
    dense_x = self._dense(x, **kwargs)
    dense_x = tf.reshape(dense_x * tf.expand_dims(mask_rescale, -1),
                         [-1, self._input_shape[0], self._output_channels])

    return dense_x, mask


def _swish(x):
  return tf.keras.activations.swish(x)


class _EntryFlow(tf.keras.layers.Layer):

  def __init__(self, input_shape, **kwargs):
    super(_EntryFlow, self).__init__(**kwargs)

    self._input_shape = input_shape

    self._temporal_conv_a1 = _TemporalConv3(input_shape, 4, dilation_rate=1)
    self._temporal_conv_a2 = _TemporalConv3(input_shape, 4, dilation_rate=2)
    self._temporal_conv_a3 = _TemporalConv3(input_shape, 4, dilation_rate=3)
    self._temporal_conv_a4 = _TemporalConv3(input_shape, 4, dilation_rate=4)

    self._dense1 = _Dense([input_shape[0], 16], 8)
    self._bn1 = tf.keras.layers.BatchNormalization()

    self._temporal_conv_b1 = _TemporalConv3([input_shape[0], 8], 4,
                                            dilation_rate=1)
    self._temporal_conv_b2 = _TemporalConv3([input_shape[0], 8], 4,
                                            dilation_rate=2)
    self._temporal_conv_b3 = _TemporalConv3([input_shape[0], 8], 4,
                                            dilation_rate=3)
    self._temporal_conv_b4 = _TemporalConv3([input_shape[0], 8], 4,
                                            dilation_rate=4)

    self._dense2 = _Dense([input_shape[0], 128], 16)
    self._bn2 = tf.keras.layers.BatchNormalization()

  def call(self, inputs, **kwargs):
    x, mask = inputs

    a1, a1_mask = self._temporal_conv_a1((x, mask), **kwargs)
    a2, a2_mask = self._temporal_conv_a2((x, mask), **kwargs)
    a3, a3_mask = self._temporal_conv_a3((x, mask), **kwargs)
    a4, a4_mask = self._temporal_conv_a4((x, mask), **kwargs)

    a14 = tf.concat([a1, a2, a3, a4], axis=-1)
    a14 = self._bn1(a14, **kwargs)
    a14 = _swish(a14)
    x, mask = self._dense1((a14, [a1_mask, a2_mask, a3_mask, a4_mask]),
                           **kwargs)

    b1, b1_mask = self._temporal_conv_b1((x, mask), **kwargs)
    b2, b2_mask = self._temporal_conv_b2((x, mask), **kwargs)
    b3, b3_mask = self._temporal_conv_b3((x, mask), **kwargs)
    b4, b4_mask = self._temporal_conv_b4((x, mask), **kwargs)

    b14 = tf.concat([b1, b2, b3, b4], axis=-1)
    b14 = self._bn2(b14, **kwargs)
    b14 = _swish(b14)
    x, mask = self._dense2((b14, [b1_mask, b2_mask, b3_mask, b4_mask]),
                           **kwargs)

    return x, mask


class _XConv(tf.keras.layers.Layer):

  def __init__(self, input_shape, output_channels, dilation_rate=1, **kwargs):
    super(_XConv, self).__init__(**kwargs)

    self._dense = _Dense(input_shape, output_channels)
    self._bn1 = tf.keras.layers.BatchNormalization()

    self._temporal_conv = _TemporalConv3([input_shape[0], output_channels],
                                        dilation_rate=dilation_rate)
    self._bn2 = tf.keras.layers.BatchNormalization()

  def call(self, inputs, **kwargs):
    x, mask = inputs

    x, mask = self._dense((x, [mask]), **kwargs)
    x = self._bn1(x, **kwargs)
    x = _swish(x)

    x, mask = self._temporal_conv((x, mask), **kwargs)
    x = self._bn2(x, **kwargs)
    x = _swish(x)

    return x, mask


class _XConv3(tf.keras.layers.Layer):

  def __init__(self, input_shape, output_channels, final_dilation_rate=1,
               **kwargs):
    super(_XConv3, self).__init__(**kwargs)

    self._xconv1 = _XConv(input_shape, output_channels)
    self._xconv2 = _XConv([input_shape[0], output_channels], output_channels)
    self._xconv3 = _XConv([input_shape[0], output_channels], output_channels,
                          dilation_rate=final_dilation_rate)

  def call(self, inputs, **kwargs):
    x, mask = inputs

    x, mask = self._xconv1((x, mask), **kwargs)
    x, mask = self._xconv2((x, mask), **kwargs)
    x, mask = self._xconv3((x, mask), **kwargs)

    return x, mask


class _RBBlock(tf.keras.layers.Layer):

  def __init__(self, input_shape, output_channels, transition,
               final_dilation_rate=1, **kwargs):
    super(_RBBlock, self).__init__(**kwargs)

    self._xconv3_1 = _XConv3(input_shape, output_channels, final_dilation_rate)
    self._xconv3_2 = _XConv3(input_shape, output_channels, final_dilation_rate)

    self._transition = transition

    if transition:
      self._dense = _Dense(input_shape, output_channels)
      self._bn = tf.keras.layers.BatchNormalization()

  def call(self, inputs, **kwargs):
    x, mask = inputs

    r, r_mask = self._xconv3_1((x, mask), **kwargs)
    b, b_mask = self._xconv3_2((x, mask), **kwargs)

    if self._transition:
      skip, skip_mask = self._dense((x, [mask]), **kwargs)
      skip = self._bn(skip, **kwargs)
      skip = _swish(skip)
    else:
      skip, skip_mask = x, mask

    r = r + skip
    r_invalid = tf.cast(
        tf.logical_and(tf.greater(r_mask, 0), tf.equal(skip_mask, 0)),
        tf.float32)
    r = r * tf.expand_dims((1.0 - r_invalid), axis=-1)
    b = b * tf.expand_dims(r_invalid, axis=-1)

    x = r + b
    mask = r_mask * (1.0 - r_invalid) + b_mask * r_invalid

    return x, mask


class DCCDC(tf.keras.Model):

  def __init__(self, length, **kwargs):
    super(DCCDC, self).__init__(**kwargs)
    self._length = length

    self._entry_flow = _EntryFlow([length, 1])

    self._rb1_a = _RBBlock([length, 16], 32, True, 2)
    self._rb1_b = _RBBlock([length, 32], 32, False, 2)
    self._rb1_c = _RBBlock([length, 32], 32, False, 2)

    self._rb2_a = _RBBlock([length, 32], 64, True, 2)
    self._rb2_b = _RBBlock([length, 64], 64, False, 2)
    self._rb2_c = _RBBlock([length, 64], 64, False, 2)

    self._a1 = _TemporalConv3([length, 64], dilation_rate=1)
    self._a6 = _TemporalConv3([length, 64], dilation_rate=6)
    self._a12 = _TemporalConv3([length, 64], dilation_rate=12)
    self._a18 = _TemporalConv3([length, 64], dilation_rate=18)
    self._a_pool = _TemporalLocalPool([length, 64], 30)

    self._dense1 = _Dense([length, 320], 64)
    self._bn = tf.keras.layers.BatchNormalization()

    self._xconv = _XConv([length, 64], 64)
    self._dense2 = _Dense([length, 64], 6, True)

  def call(self, inputs, **kwargs):
    x, mask = inputs
    x = tf.expand_dims(x, axis=-1)

    x, mask = self._entry_flow((x, mask), **kwargs)

    x, mask = self._rb1_a((x, mask), **kwargs)
    x, mask = self._rb1_b((x, mask), **kwargs)
    x, mask = self._rb1_c((x, mask), **kwargs)

    x, mask = self._rb2_a((x, mask), **kwargs)
    x, mask = self._rb2_b((x, mask), **kwargs)
    x, mask = self._rb2_c((x, mask), **kwargs)

    a1, a1_mask = self._a1((x, mask), **kwargs)
    a6, a6_mask = self._a6((x, mask), **kwargs)
    a12, a12_mask = self._a12((x, mask), **kwargs)
    a18, a18_mask = self._a18((x, mask), **kwargs)
    pool, pool_mask = self._a_pool((x, mask), **kwargs)

    aspp = tf.concat([a1, a6, a12, a18, pool], axis=-1)
    x, mask = self._dense1(
        (aspp, [a1_mask, a6_mask, a12_mask, a18_mask, pool_mask]), **kwargs)
    x = self._bn(x, **kwargs)
    x = _swish(x)

    x, mask = self._xconv((x, mask), **kwargs)
    x, mask = self._dense2((x, [mask]), **kwargs)

    return x, mask

  def plot_model(self, filename):
      ts = tf.keras.layers.Input((self._length,))
      mask = tf.keras.layers.Input((self._length,))
      wrapped_model = tf.keras.Model(inputs=[ts, mask],
                                     outputs=self.call((ts, mask)))
      tf.keras.utils.plot_model(
          wrapped_model, to_file=filename, dpi=220, show_shapes=True,
          show_layer_names=True, expand_nested=True)
