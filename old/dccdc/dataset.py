"""Dataset processing."""

import tensorflow as tf

LENGTH_N = 256
AFFINITY_N = 6

_PARSING_DESCRIPTORS = {
  'ts': tf.io.FixedLenFeature([LENGTH_N], tf.float32),
  'mask': tf.io.FixedLenFeature([], tf.string),
  'affinity': tf.io.FixedLenFeature([], tf.string)}


def create_dataset(glob_path, batch_n, validate=False):
  raw_ds = tf.data.TFRecordDataset(
    tf.io.gfile.glob(glob_path),
    compression_type='GZIP',
    num_parallel_reads=2)

  @tf.function
  def parse_fn(serialized_example):
    result_dict = tf.io.parse_single_example(serialized_example, _PARSING_DESCRIPTORS)
    ts = result_dict['ts']
    mask = tf.cast(tf.io.decode_raw(result_dict['mask'], tf.uint8), tf.float32)
    affinity = tf.reshape(
      tf.cast(tf.io.decode_raw(result_dict['affinity'], tf.uint8),
              tf.float32),
      [LENGTH_N, AFFINITY_N])

    if not validate:
      ts, mask, affinity = tf.cond(
        tf.random.uniform([]) >= 0.5,
        lambda: (tf.reverse(ts, [0]), tf.reverse(mask, [0]), tf.reverse(affinity, [0, 1])),
        lambda: (ts, mask, affinity))

      ts = tf.cond(tf.random.uniform([]) >= 0.5, lambda: 1.0 - ts, lambda: ts)
      ts = ts * mask

    return ts, mask, affinity

  @tf.function
  def postbatch_fn(ts, mask, affinity):
    internal_members = tf.reduce_prod(affinity, axis=-1)
    internal_members_count = tf.reduce_sum(internal_members, axis=-1)

    internal_weight = tf.expand_dims(
        tf.math.divide_no_nan(0.5, internal_members_count), axis=-1)
    external_weight = tf.expand_dims(
        tf.math.divide_no_nan(0.5, LENGTH_N - internal_members_count), axis=-1)

    weights = (
        (internal_weight - external_weight) * internal_members
        + external_weight)

    affinity = 0.1 + affinity * 0.8

    return ts, mask, affinity, weights

  @tf.function
  def filter_fn(ts, mask, affinity):
    return tf.math.logical_not(tf.reduce_any(tf.math.is_nan(ts)))

  ds = raw_ds.map(parse_fn, num_parallel_calls=2).filter(filter_fn)
  if not validate:
    ds = ds.repeat(-1).shuffle(2000)
  ds = ds.batch(batch_n, drop_remainder=True)

  return ds.map(postbatch_fn, num_parallel_calls=2)
