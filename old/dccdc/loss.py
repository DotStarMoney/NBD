"""Loss/constraint functions."""

import tensorflow as tf


def affinity_log_loss(y_true_affinity, y_pred_affinity_logits, weights):
  log_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true_affinity,
                                                     y_pred_affinity_logits)
  avg_position_log_loss = tf.reduce_mean(log_loss, axis=-1)
  example_loss = tf.reduce_sum(weights * avg_position_log_loss, axis=-1)
  return tf.reduce_mean(example_loss)
