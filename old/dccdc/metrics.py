"""Model performance metrics."""

import numpy as np
import tensorflow as tf

_EFFECTIVE_ONE = 0.99


class BreakStats(tf.keras.layers.Layer):

  def __init__(self, batch_n, depth, max_breaks=16, affinity_cutoff=0.5,
               name='break_stats', **kwargs):
    super(BreakStats, self).__init__(name=name, **kwargs)

    self._affinity_cutoff = affinity_cutoff
    self._max_breaks = max_breaks

    self._n_indices = tf.cast(tf.range(batch_n * depth) / depth, tf.int32)
    self._n_indices = tf.expand_dims(self._n_indices, -1)

    self._time_range = tf.cast(tf.range(tf.shape(self._n_indices)[0]), tf.int32)
    self._time_range = self._time_range % depth

    self._agg_shape = [batch_n, max_breaks + 1]
    self._break_position_indices = tf.tile(
        tf.reshape(tf.range(self._max_breaks), [1, -1]),
        [self._agg_shape[0], 1])

    self._num_breaks_absolute_error_n = self.add_weight(
        'num_breaks_absolute_error_n', dtype=tf.float64, initializer='zeros')
    self._min_break_covering_radius_n = self.add_weight(
        'min_break_covering_radius_n', dtype=tf.float64, initializer='zeros')

    self._num_breaks_absolute_error_acc = self.add_weight(
        'num_breaks_absolute_error_acc', dtype=tf.float64, initializer='zeros')
    self._min_break_covering_radius_acc = self.add_weight(
        'min_break_covering_radius_acc', dtype=tf.float64, initializer='zeros')

  def _to_interior_mask(self, affinity):
    return tf.cast(
        tf.reduce_all(
            tf.greater_equal(affinity, self._affinity_cutoff), axis=-1),
        tf.float32)

  def _label_breaks(self, interior_mask):
    interior_mask = tf.expand_dims(interior_mask, axis=-1)
    interior_mask = 1.0 - interior_mask
    padded_interior_mask = tf.pad(interior_mask, [[0, 0], [1, 0], [0, 0]])
    filters = tf.reshape(tf.constant([-1, 1], tf.float32), [2, 1, 1])
    interior_mask_edges = tf.cast(
        tf.greater(tf.nn.conv1d(padded_interior_mask, filters, 1, 'SAME',
                                'NWC'),
                   _EFFECTIVE_ONE),
        tf.float32)
    labeled_breaks = (tf.math.cumsum(interior_mask_edges, axis=1)
                      * padded_interior_mask)[:, 1:, 0]
    return tf.cast(
        tf.where(labeled_breaks >= self._max_breaks, 0.0, labeled_breaks),
        tf.int32)

  def _break_counts_and_positions(self, affinity_mask):
    interior = self._to_interior_mask(affinity_mask)
    labeled_breaks = self._label_breaks(interior)
    labeled_breaks_indices = tf.reshape(labeled_breaks,
                                        [tf.shape(self._n_indices)[0], 1])
    indices = tf.concat([self._n_indices, labeled_breaks_indices], axis=-1)

    acc_seg_positions = tf.scatter_nd(indices, self._time_range,
                                      self._agg_shape)[:, 1:]
    seg_counts = tf.scatter_nd(indices, tf.ones_like(self._time_range),
                               self._agg_shape)[:, 1:]

    break_positions = tf.math.divide_no_nan(
        tf.cast(acc_seg_positions, tf.float64),
        tf.cast(seg_counts, tf.float64))
    num_breaks = tf.reduce_max(labeled_breaks, axis=1, keepdims=True)

    break_positions = tf.where(
        self._break_position_indices >= num_breaks,
        tf.cast(np.nan, tf.float64),
        break_positions)

    return tf.cast(num_breaks, tf.float64), break_positions

  def reset_state(self):
    self._num_breaks_absolute_error_n.assign(0)
    self._num_breaks_absolute_error_acc.assign(0)
    self._min_break_covering_radius_n.assign(0)
    self._min_break_covering_radius_acc.assign(0)

  def result(self):
    return {
        'mae_num_breaks': tf.math.divide_no_nan(
            self._num_breaks_absolute_error_acc,
            self._num_breaks_absolute_error_n),
        'mean_min_covering_radius': tf.math.divide_no_nan(
            self._min_break_covering_radius_acc,
            self._min_break_covering_radius_n)}

  def update_state(self, y_true_affinity, y_pred_affinity):
    true_counts, true_positions = self._break_counts_and_positions(
        y_true_affinity)
    pred_counts, pred_positions = self._break_counts_and_positions(
        y_pred_affinity)

    self._num_breaks_absolute_error_n.assign_add(
        tf.cast(tf.shape(y_true_affinity)[0], dtype=tf.float64))
    self._num_breaks_absolute_error_acc.assign_add(
        tf.reduce_sum(tf.abs(true_counts - pred_counts)))

    true_positions_flat_tiled = tf.tile(true_positions, [1, self._max_breaks])
    pred_positions_flat_tiled = tf.reshape(
        tf.tile(tf.expand_dims(pred_positions, -1), [1, 1, self._max_breaks]),
        [-1, self._max_breaks ** 2])

    abs_diff_positions = tf.abs(
        true_positions_flat_tiled - pred_positions_flat_tiled)
    abs_diff_positions = tf.reshape(abs_diff_positions,
                                    [-1, self._max_breaks, self._max_breaks])
    abs_diff_positions = tf.where(tf.math.is_nan(abs_diff_positions),
                                  tf.cast(np.inf, tf.float64),
                                  abs_diff_positions)

    closest_pred_to_true_breaks = tf.reduce_min(abs_diff_positions, axis=-1)
    closest_pred_to_true_breaks = tf.where(
        tf.math.is_inf(closest_pred_to_true_breaks),
        tf.cast(-np.inf, tf.float64),
        closest_pred_to_true_breaks)
    min_break_covering_radius = tf.reduce_max(closest_pred_to_true_breaks,
                                              axis=-1)

    non_inf = min_break_covering_radius != tf.cast(-np.inf, tf.float64)
    min_break_covering_radius = tf.where(non_inf,
                                         min_break_covering_radius,
                                         tf.cast(0.0, tf.float64))

    self._min_break_covering_radius_n.assign_add(
        tf.reduce_sum(tf.cast(non_inf, tf.float64)))
    self._min_break_covering_radius_acc.assign_add(
        tf.reduce_sum(min_break_covering_radius))
