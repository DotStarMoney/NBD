"""Optimization utilities."""

import tensorflow as tf
import tensorflow_addons as tfa


class _ExponentialDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, warmup_steps, total_steps, init_lr, total_decay):
    self._warmup_steps = float(warmup_steps)
    self._init_lr = init_lr
    self._total_decay = total_decay
    self._total_steps = total_steps

  def __call__(self, step):
    step = tf.cast(step, tf.float32)
    return self._init_lr * tf.cond(
        step < self._warmup_steps,
        lambda: step / self._warmup_steps,
        lambda: self._total_decay ** ((step - self._warmup_steps) /
                                      (self._total_steps - self._warmup_steps)))


def adam_with_exp_decay(learning_rate, decay_rate, training_steps):
  lr_fn = tf.keras.optimizers.schedules.ExponentialDecay(
      learning_rate, training_steps, decay_rate)
  return tf.keras.optimizers.Adam(learning_rate=lr_fn)


def ranger_with_exp_decay(learning_rate, warmup_steps, decay_rate,
                          training_steps):
  opt = tfa.optimizers.RectifiedAdam(
      learning_rate=_ExponentialDecayWithWarmup(
          warmup_steps, training_steps, learning_rate, decay_rate))
  return tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)
