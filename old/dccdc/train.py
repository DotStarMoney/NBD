import datetime
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from absl import app
from absl import flags

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import dataset
import loss
import metrics
import model
import optimizer

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path_prefix', 'models',
                    'Local path to saved models.')


def plot_ts(ts, mask):
  mask = tf.cast(mask, tf.bool).numpy()
  x = np.array(range(ts.shape[0]))[mask]
  y = ts[mask]

  with plt.xkcd():
    plt.figure(figsize=(10, 3))
    plt.scatter(x, y)
    plt.show()


def _create_multistep_fn(model, optimizer, ds_iter, metrics, steps_n):
  @tf.function
  def multistep_fn():
    loss_acc = 0.0
    for _ in tf.range(steps_n):
      ts, mask, affinity, weights = next(ds_iter)

      with tf.GradientTape() as tape:
        y_logits, y_mask = model((ts, mask), training=True)
        constraint = loss.affinity_log_loss(affinity, y_logits, weights)

      grads = tape.gradient(constraint, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      loss_acc += constraint

      metrics.update_state(affinity, tf.sigmoid(y_logits))

    return loss_acc / float(steps_n)

  return multistep_fn


# 4 epochs
STEPS_N = int(8e6)
TS_LENGTH = 256
BATCH_N = 128
MULTISTEP_N = 4


def main(argv):
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

  train_ds = dataset.create_dataset('data/ccdc_train_*.tfrecord.gz', BATCH_N)
  validation_ds = dataset.create_dataset('data/ccdc_validation.tfrecord.gz',
                                         BATCH_N)

  train_iterations = math.floor(STEPS_N / BATCH_N / MULTISTEP_N)

  dccdc = model.DCCDC(TS_LENGTH)
  opt = optimizer.ranger_with_exp_decay(0.1, 300, 0.005,
                                        train_iterations * MULTISTEP_N)

  break_stats = metrics.BreakStats(BATCH_N, TS_LENGTH, max_breaks=64)
  ds_iter = iter(train_ds)

  step_fn = _create_multistep_fn(dccdc, opt, ds_iter, break_stats, MULTISTEP_N)

  print(f'Training for {train_iterations} iterations.')
  for iteration in range(train_iterations):
    if iteration % 10 == 0:
      print(break_stats.result())
      print('Reset...')
      break_stats.reset_state()

    loss_value = step_fn()
    step = opt.iterations
    print('Step: {}'.format(step))
    print(loss_value, opt.learning_rate(step))

  print('Training complete!')
  model_string_ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  model_name = FLAGS.model_path_prefix + '\\' + model_string_ts
  dccdc.save(model_name, include_optimizer=False)


if __name__ == '__main__':
  app.run(main)
