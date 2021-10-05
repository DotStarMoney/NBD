import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

from absl import app

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import dataset


def to_interior_mask(affinity):
  return tf.cast(
      tf.reduce_all(tf.greater_equal(affinity, 0.5), axis=-1),
      tf.float32)


def label_breaks(affinity):
  affinity = tf.expand_dims(affinity, 0)
  interior_mask = to_interior_mask(affinity)
  interior_mask = tf.expand_dims(interior_mask, axis=-1)
  padded_interior_mask = tf.pad(interior_mask, [[0, 0], [1, 0], [0, 0]])
  filters = tf.reshape(tf.constant([-1, 1], tf.float32), [2, 1, 1])
  interior_mask_edges = tf.cast(
      tf.greater(tf.nn.conv1d(padded_interior_mask, filters, 1, 'SAME',
                              'NWC'),
                 0.9),
      tf.float32)
  labeled_breaks = (tf.math.cumsum(interior_mask_edges, axis=1)
                    * padded_interior_mask)[:, 1:, 0]
  return tf.cast(labeled_breaks, tf.float32)


def plot_ts(examples, affinities_pred):
  examples_n = examples[0].shape[0]
  for i in range(examples_n):
    ts = examples[0][i]
    mask = examples[1][i]
    affinity_true = examples[2][i]
    affinity_pred = affinities_pred[i]
    mask = tf.cast(mask, tf.bool).numpy()
    x = np.array(range(ts.shape[0]))
    x_masked = x[mask]
    ts_y = ts[mask]

    labels_true = tf.squeeze(label_breaks(affinity_true)).numpy()
    num_segs_true = np.max(labels_true)
    labels_true /= num_segs_true
    labels_true_colors = cm.winter(labels_true)

    labels_pred = tf.squeeze(label_breaks(affinity_pred)).numpy()
    num_segs_pred = np.max(labels_pred)
    labels_pred /= num_segs_pred
    labels_pred_colors = cm.rainbow(labels_pred)

    labels_one_true = (labels_true > 0).astype(float) * 0.5
    labels_one_pred = (labels_pred > 0).astype(float) * 0.5

    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'Georgia'

    plt.figure(figsize=(10, 3))
    plt.ylim(0, 1)
    plt.xlim(0, 255)
    barlist0 = plt.bar(x, labels_one_true, width=1.0, bottom=0)
    barlist1 = plt.bar(x, -labels_one_pred, width=1.0, bottom=1)
    for i in range(255):
      barlist0[i].set_color('dimgrey')#labels_true_colors[i])
      barlist1[i].set_color(labels_pred_colors[i])

    plt.axhline(0.5, c='k', lw=5, zorder=1)
    plt.scatter(x_masked, ts_y, c='white', edgecolors='k', zorder=2)

    plt.xlabel('Time (days)')
    plt.ylabel('NDVI (tbsp)')

    plt.tight_layout()
    plt.show()


BATCH_N = 256


def main(argv):
  validation_ds = dataset.create_dataset('data/ccdc_validation.tfrecord.gz',
                                         BATCH_N, True)

  dccdc = tf.keras.models.load_model('models/20210525-220224')

  print(dccdc.summary())

  for example in validation_ds.skip(1).take(1):
    examples = example

  affinities_pred, masks_pred = dccdc((examples[0], examples[1]))

  start_time = time.time()
  print('start')
  affinities_pred, masks_pred = dccdc((tf.ones([256 * 256, 256]), tf.ones([256 * 256, 256])))
  print('end')
  print("--- %s seconds ---" % (time.time() - start_time))

  plot_ts(examples, tf.sigmoid(affinities_pred))


if __name__ == '__main__':
  app.run(main)
