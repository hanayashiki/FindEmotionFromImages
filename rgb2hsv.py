from keras.layers.core import Lambda
import keras.backend as K
import tensorflow as tf
import numpy as np

rgb_bin_count = 10

def rgb2hsv_convert(x):
    x = K.cast(x, dtype=K.floatx())
    max_rgb = K.max(x, axis=-1)
    min_rgb = K.min(x, axis=-1)
    max_which = K.argmax(x, axis=-1)
    min_which = K.argmin(x, axis=-1)

    R = x[:, :, :, 0]
    B = x[:, :, :, 1]
    G = x[:, :, :, 2]

    # print(K.eval(R))
    # print(K.eval(max_which))

    H1 = (G - B) / (max_rgb - min_rgb)
    H2 = 2 + (B - R) / (max_rgb - min_rgb)
    H3 = 4 + (R - G) / (max_rgb - min_rgb)

    H = K.cast((max_which - 1) * (max_which - 2), K.floatx()) * H1 / 2 + \
        K.cast((max_which - 0) * (max_which - 2), K.floatx()) * H2 / (-1) + \
        K.cast((max_which - 0) * (max_which - 1), K.floatx()) * H3 / 2

    H = H / 6

    # print(K.eval(H))

    H = H + K.cast(H < 0, dtype=K.floatx())

    V = max_rgb / 255
    S = (max_rgb - min_rgb) / max_rgb

    HSV = K.stack([H, S, V], axis=-1)
    return HSV

def hsv_histogram(x):
    # Assumption: img is a tensor of the size [?, img_width, img_height, 3], normalized to the range [0, 1].
    img = rgb2hsv_convert(x)
    hists = []
    with tf.variable_scope('color_hist_producer') as scope:
        bin_size = 1 / rgb_bin_count
        hist_entries = []
        # Split image into single channels
        img_r, img_g, img_b = img[:, :, :, 0], img[:, :, :, 1], img[:, :, :, 2]
        for img_chan in [img_r, img_g, img_b]:
            for idx, i in enumerate(np.arange(0, 1, bin_size)):
                gt = tf.greater(img_chan, i)
                leq = tf.less_equal(img_chan, i + bin_size)
                # Put together with logical_and, cast to float and sum up entries -> gives count for current bin.
                hist_entries.append(tf.reduce_sum(tf.cast(tf.logical_and(gt, leq), tf.float32), [1, 2]))

        # Pack scalars together to a tensor, then normalize histogram.
        hist = tf.nn.l2_normalize(tf.stack(hist_entries, axis=1), 1)
        return hist

def shape_histogram(input_shape):
    print(input_shape)
    print((input_shape[0], 3 * rgb_bin_count))
    return (input_shape[0], 3 * rgb_bin_count)

Histogram = Lambda(hsv_histogram, output_shape=shape_histogram, name="histogram")

if __name__ == '__main__':
    x = \
        [
            [[0, 255, 0], [255, 0, 0]],
            [[255, 0, 0], [255, 0, 0]],
        ]

    print(K.eval(rgb2hsv_convert(x)))

    print(K.eval(hsv_histogram(x)))