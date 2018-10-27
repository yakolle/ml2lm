import inspect
import os
import random

import keras.backend as bk
import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Flatten, Dense, BatchNormalization, Activation, Dropout, Lambda

from ml2lm.calc.model.units.SegRightAngleLayer import SegRightAngleLayer
from ml2lm.calc.model.units.SegTriangleLayer import SegTriangleLayer
from ml2lm.calc.model.units.activations import seu


def init_tensorflow():
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1)
    bk.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))


def set_seed(_seed=10000):
    os.environ['PYTHONHASHSEED'] = str(_seed + 6)
    np.random.seed(_seed + 7)
    random.seed(_seed + 8)
    tf.set_random_seed(_seed + 9)


def read_weights(model, weights_path):
    model.load_weights(weights_path)
    return model


def get_out_dim(vocab_size, scale=10, shrink_factor=0.5, max_out_dim=None):
    if vocab_size <= 10:
        out_dim = max(2, vocab_size)
    elif vocab_size <= 40:
        out_dim = max(10, int(shrink_factor * vocab_size // 2))
    else:
        out_dim = max(10, int(shrink_factor * 20), int(shrink_factor * vocab_size / np.log2(vocab_size / scale)))
    out_dim = out_dim if max_out_dim is None else max_out_dim
    return out_dim


def get_seg_num(val_cnt, shrink_factor=0.5, max_seg_dim=None):
    seg_dim = max(2, int(np.sqrt(val_cnt * shrink_factor)))

    seg_dim = seg_dim if max_seg_dim is None else max_seg_dim
    return seg_dim


def get_seg_num_by_value(x, precision=4, shrink_factor=0.5):
    val_mean = np.mean(np.abs(x))
    cur_precision = np.round(np.log10(val_mean))
    x = (x * 10 ** (precision - cur_precision)).astype(np.int64)
    val_cnt = len(np.unique(x))
    return get_seg_num(val_cnt, shrink_factor=shrink_factor)


def to_tnn_data(x, oh_indices=None, cat_indices=None, seg_indices=None, num_indices=None):
    nn_data = {}
    if oh_indices is not None:
        nn_data['ohs'] = x[:, oh_indices]
    if cat_indices is not None:
        cat_indices['cats'] = x[:, cat_indices]
    if seg_indices is not None:
        seg_indices['segs'] = x[:, seg_indices]
    if num_indices is not None:
        num_indices['nums'] = x[:, num_indices]
    return nn_data


def add_dense(x, units, bn=True, activation=seu, dropout=0.2):
    x = Dense(units)(x)
    if bn:
        x = BatchNormalization()(x)
    x = activation()(x) if inspect.isclass(activation) else Activation(activation)(x)
    x = Dropout(dropout)(x)
    return x


def shrink(dim, shrink_factor):
    if dim > 10:
        return max(10, int(dim * shrink_factor))
    return dim


def get_embeds(cat_input, cat_in_dims, cat_out_dims, shrink_factor=1.0):
    embeds = []
    for i, in_dim in enumerate(cat_in_dims):
        embed = Lambda(lambda cats: cats[:, i, None])(cat_input)
        embed = Embedding(in_dim, shrink(cat_out_dims[i], shrink_factor))(embed)
        embeds.append(Flatten()(embed))
    return embeds


def get_segments(seg_input, seg_out_dims, shrink_factor=1.0, seg_type=0, seg_func=seu, seg_input_val_range=(0, 1)):
    segments = []
    for i, out_dim in enumerate(seg_out_dims):
        segment = Lambda(lambda segs: segs[:, i, None])(seg_input)
        if not seg_type:
            segment = SegTriangleLayer(shrink(out_dim, shrink_factor), input_val_range=seg_input_val_range,
                                       seg_func=seg_func)(segment)
        else:
            segment = SegRightAngleLayer(shrink(out_dim, shrink_factor), input_val_range=seg_input_val_range,
                                         seg_func=seg_func)(segment)
        segments.append(segment)
    return segments
