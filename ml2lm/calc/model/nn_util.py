import os
import random

from keras.layers import Dropout

from ml2lm.calc.model.units.rel import *
from ml2lm.calc.model.units.seg import *


def init_tensorflow():
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1)
    bk.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))


def set_seed(_seed=10000):
    os.environ['PYTHONHASHSEED'] = str(_seed + 6)
    np.random.seed(_seed + 7)
    random.seed(_seed + 8)
    try:
        tf.random.set_seed(_seed + 9)
    except Exception as e:
        print(e)
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
    out_dim = max_out_dim if max_out_dim is not None and out_dim > max_out_dim else out_dim
    return out_dim


def get_seg_num(val_cnt, multi_factor=1.0, max_seg_dim=None):
    seg_dim = max(2, int(np.sqrt(val_cnt * multi_factor)))

    seg_dim = max_seg_dim if max_seg_dim is not None and seg_dim > max_seg_dim else seg_dim
    return seg_dim


def calc_val_cnt(x, precision=4):
    val_mean = np.mean(np.abs(x[x != 0]))
    cur_precision = np.round(np.log10(val_mean))
    x = (x * 10 ** (precision - cur_precision)).astype(np.int64)
    val_cnt = len(np.unique(x))
    return val_cnt


def get_seg_num_by_value(x, precision=4, multi_factor=1.0):
    val_cnt = calc_val_cnt(x, precision)
    return get_seg_num(val_cnt, multi_factor=multi_factor)


def to_tnn_data(x, cat_indices=None, seg_indices=None, num_indices=None):
    nn_data = {}
    if cat_indices is not None:
        cat_indices['cats'] = x[:, cat_indices]
    if seg_indices is not None:
        seg_indices['segs'] = x[:, seg_indices]
    if num_indices is not None:
        num_indices['nums'] = x[:, num_indices]
    return nn_data


def shrink(dim, shrink_factor):
    if dim > 10:
        return max(10, int(dim * shrink_factor))
    return dim


def get_default_rel_conf():
    return [{'rel_id': 'fm', 'dropout': 0.4, 'dropout_handler': Dropout,
             'conf': {'factor_rank': 320, 'dist_func': lrelu, 'rel_types': 'd', 'exclude_selves': (False,)}},
            {'rel_id': 'br', 'dropout': 0.8, 'dropout_handler': Dropout,
             'conf': {'factor_rank': 320, 'trans_func': lrelu, 'op_func': rel_mul, 'rel_types': 'd'}}]
