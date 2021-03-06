import os
import random

from keras.layers import Embedding, Flatten, BatchNormalization, Activation, Lambda

from ml2lm.calc.model.units.dropouts import *
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


def make_cutoff(off_val=1e-4, cut_norm_func=None, keep_cut_amp=False, keep_drop_amp=True, cut_in_test=False,
                epsilon=1e-6):
    def _cutoff(dr):
        return Cutoff(off_val=off_val, dist_rate=dr, cut_norm_func=cut_norm_func, keep_cut_amp=keep_cut_amp,
                      keep_drop_amp=keep_drop_amp, cut_in_test=cut_in_test, epsilon=epsilon)

    return _cutoff


def get_default_cutoff(dr):
    return make_cutoff(off_val=1e-4, cut_norm_func=None, keep_cut_amp=False, keep_drop_amp=True, cut_in_test=False,
                       epsilon=1e-6)(dr)


def make_segdropout(anneal=0.1, agg_method='mean', smooth_rate=0., noise_type='gaussian', keep_amp_type='abs',
                    epsilon=1e-6):
    def _seg_dropout(dr):
        return SegDropout(dr, anneal=anneal, agg_method=agg_method, smooth_rate=smooth_rate, noise_type=noise_type,
                          keep_amp_type=keep_amp_type, epsilon=epsilon)

    return _seg_dropout


def get_default_segdropout(dr):
    return make_segdropout(anneal=0.1, agg_method='mean', noise_type='gaussian')(dr)


def add_dense(x, units, bn=True, activation=seu, dropout=0.2, dropout_handler=Dropout):
    x = Dense(units)(x)
    if bn:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    if dropout > 0:
        x = dropout_handler(dropout)(x)
    return x


def shrink(dim, shrink_factor):
    if dim > 10:
        return max(10, int(dim * shrink_factor))
    return dim


def get_embeds(cat_input, cat_in_dims, cat_out_dims, shrink_factor=1.0):
    import keras

    embeds = []
    for i, in_dim in enumerate(cat_in_dims):
        embed = cat_input[:, i, None] if keras.__version__ >= '2.4.0' else Lambda(lambda cats: cats[:, i, None])(
            cat_input)
        embed = Embedding(in_dim, shrink(cat_out_dims[i], shrink_factor))(embed)
        embeds.append(Flatten()(embed))
    return embeds


def get_segments(seg_input, seg_out_dims, shrink_factor=1.0, seg_type=0, seg_func=seu, seg_input_val_range=(0, 1),
                 seg_bin=False, only_bin=False, scale_n=0, scope_type='global', bundle_scale=False):
    import keras

    segments = []
    for i, out_dim in enumerate(seg_out_dims):
        segment = seg_input[:, i, None] if keras.__version__ >= '2.4.0' else Lambda(lambda segs: segs[:, i, None])(
            seg_input)
        if scale_n > 0 or bundle_scale:
            segment = WaveletWrapper(shrink(out_dim, shrink_factor), input_val_range=seg_input_val_range,
                                     seg_func=seg_func, include_seg_bin=seg_bin, only_seg_bin=only_bin,
                                     seg_type=seg_type, scale_n=scale_n, scope_type=scope_type,
                                     bundle_scale=bundle_scale)(segment)
        else:
            if not seg_type:
                segment = SegTriangleLayer(shrink(out_dim, shrink_factor), input_val_range=seg_input_val_range,
                                           seg_func=seg_func, include_seg_bin=seg_bin, only_seg_bin=only_bin)(segment)
            else:
                segment = SegRightAngleLayer(shrink(out_dim, shrink_factor), input_val_range=seg_input_val_range,
                                             seg_func=seg_func, include_seg_bin=seg_bin, only_seg_bin=only_bin)(segment)
        segments.append(segment)
    return segments
