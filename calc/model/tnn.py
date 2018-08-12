from keras import Input, Model
from keras.layers import concatenate
from keras.optimizers import Adam

from calc.model.nn_util import *
from calc.model.units.FMLayer import FMLayer


def get_default_mse_output(flat, oh_input=None, cat_input=None, seg_input=None, num_input=None):
    flat = add_dense(flat, 64, bn=False, dropout=0.05)
    output = Dense(1)(flat)

    inputs = [oh_input] if oh_input is not None else []
    if cat_input is not None:
        inputs.append(cat_input)
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)

    dnn = Model(inputs, output)
    dnn.compile(loss='mse', optimizer=Adam(lr=1e-3))
    return dnn


def get_default_bce_output(flat, oh_input=None, cat_input=None, seg_input=None, num_input=None):
    flat = add_dense(flat, 64, bn=False, dropout=0.05)
    output = Dense(1, activation='sigmoid')(flat)

    inputs = [oh_input] if oh_input is not None else []
    if cat_input is not None:
        inputs.append(cat_input)
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)

    dnn = Model(inputs, output)
    dnn.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['acc'])
    return dnn


def get_tnn_model(x, get_output=get_default_mse_output, cat_in_dims=None, cat_out_dims=None, seg_out_dims=None,
                  num_segs=None, use_fm=False, seg_flag=True, add_seg_src=True, seg_num_flag=True,
                  get_extra_layers=None, fm_dim=320, fm_dropout=0.2, fm_activation=None, hidden_units=320,
                  hidden_dropout=0.2):
    oh_input = Input(shape=[x['ohs'].shape[1]], name='ohs') if 'ohs' in x else None
    cat_input = Input(shape=[x['cats'].shape[1]], name='cats') if 'cats' in x else None
    seg_input = Input(shape=[x['segs'].shape[1]], name='segs') if 'segs' in x else None
    num_input = Input(shape=[x['nums'].shape[1]], name='nums') if 'nums' in x else None

    embeds = [Flatten()(Embedding(3, 2)(oh_input))] if oh_input is not None else []
    if cat_input is not None:
        embeds += get_embeds(cat_input, cat_in_dims, cat_out_dims)
    embeds = Dropout(0.2)(concatenate(embeds)) if embeds else None

    segments = get_segments(seg_input, seg_out_dims) if seg_flag and seg_input is not None else[]
    segments = segments + (get_segments(num_input, num_segs) if seg_num_flag and num_input is not None else [])
    segments = Dropout(0.2)(concatenate(segments)) if segments else None

    feats = [embeds] if embeds is not None else []
    if segments is not None:
        feats.append(segments)
    if seg_input is not None and (add_seg_src or not seg_flag):
        feats.append(seg_input)
    if num_input is not None:
        feats.append(num_input)
    feats = concatenate(feats)

    extra_feats = get_extra_layers(x, feats) if get_extra_layers is not None else None

    if use_fm:
        fm = FMLayer(fm_dim, activation=fm_activation)(feats)
        fm = Dropout(fm_dropout)(fm)
        flat = concatenate([feats, fm])
    else:
        flat = feats

    flat = concatenate([flat, extra_feats]) if extra_feats is not None else flat

    flat = add_dense(flat, hidden_units, bn=True, dropout=hidden_dropout)
    tnn = get_output(flat, oh_input, cat_input, seg_input, num_input)
    return tnn
