from keras import Input, Model
from keras.layers import concatenate, Multiply
from keras.optimizers import Adam

from ml2lm.calc.model.nn_util import *
from ml2lm.calc.model.units.FMLayer import FMLayer


def get_simple_linear_output(flat, name=None, unit_activation=seu):
    flat = add_dense(flat, 64, bn=False, activation=unit_activation, dropout=0.05)
    return Dense(1, name=name)(flat)


def compile_default_mse_output(outputs, oh_input=None, cat_input=None, seg_input=None, num_input=None,
                               other_inputs=None, loss_weights=None):
    inputs = [oh_input] if oh_input is not None else []
    if cat_input is not None:
        inputs.append(cat_input)
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)
    if other_inputs:
        inputs.extend(other_inputs)

    dnn = Model(inputs, outputs)
    dnn.compile(loss='mse', optimizer=Adam(lr=1e-3), loss_weights=loss_weights)
    return dnn


def get_simple_sigmoid_output(flat, name=None, unit_activation=seu):
    flat = add_dense(flat, 64, bn=False, activation=unit_activation, dropout=0.05)
    return Dense(1, activation='sigmoid', name=name)(flat)


def compile_default_bce_output(outputs, oh_input=None, cat_input=None, seg_input=None, num_input=None,
                               other_inputs=None, loss_weights=None):
    inputs = [oh_input] if oh_input is not None else []
    if cat_input is not None:
        inputs.append(cat_input)
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)
    if other_inputs:
        inputs.extend(other_inputs)

    dnn = Model(inputs, outputs)
    dnn.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), loss_weights=loss_weights)
    return dnn


def get_seish_tnn_block(block_no, get_output=get_simple_linear_output, oh_input=None, cat_input=None, seg_input=None,
                        num_input=None, pre_output=None, cat_in_dims=None, cat_out_dims=None, seg_out_dims=None,
                        num_segs=None, seg_type=0, seg_x_val_range=(0, 1), seg_y_val_range=(0, 1), seg_y_dim=50,
                        shrink_factor=1.0, use_fm=False, seg_flag=True, add_seg_src=True, seg_num_flag=True, x=None,
                        get_extra_layers=None, embed_dropout=0.2, seg_func=seu, seg_dropout=0.2, fm_dim=320,
                        fm_dropout=0.2, fm_activation=None, hidden_units=320, hidden_activation=seu,
                        hidden_dropout=0.2):
    embeds = [Flatten()(Embedding(3, 2)(oh_input))] if oh_input is not None else []
    if cat_input is not None:
        embeds += get_embeds(cat_input, cat_in_dims, cat_out_dims, shrink_factor=shrink_factor ** block_no)
    embeds = Dropout(embed_dropout)(concatenate(embeds)) if embeds else None

    splitters = get_segments(seg_input, seg_out_dims, shrink_factor ** block_no, seg_type, seg_func,
                             seg_x_val_range) if seg_flag and seg_input is not None else[]
    splitters += get_segments(num_input, num_segs, shrink_factor ** block_no, seg_type, seg_func,
                              seg_x_val_range) if seg_num_flag and num_input is not None else []

    if pre_output is not None:
        seg_y_dim = shrink(seg_y_dim, shrink_factor ** block_no)
        if not seg_type:
            segment = SegTriangleLayer(seg_y_dim, input_val_range=seg_y_val_range, seg_func=seg_func)(pre_output)
        else:
            segment = SegRightAngleLayer(seg_y_dim, input_val_range=seg_y_val_range, seg_func=seg_func)(pre_output)
        splitters.append(segment)
    splitters = Dropout(seg_dropout)(concatenate(splitters)) if splitters else None

    estimator_inputs = []
    if embeds is not None:
        estimator_inputs.append(embeds)
    if seg_input is not None:
        estimator_inputs.append(seg_input)
    if num_input is not None:
        estimator_inputs.append(num_input)
    if pre_output is not None:
        estimator_inputs.append(pre_output)

    estimator_inputs = concatenate(estimator_inputs)
    estimators = Dense(bk.int_shape(splitters)[-1])(estimator_inputs) if splitters is not None else None
    segments = Multiply()([splitters, estimators]) if splitters is not None else None

    feats = [embeds] if embeds is not None else []
    if segments is not None:
        feats.append(segments)
    if seg_input is not None and (add_seg_src or not seg_flag):
        feats.append(seg_input)
    if num_input is not None:
        feats.append(num_input)
    if pre_output is not None:
        feats.append(pre_output)
    feats = concatenate(feats) if len(feats) > 1 else feats[0]

    extra_feats = get_extra_layers(x, feats) if get_extra_layers is not None else None

    if use_fm:
        fm = FMLayer(fm_dim, activation=fm_activation)(feats)
        fm = Dropout(fm_dropout)(fm)
        flat = concatenate([feats, fm])
    else:
        flat = feats

    flat = concatenate([flat, extra_feats]) if extra_feats is not None else flat

    flat = add_dense(flat, hidden_units, bn=True, activation=hidden_activation, dropout=hidden_dropout)
    tnn_block = get_output(flat, name=f'out{block_no}', unit_activation=hidden_activation)
    return tnn_block


def get_tnn_block(block_no, get_output=get_simple_linear_output, oh_input=None, cat_input=None, seg_input=None,
                  num_input=None, pre_output=None, cat_in_dims=None, cat_out_dims=None, seg_out_dims=None,
                  num_segs=None, seg_type=0, seg_x_val_range=(0, 1), seg_y_val_range=(0, 1), seg_y_dim=50,
                  shrink_factor=1.0, use_fm=False, seg_flag=True, add_seg_src=True, seg_num_flag=True, x=None,
                  get_extra_layers=None, embed_dropout=0.2, seg_func=seu, seg_dropout=0.2, fm_dim=320, fm_dropout=0.2,
                  fm_activation=None, hidden_units=320, hidden_activation=seu, hidden_dropout=0.2):
    embeds = [Flatten()(Embedding(3, 2)(oh_input))] if oh_input is not None else []
    if cat_input is not None:
        embeds += get_embeds(cat_input, cat_in_dims, cat_out_dims, shrink_factor=shrink_factor ** block_no)
    embeds = Dropout(embed_dropout)(concatenate(embeds)) if embeds else None

    segments = get_segments(seg_input, seg_out_dims, shrink_factor ** block_no, seg_type, seg_func,
                            seg_x_val_range) if seg_flag and seg_input is not None else[]
    segments += get_segments(num_input, num_segs, shrink_factor ** block_no, seg_type, seg_func,
                             seg_x_val_range) if seg_num_flag and num_input is not None else []

    if pre_output is not None:
        seg_y_dim = shrink(seg_y_dim, shrink_factor ** block_no)
        if not seg_type:
            segment = SegTriangleLayer(seg_y_dim, input_val_range=seg_y_val_range, seg_func=seg_func)(pre_output)
        else:
            segment = SegRightAngleLayer(seg_y_dim, input_val_range=seg_y_val_range, seg_func=seg_func)(pre_output)
        segments.append(segment)
    segments = Dropout(seg_dropout)(concatenate(segments)) if segments else None

    feats = [embeds] if embeds is not None else []
    if segments is not None:
        feats.append(segments)
    if seg_input is not None and (add_seg_src or not seg_flag):
        feats.append(seg_input)
    if num_input is not None:
        feats.append(num_input)
    if pre_output is not None:
        feats.append(pre_output)
    feats = concatenate(feats) if len(feats) > 1 else feats[0]

    extra_feats = get_extra_layers(x, feats) if get_extra_layers is not None else None

    if use_fm:
        fm = FMLayer(fm_dim, activation=fm_activation)(feats)
        fm = Dropout(fm_dropout)(fm)
        flat = concatenate([feats, fm])
    else:
        flat = feats

    flat = concatenate([flat, extra_feats]) if extra_feats is not None else flat

    flat = add_dense(flat, hidden_units, bn=True, activation=hidden_activation, dropout=hidden_dropout)
    tnn_block = get_output(flat, name=f'out{block_no}', unit_activation=hidden_activation)
    return tnn_block


def get_tnn_model(x, get_output=get_simple_linear_output, compile_func=compile_default_mse_output, cat_in_dims=None,
                  cat_out_dims=None, seg_out_dims=None, num_segs=None, seg_type=0, seg_x_val_range=(0, 1), use_fm=False,
                  seg_flag=True, add_seg_src=True, seg_num_flag=True, get_extra_layers=None, embed_dropout=0.2,
                  seg_func=seu, seg_dropout=0.2, fm_dim=320, fm_dropout=0.2, fm_activation=None, hidden_units=320,
                  hidden_activation=seu, hidden_dropout=0.2):
    oh_input = Input(shape=[x['ohs'].shape[1]], name='ohs') if 'ohs' in x else None
    cat_input = Input(shape=[x['cats'].shape[1]], name='cats') if 'cats' in x else None
    seg_input = Input(shape=[x['segs'].shape[1]], name='segs') if 'segs' in x else None
    num_input = Input(shape=[x['nums'].shape[1]], name='nums') if 'nums' in x else None

    tnn = get_tnn_block(0, get_output=get_output, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input,
                        num_input=num_input, cat_in_dims=cat_in_dims, cat_out_dims=cat_out_dims,
                        seg_out_dims=seg_out_dims, num_segs=num_segs, seg_type=seg_type,
                        seg_x_val_range=seg_x_val_range, use_fm=use_fm, seg_flag=seg_flag, add_seg_src=add_seg_src,
                        seg_num_flag=seg_num_flag, x=x, get_extra_layers=get_extra_layers, embed_dropout=embed_dropout,
                        seg_func=seg_func, seg_dropout=seg_dropout, fm_dim=fm_dim, fm_dropout=fm_dropout,
                        fm_activation=fm_activation, hidden_units=hidden_units, hidden_activation=hidden_activation,
                        hidden_dropout=hidden_dropout)
    tnn = compile_func(tnn, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input, num_input=num_input)
    return tnn
