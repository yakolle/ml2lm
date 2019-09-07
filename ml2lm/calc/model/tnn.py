from keras import Input, Model
from keras.layers import concatenate, Multiply
from keras.optimizers import Adam

from ml2lm.calc.model.nn_util import *
from ml2lm.calc.model.units.rel import *


def get_linear_output(flat, name=None):
    return Dense(1, name=name)(flat)


def compile_default_mse_output(outputs, cat_input=None, seg_input=None, num_input=None, other_inputs=None,
                               loss_weights=None):
    inputs = [cat_input] if cat_input is not None else []
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)
    if other_inputs:
        inputs.extend(other_inputs)

    dnn = Model(inputs, outputs)
    dnn.compile(loss='mse', optimizer=Adam(lr=1e-3), loss_weights=loss_weights)
    return dnn


def get_sigmoid_output(flat, name=None):
    return Dense(1, activation='sigmoid', name=name)(flat)


def get_default_dense_layers(feats, extra_feats, hidden_units=(320, 64), hidden_activation=seu,
                             hidden_dropouts=(0.3, 0.05)):
    flat = concatenate([feats, extra_feats]) if feats is not None and extra_feats is not None else \
        feats if feats is not None else extra_feats
    if hidden_units:
        hidden_layer_num = len(hidden_units)
        for i in range(hidden_layer_num):
            flat = add_dense(flat, hidden_units[i], bn=i < hidden_layer_num - 1 or 1 == hidden_layer_num,
                             activation=hidden_activation, dropout=hidden_dropouts[i])
    return flat


def compile_default_bce_output(outputs, cat_input=None, seg_input=None, num_input=None, other_inputs=None,
                               loss_weights=None):
    inputs = [cat_input] if cat_input is not None else []
    if seg_input is not None:
        inputs.append(seg_input)
    if num_input is not None:
        inputs.append(num_input)
    if other_inputs:
        inputs.extend(other_inputs)

    dnn = Model(inputs, outputs)
    dnn.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), loss_weights=loss_weights)
    return dnn


def get_default_rel_conf():
    return [{'rel_id': 'fm', 'dropout': 0.4,
             'conf': {'factor_rank': 320, 'dist_func': lrelu, 'rel_types': 'd', 'exclude_selves': (False,)}},
            {'rel_id': 'bi_rel', 'dropout': 0.8,
             'conf': {'factor_rank': 320, 'trans_func': lrelu, 'op_func': rel_mul, 'rel_types': 'd'}}]


def get_rel_layer(rel_conf, feats):
    rels = []
    for conf in rel_conf:
        if 'fm' == conf['rel_id']:
            rel = FMLayer(**dict(conf['conf']))(feats)
        else:
            rel = BiRelLayer(**dict(conf['conf']))(feats)
        dropout = conf['dropout']
        if dropout > 0:
            rel = Dropout(dropout)(rel)
        rels.append(rel)
    return concatenate([feats] + rels)


def get_seish_tnn_block(block_no, get_output=get_linear_output, cat_input=None, seg_input=None, num_input=None,
                        pre_output=None, cat_in_dims=None, cat_out_dims=None, seg_out_dims=None, num_segs=None,
                        seg_type=0, seg_x_val_range=(0, 1), seg_y_val_range=(0, 1), seg_y_dim=100, shrink_factor=1.0,
                        seg_flag=True, add_seg_src=True, seg_num_flag=True, x=None, extra_inputs=None,
                        get_extra_layers=None, embed_dropout=0.2, seg_func=seu, seg_dropout=0.1,
                        rel_conf=get_default_rel_conf(), get_last_layers=get_default_dense_layers,
                        hidden_units=(320, 64), hidden_activation=seu, hidden_dropouts=(0.3, 0.05), feat_seg_bin=False,
                        feat_only_bin=False, pred_seg_bin=False, add_pred=False):
    embeds = get_embeds(cat_input, cat_in_dims, cat_out_dims,
                        shrink_factor=shrink_factor ** block_no) if cat_input is not None else []
    embeds = BatchNormalization()(concatenate(embeds)) if embeds else None
    if embed_dropout > 0:
        embeds = Dropout(embed_dropout)(embeds) if embeds is not None else None

    splitters = get_segments(seg_input, seg_out_dims, shrink_factor ** block_no, seg_type, seg_func, seg_x_val_range,
                             feat_seg_bin, feat_only_bin) if seg_flag and seg_input is not None else[]
    splitters += get_segments(num_input, num_segs, shrink_factor ** block_no, seg_type, seg_func, seg_x_val_range,
                              feat_seg_bin, feat_only_bin) if seg_num_flag and num_input is not None else []

    if pre_output is not None:
        seg_y_dim = shrink(seg_y_dim, shrink_factor ** block_no)
        if not seg_type:
            segment = SegTriangleLayer(seg_y_dim, seg_y_val_range, seg_func, include_seg_bin=feat_seg_bin,
                                       only_seg_bin=feat_only_bin)(pre_output)
        else:
            segment = SegRightAngleLayer(seg_y_dim, seg_y_val_range, seg_func, include_seg_bin=feat_seg_bin,
                                         only_seg_bin=feat_only_bin)(pre_output)
        splitters.append(segment)
    splitters = concatenate(splitters) if splitters else None

    estimator_inputs = []
    if splitters is not None:
        if embeds is not None:
            estimator_inputs.append(embeds)
        if seg_input is not None:
            estimator_inputs.append(seg_input)
        if num_input is not None:
            estimator_inputs.append(num_input)
        if pre_output is not None:
            estimator_inputs.append(pre_output)
    estimator_inputs = concatenate(estimator_inputs) if estimator_inputs else None
    estimators = Dense(bk.int_shape(splitters)[-1])(
        estimator_inputs) if splitters is not None and estimator_inputs is not None else None

    segments = Multiply()([splitters, estimators]) if splitters is not None and estimators is not None else None
    segments = BatchNormalization()(segments) if segments is not None else None
    if seg_dropout > 0:
        segments = Dropout(seg_dropout)(segments) if segments is not None else None

    feats = [embeds] if embeds is not None else []
    if segments is not None:
        feats.append(segments)
    if seg_input is not None and (add_seg_src or not seg_flag):
        feats.append(seg_input)
    if num_input is not None:
        feats.append(num_input)
    if pre_output is not None:
        feats.append(pre_output)
    feats = concatenate(feats) if len(feats) > 1 else feats[0] if feats else None

    extra_feats, extra_inputs = get_extra_layers(x, feats, extra_inputs) if get_extra_layers is not None else (
        None, extra_inputs)

    if rel_conf and feats is not None:
        feats = get_rel_layer(rel_conf, feats)

    flat = get_last_layers(feats, extra_feats, hidden_units=hidden_units, hidden_activation=hidden_activation,
                           hidden_dropouts=hidden_dropouts)
    tnn_block = get_output(flat, name=f'out{block_no}')

    if pred_seg_bin:
        if not seg_type:
            bin_block = SegTriangleLayer(seg_y_dim, seg_y_val_range, seg_func, include_seg_bin=True,
                                         only_seg_bin=False)(tnn_block)
        else:
            bin_block = SegRightAngleLayer(seg_y_dim, seg_y_val_range, seg_func, include_seg_bin=True,
                                           only_seg_bin=False)(tnn_block)

        if add_pred:
            tnn_block = Dense(1)(concatenate([tnn_block, bin_block]))
        else:
            tnn_block = Dense(1)(bin_block)

    return tnn_block, extra_inputs


def get_tnn_block(block_no, get_output=get_linear_output, cat_input=None, seg_input=None, num_input=None,
                  pre_output=None, cat_in_dims=None, cat_out_dims=None, seg_out_dims=None, num_segs=None, seg_type=0,
                  seg_x_val_range=(0, 1), seg_y_val_range=(0, 1), seg_y_dim=50, shrink_factor=1.0, seg_flag=True,
                  add_seg_src=True, seg_num_flag=True, x=None, extra_inputs=None, get_extra_layers=None,
                  embed_dropout=0.2, seg_func=seu, seg_dropout=0.1, rel_conf=get_default_rel_conf(),
                  get_last_layers=get_default_dense_layers, hidden_units=(320, 64), hidden_activation=seu,
                  hidden_dropouts=(0.3, 0.05), feat_seg_bin=False, feat_only_bin=False, pred_seg_bin=False,
                  add_pred=False):
    embeds = get_embeds(cat_input, cat_in_dims, cat_out_dims,
                        shrink_factor=shrink_factor ** block_no) if cat_input is not None else []
    embeds = BatchNormalization()(concatenate(embeds)) if embeds else None
    if embed_dropout > 0:
        embeds = Dropout(embed_dropout)(embeds) if embeds is not None else None

    segments = get_segments(seg_input, seg_out_dims, shrink_factor ** block_no, seg_type, seg_func, seg_x_val_range,
                            feat_seg_bin, feat_only_bin) if seg_flag and seg_input is not None else[]
    segments += get_segments(num_input, num_segs, shrink_factor ** block_no, seg_type, seg_func, seg_x_val_range,
                             feat_seg_bin, feat_only_bin) if seg_num_flag and num_input is not None else []

    if pre_output is not None:
        seg_y_dim = shrink(seg_y_dim, shrink_factor ** block_no)
        if not seg_type:
            segment = SegTriangleLayer(seg_y_dim, seg_y_val_range, seg_func, include_seg_bin=feat_seg_bin,
                                       only_seg_bin=feat_only_bin)(pre_output)
        else:
            segment = SegRightAngleLayer(seg_y_dim, seg_y_val_range, seg_func, include_seg_bin=feat_seg_bin,
                                         only_seg_bin=feat_only_bin)(pre_output)
        segments.append(segment)
    segments = BatchNormalization()(concatenate(segments)) if segments else None
    if seg_dropout > 0:
        segments = Dropout(seg_dropout)(segments) if segments is not None else None

    feats = [embeds] if embeds is not None else []
    if segments is not None:
        feats.append(segments)
    if seg_input is not None and (add_seg_src or not seg_flag):
        feats.append(seg_input)
    if num_input is not None:
        feats.append(num_input)
    if pre_output is not None:
        feats.append(pre_output)
    feats = concatenate(feats) if len(feats) > 1 else feats[0] if feats else None

    extra_feats, extra_inputs = get_extra_layers(x, feats, extra_inputs) if get_extra_layers is not None else (
        None, extra_inputs)

    if rel_conf and feats is not None:
        feats = get_rel_layer(rel_conf, feats)

    flat = get_last_layers(feats, extra_feats, hidden_units=hidden_units, hidden_activation=hidden_activation,
                           hidden_dropouts=hidden_dropouts)
    tnn_block = get_output(flat, name=f'out{block_no}')

    if pred_seg_bin:
        if not seg_type:
            bin_block = SegTriangleLayer(seg_y_dim, seg_y_val_range, seg_func, include_seg_bin=True,
                                         only_seg_bin=False)(tnn_block)
        else:
            bin_block = SegRightAngleLayer(seg_y_dim, seg_y_val_range, seg_func, include_seg_bin=True,
                                           only_seg_bin=False)(tnn_block)

        if add_pred:
            tnn_block = Dense(1)(concatenate([tnn_block, bin_block]))
        else:
            tnn_block = Dense(1)(bin_block)

    return tnn_block, extra_inputs


def get_tnn_model(x, get_output=get_linear_output, compile_func=compile_default_mse_output, cat_in_dims=None,
                  cat_out_dims=None, seg_out_dims=None, num_segs=None, seg_type=0, seg_x_val_range=(0, 1),
                  seg_flag=True, add_seg_src=True, seg_num_flag=True, get_extra_layers=None, embed_dropout=0.2,
                  seg_func=seu, seg_dropout=0.1, rel_conf=get_default_rel_conf(),
                  get_last_layers=get_default_dense_layers, hidden_units=(320, 64), hidden_activation=seu,
                  hidden_dropouts=(0.3, 0.05), feat_seg_bin=False, feat_only_bin=False, pred_seg_bin=False,
                  add_pred=False):
    cat_input = Input(shape=[x['cats'].shape[1]], name='cats') if 'cats' in x else None
    seg_input = Input(shape=[x['segs'].shape[1]], name='segs') if 'segs' in x else None
    num_input = Input(shape=[x['nums'].shape[1]], name='nums') if 'nums' in x else None

    tnn, extra_inputs = get_tnn_block(
        0, get_output=get_output, cat_input=cat_input, seg_input=seg_input, num_input=num_input,
        cat_in_dims=cat_in_dims, cat_out_dims=cat_out_dims, seg_out_dims=seg_out_dims, num_segs=num_segs,
        seg_type=seg_type, seg_x_val_range=seg_x_val_range, seg_flag=seg_flag, add_seg_src=add_seg_src,
        seg_num_flag=seg_num_flag, x=x, get_extra_layers=get_extra_layers, embed_dropout=embed_dropout,
        seg_func=seg_func, seg_dropout=seg_dropout, rel_conf=rel_conf, get_last_layers=get_last_layers,
        hidden_units=hidden_units, hidden_activation=hidden_activation, hidden_dropouts=hidden_dropouts,
        feat_seg_bin=feat_seg_bin, feat_only_bin=feat_only_bin, pred_seg_bin=pred_seg_bin, add_pred=add_pred)
    tnn = compile_func(tnn, cat_input=cat_input, seg_input=seg_input, num_input=num_input, other_inputs=extra_inputs)
    return tnn
