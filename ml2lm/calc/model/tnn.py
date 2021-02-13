from keras import Input, Model
from keras.layers import concatenate, Multiply
from keras.optimizers import Adam

from ml2lm.calc.model.nn_util import *
from ml2lm.calc.model.units.embed import *
from ml2lm.calc.model.units.rel import *


def make_output(activation=None):
    def _output(flat, name=None):
        return Dense(1, activation=activation, name=name)(flat)

    return _output


def get_linear_output(flat, name=None):
    return make_output()(flat, name=name)


def make_compile_output(loss):
    def _compile_output(inputs, outputs, extra_inputs=None, loss_weights=None, init_lr=1e-3, metrics=None):
        inputs = list(inputs.values())
        if extra_inputs:
            inputs.extend(extra_inputs)

        dnn = Model(inputs, outputs)
        dnn.compile(loss=loss, optimizer=Adam(lr=init_lr), metrics=metrics, loss_weights=loss_weights)
        return dnn

    return _compile_output


def compile_default_mse_output(inputs, outputs, extra_inputs=None, loss_weights=None, init_lr=1e-3, metrics=None):
    return make_compile_output('mse')(inputs, outputs, extra_inputs, loss_weights, init_lr, metrics)


def get_sigmoid_output(flat, name=None):
    return make_output('sigmoid')(flat, name=name)


def get_default_dense_layers(feats, extra_feats, hidden_units=(320, 64), hidden_activation=seu,
                             hidden_dropouts=(0.3, 0.05), hidden_dropout_handler=Dropout):
    flat = concatenate([feats, extra_feats]) if feats is not None and extra_feats is not None else \
        feats if feats is not None else extra_feats
    if hidden_units:
        hidden_layer_num = len(hidden_units)
        for i in range(hidden_layer_num):
            flat = add_dense(flat, hidden_units[i], bn=i < hidden_layer_num - 1 or 1 == hidden_layer_num,
                             activation=hidden_activation, dropout=hidden_dropouts[i],
                             dropout_handler=hidden_dropout_handler)
    return flat


def compile_default_bce_output(inputs, outputs, extra_inputs=None, loss_weights=None, init_lr=1e-3, metrics=None):
    return make_compile_output('binary_crossentropy')(inputs, outputs, extra_inputs, loss_weights, init_lr, metrics)


def get_default_rel_conf():
    return [{'rel_id': 'fm', 'dropout': 0.4, 'dropout_handler': Dropout,
             'conf': {'factor_rank': 320, 'dist_func': lrelu, 'rel_types': 'd', 'exclude_selves': (False,)}},
            {'rel_id': 'br', 'dropout': 0.8, 'dropout_handler': Dropout,
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
            rel = conf.get('dropout_handler', Dropout)(dropout)(rel)
        rels.append(rel)
    return concatenate([feats] + rels)


def get_seish_tnn_block(block_no, inputs, get_output=get_linear_output, pre_output=None, cat_in_dims=None,
                        cat_out_dims=None, seg_out_dims=None, num_segs=None, seg_type=0, seg_x_val_range=(0, 1),
                        seg_y_val_range=(0, 1), seg_y_dim=100, shrink_factor=1.0, seg_flag=True, add_seg_src=True,
                        seg_num_flag=True, x=None, extra_inputs=None, get_extra_layers=None, embed_dropout=0.2,
                        seg_func=seu, seg_dropout=0.1, rel_conf=get_default_rel_conf(),
                        get_last_layers=get_default_dense_layers, hidden_units=(320, 64), hidden_activation=seu,
                        hidden_dropouts=(0.3, 0.05), feat_seg_bin=False, feat_only_bin=False, pred_seg_bin=False,
                        add_pred=False, scale_n=0, scope_type='global', bundle_scale=False,
                        embed_dropout_handler=Dropout, seg_dropout_handler=Dropout, hidden_dropout_handler=Dropout):
    cat_input, seg_input, num_input = inputs.get('cats'), inputs.get('segs'), inputs.get('nums')

    embeds = get_embeds(cat_input, cat_in_dims, cat_out_dims,
                        shrink_factor=shrink_factor ** block_no) if cat_input is not None else []
    embeds = BatchNormalization()(concatenate(embeds)) if embeds else None
    if embed_dropout > 0:
        embeds = embed_dropout_handler(embed_dropout)(embeds) if embeds is not None else None

    splitters = get_segments(seg_input, seg_out_dims, shrink_factor ** block_no, seg_type, seg_func, seg_x_val_range,
                             feat_seg_bin, feat_only_bin, scale_n,
                             scope_type, bundle_scale) if seg_flag and seg_input is not None else[]
    splitters += get_segments(num_input, num_segs, shrink_factor ** block_no, seg_type, seg_func, seg_x_val_range,
                              feat_seg_bin, feat_only_bin, scale_n,
                              scope_type, bundle_scale) if seg_num_flag and num_input is not None else []

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
        segments = seg_dropout_handler(seg_dropout)(segments) if segments is not None else None

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
                           hidden_dropouts=hidden_dropouts, hidden_dropout_handler=hidden_dropout_handler)
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


def get_tnn_block(block_no, inputs, get_output=get_linear_output, pre_output=None, cat_in_dims=None, cat_out_dims=None,
                  seg_out_dims=None, num_segs=None, seg_type=0, seg_x_val_range=(0, 1), seg_y_val_range=(0, 1),
                  seg_y_dim=50, shrink_factor=1.0, seg_flag=True, add_seg_src=True, seg_num_flag=True, x=None,
                  extra_inputs=None, get_extra_layers=None, embed_dropout=0.2, seg_func=seu, seg_dropout=0.1,
                  rel_conf=get_default_rel_conf(), get_last_layers=get_default_dense_layers, hidden_units=(320, 64),
                  hidden_activation=seu, hidden_dropouts=(0.3, 0.05), feat_seg_bin=False, feat_only_bin=False,
                  pred_seg_bin=False, add_pred=False, scale_n=0, scope_type='global', bundle_scale=False,
                  embed_dropout_handler=Dropout, seg_dropout_handler=Dropout, hidden_dropout_handler=Dropout):
    cat_input, seg_input, num_input = inputs.get('cats'), inputs.get('segs'), inputs.get('nums')

    embeds = get_embeds(cat_input, cat_in_dims, cat_out_dims,
                        shrink_factor=shrink_factor ** block_no) if cat_input is not None else []
    embeds = BatchNormalization()(concatenate(embeds)) if embeds else None
    if embed_dropout > 0:
        embeds = embed_dropout_handler(embed_dropout)(embeds) if embeds is not None else None

    segments = get_segments(seg_input, seg_out_dims, shrink_factor ** block_no, seg_type, seg_func, seg_x_val_range,
                            feat_seg_bin, feat_only_bin, scale_n,
                            scope_type, bundle_scale) if seg_flag and seg_input is not None else[]
    segments += get_segments(num_input, num_segs, shrink_factor ** block_no, seg_type, seg_func, seg_x_val_range,
                             feat_seg_bin, feat_only_bin, scale_n,
                             scope_type, bundle_scale) if seg_num_flag and num_input is not None else []

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
        segments = seg_dropout_handler(seg_dropout)(segments) if segments is not None else None

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
                           hidden_dropouts=hidden_dropouts, hidden_dropout_handler=hidden_dropout_handler)
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
                  add_pred=False, scale_n=0, scope_type='global', bundle_scale=False, init_lr=1e-3, metrics=None,
                  embed_dropout_handler=Dropout, seg_dropout_handler=Dropout, hidden_dropout_handler=Dropout):
    inputs = {k: Input(shape=[v.shape[-1] if len(v.shape) > 1 else 1], name=k) for k, v in x.items()}

    tnn, extra_inputs = get_tnn_block(
        0, inputs, get_output=get_output, cat_in_dims=cat_in_dims, cat_out_dims=cat_out_dims, seg_out_dims=seg_out_dims,
        num_segs=num_segs, seg_type=seg_type, seg_x_val_range=seg_x_val_range, seg_flag=seg_flag,
        add_seg_src=add_seg_src, seg_num_flag=seg_num_flag, x=x, get_extra_layers=get_extra_layers,
        embed_dropout=embed_dropout, seg_func=seg_func, seg_dropout=seg_dropout, rel_conf=rel_conf,
        get_last_layers=get_last_layers, hidden_units=hidden_units, hidden_activation=hidden_activation,
        hidden_dropouts=hidden_dropouts, feat_seg_bin=feat_seg_bin, feat_only_bin=feat_only_bin,
        pred_seg_bin=pred_seg_bin, add_pred=add_pred, scale_n=scale_n, scope_type=scope_type, bundle_scale=bundle_scale,
        embed_dropout_handler=embed_dropout_handler, seg_dropout_handler=seg_dropout_handler,
        hidden_dropout_handler=hidden_dropout_handler)
    tnn = compile_func(inputs, tnn, extra_inputs=extra_inputs, init_lr=init_lr, metrics=metrics)
    return tnn


class TnnGenerator(object):
    def __init__(self, x, cat_in_dims=None, cat_out_dims=None, embed_dropout=0.2, embed_dropout_handler=Dropout,
                 add_cat_src=False, seg_type=0, seg_x_val_range=(0, 1), seg_func=seu, feat_seg_bin=False,
                 feat_only_bin=False, scale_n=0, scope_type='global', bundle_scale=False, seg_dropout=0.1,
                 seg_dropout_handler=Dropout, seg_flag=True, seg_out_dims=None, add_seg_src=True, seg_num_flag=True,
                 num_segs=None, rel_conf=None, rel_bn_num_flag=False, rel_embed_src_flag=False, hidden_units=(320, 64),
                 hidden_activation=seu, hidden_dropouts=(0.3, 0.05), hidden_dropout_handler=Dropout,
                 hid_bn_num_flag=False, output_activation=None, loss='mse', init_lr=1e-3, metrics=None):
        self.inputs = {k: Input(shape=[v.shape[-1] if len(v.shape) > 1 else 1], name=k) for k, v in x.items()}

        self.cat_in_dims = cat_in_dims
        self.cat_out_dims = cat_out_dims
        self.embed_dropout = embed_dropout
        self.embed_dropout_handler = embed_dropout_handler
        self.add_cat_src = add_cat_src

        self.seg_type = seg_type
        self.seg_x_val_range = seg_x_val_range
        self.seg_func = seg_func
        self.feat_seg_bin = feat_seg_bin
        self.feat_only_bin = feat_only_bin
        self.scale_n = scale_n
        self.scope_type = scope_type
        self.bundle_scale = bundle_scale
        self.seg_dropout = seg_dropout
        self.seg_dropout_handler = seg_dropout_handler

        self.seg_flag = seg_flag
        self.seg_out_dims = seg_out_dims
        self.add_seg_src = add_seg_src

        self.seg_num_flag = seg_num_flag
        self.num_segs = num_segs

        self.rel_conf = rel_conf
        self.rel_bn_num_flag = rel_bn_num_flag
        self.rel_embed_src_flag = rel_embed_src_flag

        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.hidden_dropouts = hidden_dropouts
        self.hidden_dropout_handler = hidden_dropout_handler
        self.hid_bn_num_flag = hid_bn_num_flag

        self.output_activation = output_activation
        self.loss = loss
        self.init_lr = init_lr
        self.metrics = metrics

        self._embed_src = None
        self._embed = None
        self._seg_src = None
        self._seg = None
        self._cat_src = None
        self._num_src = None
        self._rel_outputs = None
        self._hid_output = None
        self._output = None

    def _build_embed(self):
        cat_input = self.inputs.get('cats')
        if cat_input is not None:
            self._embed_src = concatenate(get_embeds(cat_input, self.cat_in_dims, self.cat_out_dims))

            self._embed = BatchNormalization()(self._embed_src)
            if self.embed_dropout > 0:
                self._embed = self.embed_dropout_handler(self.embed_dropout)(self._embed)

    def _build_seg(self):
        seg_input, num_input = self.inputs.get('segs'), self.inputs.get('nums')
        segments = get_segments(seg_input, self.seg_out_dims, seg_type=self.seg_type, seg_func=self.seg_func,
                                seg_input_val_range=self.seg_x_val_range, seg_bin=self.feat_seg_bin,
                                only_bin=self.feat_only_bin, scale_n=self.scale_n, scope_type=self.scope_type,
                                bundle_scale=self.bundle_scale) if self.seg_flag and seg_input is not None else[]
        segments += get_segments(num_input, self.num_segs, seg_type=self.seg_type, seg_func=self.seg_func,
                                 seg_input_val_range=self.seg_x_val_range, seg_bin=self.feat_seg_bin,
                                 only_bin=self.feat_only_bin, scale_n=self.scale_n, scope_type=self.scope_type,
                                 bundle_scale=self.bundle_scale) if self.seg_num_flag and num_input is not None else []
        self._seg_src = concatenate(segments) if segments else None

        self._seg = BatchNormalization()(self._seg_src) if self._seg_src is not None else None
        if self.seg_dropout > 0:
            self._seg = self.seg_dropout_handler(self.seg_dropout)(self._seg) if segments is not None else None

    def _build_cat(self):
        cat_input = self.inputs.get('cats')
        if cat_input is not None and self.add_cat_src:
            self._cat_src = cat_input

    @staticmethod
    def _merge(feats):
        return concatenate(feats) if len(feats) > 1 else feats[0] if feats else None

    def _build_num(self):
        seg_input, num_input = self.inputs.get('segs'), self.inputs.get('nums')
        nums = []
        if seg_input is not None and (self.add_seg_src or not self.seg_flag):
            nums.append(seg_input)
        if num_input is not None:
            nums.append(num_input)
        self._num_src = self._merge(nums)

    def _get_rels(self, rel_feats):
        rels = []
        for conf in self.rel_conf:
            if 'fm' == conf['rel_id']:
                rel = FMLayer(**dict(conf['conf']))(rel_feats)
            else:
                rel = BiRelLayer(**dict(conf['conf']))(rel_feats)
            dropout = conf['dropout']
            if dropout > 0:
                rel = conf.get('dropout_handler', Dropout)(dropout)(rel)
                rels.append(rel)
        return rels

    def _get_rel_feats(self):
        rel_feats = []
        if self._embed is not None:
            if self.rel_embed_src_flag:
                embed = self._embed_src
                if self.embed_dropout > 0:
                    embed = self.embed_dropout_handler(self.embed_dropout)(embed)
                rel_feats.append(embed)
            else:
                rel_feats.append(self._embed)
        if self._seg is not None:
            rel_feats.append(self._seg)
        if self._num_src is not None:
            rel_feats.append(BatchNormalization()(self._num_src) if self.rel_bn_num_flag else self._num_src)
        return rel_feats

    def _build_rel_block(self):
        rel_feats = self._merge(self._get_rel_feats())
        if self.rel_conf and rel_feats is not None:
            self._rel_outputs = self._get_rels(rel_feats)

    def __add_dense(self, feats, i):
        hidden_layer_num = len(self.hidden_units)
        feats = Dense(self.hidden_units[i])(feats)
        if i < hidden_layer_num - 1 or 1 == hidden_layer_num:
            feats = BatchNormalization()(feats)
        if self.hidden_activation is not None:
            feats = Activation(self.hidden_activation)(feats)
        if self.hidden_dropouts[i] > 0:
            dp_handler = self.hidden_dropout_handler if i < hidden_layer_num - 1 or 1 == hidden_layer_num else Dropout
            feats = dp_handler(self.hidden_dropouts[i])(feats)
        return feats

    def _get_hid_feats(self):
        hid_feats = []
        if self._embed is not None:
            hid_feats.append(self._embed)
        if self._seg is not None:
            hid_feats.append(self._seg)
        if self._cat_src is not None:
            hid_feats.append(BatchNormalization()(self._cat_src))
        if self._num_src is not None:
            hid_feats.append(BatchNormalization()(self._num_src) if self.hid_bn_num_flag else self._num_src)
        if self._rel_outputs:
            hid_feats += self._rel_outputs
        return hid_feats

    def _build_hid_block(self):
        hid_feats = self._merge(self._get_hid_feats())
        if self.hidden_units is not None and hid_feats is not None:
            hidden_layer_num = len(self.hidden_units)
            for i in range(hidden_layer_num):
                hid_feats = self.__add_dense(hid_feats, i)
            self._hid_output = hid_feats

    def _build_output(self):
        self._output = Dense(1, activation=self.output_activation, name='out')(self._hid_output)

    def _build_model(self, need_compile=True):
        tnn = Model(list(self.inputs.values()), self._output)
        if need_compile:
            tnn.compile(loss=self.loss, optimizer=Adam(lr=self.init_lr), metrics=self.metrics)
        return tnn

    def _build_tnn_block(self):
        self._build_embed()
        self._build_seg()
        self._build_cat()
        self._build_num()
        self._build_rel_block()
        self._build_hid_block()
        self._build_output()

    def get_tnn_model(self, need_compile=True):
        self._build_tnn_block()
        return self._build_model(need_compile)


class TnnWithTEGenerator(TnnGenerator):
    def __init__(self, te_cat_conf=None, bn_te_cat_flag=True, te_cat_dp=0., ie_te_cat_flag=False, te_num_conf=None,
                 bn_te_num_flag=True, te_num_dp=0., ie_te_num_flag=False, worker_type=None, **kwargs):
        super(TnnWithTEGenerator, self).__init__(**kwargs)

        self.te_cat_conf = te_cat_conf
        self.bn_te_cat_flag = bn_te_cat_flag
        self.te_cat_dp = min(max(te_cat_dp, 0.), 1.)
        self.ie_te_cat_flag = ie_te_cat_flag

        self.te_num_conf = te_num_conf
        self.bn_te_num_flag = bn_te_num_flag
        self.te_num_dp = min(max(te_num_dp, 0.), 1.)
        self.ie_te_num_flag = ie_te_num_flag

        self.TE = TargetEmbedding4CPU if 'cpu' == worker_type else TargetEmbedding
        self._te_cat = None
        self._te_num = None

    def _build_te_cat(self):
        if self._cat_src is not None and self.te_cat_conf is not None:
            target_input = self.inputs['target']
            te = self.TE(**self.te_cat_conf)
            self._te_cat = te([self._cat_src, target_input])

            if self.bn_te_cat_flag:
                self._te_cat = BatchNormalization()(self._te_cat)
            if self.te_cat_dp > 0:
                self._te_cat = Dropout(self.te_cat_dp)(self._te_cat)

    def _build_te_num(self):
        if self._num_src is not None and self.te_num_conf is not None:
            target_input = self.inputs['target']
            te = self.TE(**self.te_num_conf)
            self._te_num = te([self._num_src, target_input])

            if self.bn_te_num_flag:
                self._te_num = BatchNormalization()(self._te_num)
            if self.te_num_dp > 0:
                self._te_num = Dropout(self.te_num_dp)(self._te_num)

    def _build_rel_block(self):
        rel_feats = self._get_rel_feats()
        if self._te_cat is not None:
            rel_feats.append(self._te_cat)
        if self._te_num is not None:
            rel_feats.append(self._te_num)
        rel_feats = self._merge(rel_feats)

        if self.rel_conf and rel_feats is not None:
            self._rel_outputs = self._get_rels(rel_feats)

    def _build_hid_block(self):
        hid_feats = self._get_hid_feats()
        if self._te_cat is not None:
            hid_feats.append(self._te_cat)
        if self._te_num is not None:
            hid_feats.append(self._te_num)
        hid_feats = self._merge(hid_feats)

        if self.hidden_units is not None and hid_feats is not None:
            hidden_layer_num = len(self.hidden_units)
            for i in range(hidden_layer_num):
                hid_feats = self.__add_dense(hid_feats, i)
            self._hid_output = hid_feats

    def _build_output(self):
        feats = [self._hid_output]
        if self._te_cat is not None and self.ie_te_cat_flag:
            feats.append(self._te_cat)
        if self._te_num is not None and self.ie_te_num_flag:
            feats.append(self._te_num)
        self._output = Dense(1, activation=self.output_activation, name='out')(self._merge(feats))

    def _build_tnn_block(self):
        self._build_embed()
        self._build_seg()
        self._build_cat()
        self._build_num()
        self._build_te_cat()
        self._build_te_num()
        self._build_rel_block()
        self._build_hid_block()
        self._build_output()
