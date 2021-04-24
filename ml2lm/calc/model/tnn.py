import gc

import joblib
from keras import Input, Model
from keras.layers import concatenate
from keras.optimizers import Adam

from ml2lm.calc.model.nn_util import *
from ml2lm.calc.model.units.callbacks import *


class TnnGenerator(object):
    def __init__(self, x, cat_in_dims=None, cat_out_dims=None, embed_dropout=0.2, embed_dropout_handler=Dropout,
                 add_cat_src=False, seg_type=0, seg_x_val_range=(0, 1), seg_func=seu, feat_seg_bin=False,
                 feat_only_bin=False, scale_n=0, scope_type='global', bundle_scale=False, seg_dropout=0.1,
                 seg_dropout_handler=Dropout, seg_flag=True, seg_out_dims=None, add_seg_src=True, seg_num_flag=True,
                 num_segs=None, rel_conf=None, rel_bn_num_flag=False, rel_embed_src_flag=False, hidden_units=(320, 64),
                 hidden_activation=seu, hidden_dropouts=(0.3, 0.05), hidden_dropout_handler=Dropout,
                 hid_bn_num_flag=False, output_activation=None, loss='mse', init_lr=1e-3, nn_metrics=None, **kwargs):
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
        self.nn_metrics = nn_metrics

        self._embed_src = None
        self._embed = None
        self._seg_src = None
        self._seg = None
        self._cat_src = None
        self._num_src = None
        self._rel_outputs = None
        self._hid_output = None
        self._output = None

        self._main_generator = None
        self._generators = []

    def compose(self, generators):
        for generator in generators:
            assert isinstance(generator, TnnGenerator)
            self._generators.append(generator)
            generator._main_generator = self
            generator.inputs = self.inputs

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

    def _build_extra(self):
        for generator in self._generators:
            assert isinstance(generator, TnnGenerator)
            generator._build_extra()

    @staticmethod
    def _get_rels(rel_conf, rel_feats):
        rels = []
        for conf in rel_conf:
            if 'bc' == conf['rel_id']:
                rel = BiCrossLayer(**dict(conf['conf']))(rel_feats)
            else:
                rel_feat = rel_feats[0] if isinstance(rel_feats, (list, tuple)) else rel_feats
                if 'fm' == conf['rel_id']:
                    rel = FMLayer(**dict(conf['conf']))(rel_feat)
                else:
                    rel = BiRelLayer(**dict(conf['conf']))(rel_feat)
            dropout = conf['dropout']
            if dropout > 0:
                rel = conf.get('dropout_handler', Dropout)(dropout)(rel)
                rels.append(rel)
        return rels

    def _get_extra_rel_feats(self) -> list:
        pass

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

        for generator in self._generators:
            assert isinstance(generator, TnnGenerator)
            extra_rel_feats = generator._get_extra_rel_feats()
            if not extra_rel_feats:
                rel_feats += extra_rel_feats

        return rel_feats

    def _build_rel_block(self):
        rel_feats = self._merge(self._get_rel_feats())
        if self.rel_conf and rel_feats is not None:
            self._rel_outputs = self._get_rels(self.rel_conf, rel_feats)

    def _add_dense_(self, feats, i):
        hidden_layer_num = len(self.hidden_units)
        feats = Dense(self.hidden_units[i])(feats)
        if i < hidden_layer_num - 1 or 1 == hidden_layer_num:
            feats = BatchNormalization()(feats)
        if self.hidden_activation is not None:
            feats = Activation(self.hidden_activation)(feats)
        if self.hidden_dropouts[i] > 0.:
            dp_handler = self.hidden_dropout_handler if i < hidden_layer_num - 1 or 1 == hidden_layer_num else Dropout
            feats = dp_handler(self.hidden_dropouts[i])(feats)
        return feats

    def _get_extra_hid_feats(self) -> list:
        pass

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

        for generator in self._generators:
            assert isinstance(generator, TnnGenerator)
            extra_hid_feats = generator._get_extra_hid_feats()
            if not extra_hid_feats:
                hid_feats += extra_hid_feats

        return hid_feats

    def _build_hid_block(self):
        hid_feats = self._merge(self._get_hid_feats())
        if self.hidden_units is not None and hid_feats is not None:
            hidden_layer_num = len(self.hidden_units)
            for i in range(hidden_layer_num):
                hid_feats = self._add_dense_(hid_feats, i)
            self._hid_output = hid_feats

    def _get_extra_output(self) -> list:
        pass

    def _build_output(self):
        outputs = [self._hid_output]
        for generator in self._generators:
            assert isinstance(generator, TnnGenerator)
            extra_outputs = generator._get_extra_output()
            if not extra_outputs:
                outputs += extra_outputs
        self._output = Dense(1, activation=self.output_activation, name='out')(self._merge(outputs))

    def _build_model(self, need_compile=True):
        tnn = Model(list(self.inputs.values()), self._output)
        if need_compile:
            tnn.compile(loss=self.loss, optimizer=Adam(lr=self.init_lr), metrics=self.nn_metrics)
        return tnn

    def _build_tnn_block(self):
        self._build_embed()
        self._build_seg()
        self._build_cat()
        self._build_num()
        self._build_extra()
        self._build_rel_block()
        self._build_hid_block()
        self._build_output()

    def get_tnn_model(self, need_compile=True):
        self._build_tnn_block()
        return self._build_model(need_compile)


class TnnWithTEGenerator(TnnGenerator):
    def __init__(self, te_cat_conf=None, bn_te_cat_flag=True, te_cat_dp=0., ie_te_cat_flag=False, te_num_conf=None,
                 bn_te_num_flag=True, te_num_dp=0., ie_te_num_flag=False, te_dropout_handler=Dropout, worker_type=None,
                 **kwargs):
        super(TnnWithTEGenerator, self).__init__(**kwargs)

        self.te_cat_conf = te_cat_conf
        self.bn_te_cat_flag = bn_te_cat_flag
        self.te_cat_dp = min(max(te_cat_dp, 0.), 1.)
        self.ie_te_cat_flag = ie_te_cat_flag

        self.te_num_conf = te_num_conf
        self.bn_te_num_flag = bn_te_num_flag
        self.te_num_dp = min(max(te_num_dp, 0.), 1.)
        self.ie_te_num_flag = ie_te_num_flag

        self.te_dropout_handler = te_dropout_handler

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
                self._te_cat = self.te_dropout_handler(self.te_cat_dp)(self._te_cat)

    def _build_te_num(self):
        if self._num_src is not None and self.te_num_conf is not None:
            target_input = self.inputs['target']
            te = self.TE(**self.te_num_conf)
            self._te_num = te([self._num_src, target_input])

            if self.bn_te_num_flag:
                self._te_num = BatchNormalization()(self._te_num)
            if self.te_num_dp > 0:
                self._te_num = self.te_dropout_handler(self.te_num_dp)(self._te_num)

    def _build_extra(self):
        self._build_te_cat()
        self._build_te_num()

    def _get_extra_rel_feats(self):
        rel_feats = []
        if self._te_cat is not None:
            rel_feats.append(self._te_cat)
        if self._te_num is not None:
            rel_feats.append(self._te_num)
        return rel_feats

    def _get_extra_hid_feats(self):
        hid_feats = []
        if self._te_cat is not None:
            hid_feats.append(self._te_cat)
        if self._te_num is not None:
            hid_feats.append(self._te_num)
        return hid_feats

    def _get_extra_output(self):
        feats = []
        if self._te_cat is not None and self.ie_te_cat_flag:
            feats.append(self._te_cat)
        if self._te_num is not None and self.ie_te_num_flag:
            feats.append(self._te_num)
        return feats


class TnnWithSEGenerator(TnnGenerator):
    def __init__(self, beam_num=20, cur_phase=1, ev_path='.', importance_mode='add', se_dropout=0.5,
                 se_dropout_handler=Dropout, se_seg_nums=None, feat_idx0=None, embed_trainable_list=None,
                 se_input_val_range=(0., 1.), out_dim_calcor=log_out_dim_calcor, max_param_num=int(1e6), se_period=100,
                 se_pave_momentum=0.5, **kwargs):
        super(TnnWithSEGenerator, self).__init__(**kwargs)

        self.beam_num = beam_num
        self.cur_phase = cur_phase
        self.ev_path = ev_path
        self.importance_mode = importance_mode
        self.se_dropout = se_dropout
        self.se_dropout_handler = se_dropout_handler
        self.se_seg_nums = se_seg_nums
        self.feat_idx0 = feat_idx0
        self.embed_trainable_list = embed_trainable_list
        self.se_input_val_range = se_input_val_range
        self.out_dim_calcor = out_dim_calcor
        self.max_param_num = max_param_num
        self.se_period = se_period
        self.se_pave_momentum = se_pave_momentum

        if isinstance(self.se_dropout, float):
            self.se_dropout = [self.se_dropout] * self.cur_phase
        if self.cur_phase > 1:
            se_auc_list = joblib.load(os.path.join(self.ev_path, 'se_auc_list'))
            joblib.dump(se_auc_list, 'se_auc_list', compress=('gzip', 3))
            se_imp_list_add = joblib.load(os.path.join(self.ev_path, 'se_imp_list_add'))
            joblib.dump(se_imp_list_add, 'se_imp_list_add', compress=('gzip', 3))
            se_imp_list_co = joblib.load(os.path.join(self.ev_path, 'se_imp_list_co'))
            joblib.dump(se_imp_list_co, 'se_imp_list_co', compress=('gzip', 3))
            se_imp_list_co_sub = joblib.load(os.path.join(self.ev_path, 'se_imp_list_co_sub'))
            joblib.dump(se_imp_list_co_sub, 'se_imp_list_co_sub', compress=('gzip', 3))
            se_co_auc_list = joblib.load(os.path.join(self.ev_path, 'se_co_auc_list'))
            joblib.dump(se_co_auc_list, 'se_co_auc_list', compress=('gzip', 3))

            self.se_feat_idx_list = joblib.load(os.path.join(self.ev_path, 'se_feat_idx_list'))
            last_imps = joblib.load(os.path.join(self.ev_path, f'se_imp_list_{self.importance_mode}'))[
                            self.cur_phase - 2][:self.beam_num]
            cur_feat_idx = np.array(sorted([sorted(list(_idx)) for _idx, _imp in last_imps]))
            if len(self.se_feat_idx_list) < self.cur_phase:
                self.se_feat_idx_list.append(cur_feat_idx)
            else:
                self.se_feat_idx_list[self.cur_phase - 1] = cur_feat_idx
        else:
            if 1 == len(self.feat_idx0.shape):
                self.feat_idx0 = np.expand_dims(self.feat_idx0, -1)
            self.se_feat_idx_list = [self.feat_idx0]
        joblib.dump(self.se_feat_idx_list, 'se_feat_idx_list', compress=('gzip', 3))

        self._se_srcs = None
        self._se = None

    def _build_se(self):
        feat_size = len(self.se_feat_idx_list)
        if feat_size < self.cur_phase:
            self.se_feat_idx_list = joblib.load(os.path.join(self.ev_path, 'se_feat_idx_list'))
        if feat_size > self.cur_phase:
            self.se_feat_idx_list = self.se_feat_idx_list[:self.cur_phase]

        inputs = self._merge([self.inputs.get('cats'), self._num_src])
        self.SE = SegEmbedding(seg_nums=self.se_seg_nums, feat_idx_list=self.se_feat_idx_list,
                               embed_trainable_list=self.embed_trainable_list, input_val_range=self.se_input_val_range,
                               out_dim_calcor=self.out_dim_calcor, max_param_num=self.max_param_num,
                               period=self.se_period, pave_momentum=self.se_pave_momentum)
        self._se_srcs = self.SE(inputs)

        ses = []
        for i, se_src in enumerate(self._se_srcs):
            se = BatchNormalization()(se_src)
            if self.se_dropout[i] > 0.:
                se = self.se_dropout_handler(self.se_dropout[i])(se)
            ses.append(se)
        self._se = self._merge(ses)

    def _build_extra(self):
        self._build_se()

    def _get_extra_rel_feats(self):
        rel_feats = []
        if self.rel_embed_src_flag:
            ses = []
            for i, se_src in enumerate(self._se_srcs):
                se = se_src
                if self.se_dropout[i] > 0.:
                    se = self.se_dropout_handler(self.se_dropout[i])(se)
                ses.append(se)
            rel_feats += ses
        else:
            rel_feats.append(self._se)
        return rel_feats

    def _get_extra_hid_feats(self):
        return [self._se]

    def get_tnn_model(self, need_compile=True, model_name=None):
        tnn = super(TnnWithSEGenerator, self).get_tnn_model(need_compile=need_compile)
        if model_name is None or self.cur_phase <= 1:
            return tnn

        self.cur_phase -= 1
        last_trainable_flag = self.embed_trainable_list[self.cur_phase - 1]
        self.embed_trainable_list[self.cur_phase - 1] = True
        last_tnn = super(TnnWithSEGenerator, self).get_tnn_model(need_compile=False)
        last_tnn.load_weights(os.path.join(self.ev_path, f'{model_name}.h5'))
        self.embed_trainable_list[self.cur_phase - 1] = last_trainable_flag
        self.cur_phase += 1

        se_layer = get_layers_by_classes(tnn, (SegEmbedding,))[0]
        last_se_layer = get_layers_by_classes(last_tnn, (SegEmbedding,))[0]
        for i, embeds in enumerate(last_se_layer.embedding_list):
            for j, embed in enumerate(embeds):
                bk.update(se_layer.embedding_list[i][j], embed)

        return tnn

    @staticmethod
    def evaluate(model, model_name, ev_path, ev_data, co_ev_flag=False, end_time=np.inf, pred_batch_size=1024):
        def weight_getter(_layer, i, j):
            return _layer.embedding_list[i][j]

        def wids_getter(_layer, i):
            return list(range(len(_layer.embedding_list[i])))

        def get_feats_idx(_layer, ids):
            return np.hstack([_layer.feat_idx_list[i][j] for i, j in ids])

        def duplicate_judge(_layer, ids1, ids2):
            return np.intersect1d(get_feats_idx(_layer, ids1), get_feats_idx(_layer, ids2)).shape[0] > 0

        layer = evaluate_by_phase(model, model_name, ev_path, SegEmbedding, weight_getter, wids_getter, duplicate_judge,
                                  ev_data, co_ev_flag=co_ev_flag, end_time=end_time, pred_batch_size=pred_batch_size)
        if layer is not None:
            lcn = SegEmbedding.__name__
            se_auc_list = joblib.load(f'{lcn}_auc_list')
            se_auc_list[layer.phase - 1] = [(get_feats_idx(layer, ids), imp) for ids, imp in
                                            se_auc_list[layer.phase - 1]]
            joblib.dump(se_auc_list, 'se_auc_list', compress=('gzip', 3))
            se_imp_list_add = joblib.load(f'{lcn}_imp_list_add')
            se_imp_list_add[layer.phase - 1] = [(get_feats_idx(layer, ids), imp) for ids, imp in
                                                se_imp_list_add[layer.phase - 1]]
            joblib.dump(se_imp_list_add, 'se_imp_list_add', compress=('gzip', 3))
            if os.path.exists(f'{lcn}_imp_list_co'):
                se_imp_list_co = joblib.load(f'{lcn}_imp_list_co')
                se_imp_list_co[layer.phase - 1] = [(get_feats_idx(layer, ids), imp) for ids, imp in
                                                   se_imp_list_co[layer.phase - 1]]
                joblib.dump(se_imp_list_co, 'se_imp_list_co', compress=('gzip', 3))
            if os.path.exists(f'{lcn}_imp_list_co_sub'):
                se_imp_list_co_sub = joblib.load(f'{lcn}_imp_list_co_sub')
                se_imp_list_co_sub[layer.phase - 1] = [(get_feats_idx(layer, ids), imp) for ids, imp in
                                                       se_imp_list_co_sub[layer.phase - 1]]
                joblib.dump(se_imp_list_co_sub, 'se_imp_list_co_sub', compress=('gzip', 3))
            if os.path.exists(f'{lcn}_co_auc_list'):
                se_co_auc_list = joblib.load(f'{lcn}_co_auc_list')
                se_co_auc_list[layer.phase - 1] = [(get_feats_idx(layer, ids), imp) for ids, imp in
                                                   se_co_auc_list[layer.phase - 1]]
                joblib.dump(se_co_auc_list, 'se_co_auc_list', compress=('gzip', 3))


def get_layers_by_classes(model, classes):
    res = []
    for layer in model.layers:
        if isinstance(layer, classes):
            res.append(layer)
    return res


def evaluate_by_phase(model, model_name, ev_path, layer_class, weight_getter, wids_getter, duplicate_judge, ev_data,
                      evaluate_handler=make_evaluate_handler(metrics.roc_auc_score, metric_bias=-1., metric_scale=-1.),
                      score_tag='auc', co_ev_flag=False, end_time=np.inf, pred_batch_size=1024):
    model.load_weights(os.path.join(ev_path, f'{model_name}.h5'))
    layer = get_layers_by_classes(model, (layer_class,))[0]
    lcn = layer_class.__name__

    score_list = joblib.load(f'{lcn}_{score_tag}_list') if os.path.exists(f'{lcn}_{score_tag}_list') else []
    imp_list_add = joblib.load(f'{lcn}_imp_list_add') if os.path.exists(f'{lcn}_imp_list_add') else []
    imp_list_co = joblib.load(f'{lcn}_imp_list_co') if os.path.exists(f'{lcn}_imp_list_co') else []
    imp_list_co_sub = joblib.load(f'{lcn}_imp_list_co_sub') if os.path.exists(f'{lcn}_imp_list_co_sub') else []
    co_score_list = joblib.load(f'{lcn}_co_{score_tag}_list') if os.path.exists(f'{lcn}_co_{score_tag}_list') else []
    if len(score_list) < layer.phase:
        score_list.append([])
    if len(imp_list_add) < layer.phase:
        imp_list_add.append([])
    if len(imp_list_co) < layer.phase:
        imp_list_co.append([])
    if len(imp_list_co_sub) < layer.phase:
        imp_list_co_sub.append([])
    if len(co_score_list) < layer.phase:
        co_score_list.append([])

    def _calc_score():
        return evaluate_handler([_y for _x, _y in ev_data],
                                [np.squeeze(model.predict(_x, batch_size=pred_batch_size)) for _x, _y in ev_data])

    def _update_weights(_ids):
        _ws = [weight_getter(layer, _i, _j) for _i, _j in _ids]
        _vs = bk.batch_get_value(_ws)
        bk.batch_set_value(list(zip(_ws, [np.zeros_like(_v) for _v in _vs])))
        return _ws, _vs

    ev_score = _calc_score()

    if not score_list[layer.phase - 1]:
        for i in wids_getter(layer, layer.phase - 1):
            cur_ids = [(layer.phase - 1, i)]
            cur_ws, cur_vs_bak = _update_weights(cur_ids)
            cur_score = _calc_score() - ev_score
            score_list[layer.phase - 1].append((cur_ids, cur_score))
            bk.batch_set_value(list(zip(cur_ws, cur_vs_bak)))
            print(f'{score_tag}({cur_ids})={cur_score}')
            del cur_ws, cur_vs_bak
            gc.collect()
        joblib.dump(score_list, f'{lcn}_{score_tag}_list', compress=('gzip', 3))

    if not imp_list_add[layer.phase - 1]:
        for i in wids_getter(layer, layer.phase - 1):
            cur_ids, cur_score = score_list[layer.phase - 1][i]
            if cur_score > 0:
                for j in wids_getter(layer, 0):
                    ids0, score0 = score_list[0][j]
                    if score0 > 0 and ((1 == layer.phase and j > i)
                                       or (layer.phase > 1 and not duplicate_judge(layer, cur_ids, ids0))):
                        imp_list_add[layer.phase - 1].append((cur_ids + ids0, cur_score + score0))

    if co_ev_flag:
        last_time = int(time.time())
        spend_times = []
        li, lj = joblib.load('ij') if os.path.exists('ij') else (0, 0)
        i = li
        ei, ej = len(wids_getter(layer, layer.phase - 1)), len(wids_getter(layer, 0))
        while i < ei:
            cur_ids, cur_score = score_list[layer.phase - 1][i]
            if cur_score > 0:
                j = lj if i == li else 0
                while j < ej:
                    ids0, score0 = score_list[0][j]
                    if score0 > 0 and ((1 == layer.phase and j > i)
                                       or (layer.phase > 1 and not duplicate_judge(layer, cur_ids, ids0))):
                        co_ids = cur_ids + ids0
                        cur_ws, cur_vs_bak = _update_weights(co_ids)
                        co_score = _calc_score() - ev_score
                        co_score_list[layer.phase - 1].append((co_ids, co_score))
                        bk.batch_set_value(list(zip(cur_ws, cur_vs_bak)))
                        del cur_ws, cur_vs_bak
                        gc.collect()

                        if co_score > 0:
                            imp_add = cur_score + score0
                            imp_list_co[layer.phase - 1].append((co_ids, co_score))
                            imp_co_sub = co_score - imp_add
                            imp_list_co_sub[layer.phase - 1].append((co_ids, imp_co_sub))

                            print(f'{score_tag}({ids0})={score0}, {score_tag}({cur_ids})={cur_score},',
                                  f'co_{score_tag}={co_score}, imp_add={imp_add}, imp_co_sub={imp_co_sub}')

                        cur_time = int(time.time())
                        spend_times.append(cur_time - last_time)
                        next_time = cur_time + (2 * np.max(spend_times) - np.min(spend_times))
                        if next_time > end_time:
                            joblib.dump((i, j + 1), 'ij', compress=('gzip', 3))
                            joblib.dump(imp_list_add, f'{lcn}_imp_list_add', compress=('gzip', 3))
                            joblib.dump(imp_list_co, f'{lcn}_imp_list_co', compress=('gzip', 3))
                            joblib.dump(imp_list_co_sub, f'{lcn}_imp_list_co_sub', compress=('gzip', 3))
                            joblib.dump(co_score_list, f'{lcn}_co_{score_tag}_list', compress=('gzip', 3))
                            return None
                        last_time = cur_time
                    j += 1
            i += 1
        imp_list_add[layer.phase - 1] = sorted(imp_list_add[layer.phase - 1], key=lambda pair: pair[-1], reverse=True)
        joblib.dump(imp_list_add, f'{lcn}_imp_list_add', compress=('gzip', 3))
        imp_list_co[layer.phase - 1] = sorted(imp_list_co[layer.phase - 1], key=lambda pair: pair[-1], reverse=True)
        joblib.dump(imp_list_co, f'{lcn}_imp_list_co', compress=('gzip', 3))
        imp_list_co_sub[layer.phase - 1] = sorted(imp_list_co_sub[layer.phase - 1], key=lambda pair: pair[-1],
                                                  reverse=True)
        joblib.dump(imp_list_co_sub, f'{lcn}_imp_list_co_sub', compress=('gzip', 3))
        joblib.dump(co_score_list, f'{lcn}_co_{score_tag}_list', compress=('gzip', 3))
    return layer
