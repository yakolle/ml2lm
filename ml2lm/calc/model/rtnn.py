from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.engine import InputLayer
from keras.layers import Add, Multiply

from ml2lm.calc.model.tnn import *


def simple_train(tnn_model, tx, ty, vx=None, vy=None, epochs=300, batch_size=1024, model_save_dir='.', model_id='rtnn',
                 lr_patience=30, stop_patience=50):
    checkpointer = ModelCheckpoint(os.path.join(model_save_dir, f'{model_id}.h5'), monitor='val_loss', verbose=1,
                                   save_best_only=True, save_weights_only=True, period=1)
    checkpointer_reboot = ModelCheckpoint(os.path.join(model_save_dir, f'{model_id}_reboot.h5'), monitor='val_loss',
                                          verbose=1, save_best_only=False, save_weights_only=True, period=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, verbose=1, min_lr=1e-5)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=stop_patience, verbose=1)

    tnn_model.fit(tx, ty, epochs=epochs, batch_size=batch_size, validation_data=(vx, vy) if vx is not None else None,
                  verbose=2, callbacks=[checkpointer_reboot, checkpointer, lr_scheduler, early_stopper])

    return read_weights(tnn_model, os.path.join(model_save_dir, model_id))


def get_rtnn_models(x, ignore_inputs=None, num_round=20, res_shrinkage=0.1, shift_val=0.01,
                    get_output=get_linear_output, compile_func=compile_default_mse_output, cat_in_dims=None,
                    cat_out_dims=None, seg_out_dims=None, num_segs=None, seg_type=0, seg_x_val_range=(0, 1),
                    seg_flag=True, add_seg_src=True, seg_num_flag=True, get_extra_layers=None, embed_dropout=0.2,
                    seg_func=seu, seg_dropout=0.1, rel_conf=get_default_rel_conf(),
                    get_last_layers=get_default_dense_layers, hidden_units=(320, 64), hidden_activation=seu,
                    hidden_dropouts=(0.3, 0.05), feat_seg_bin=False, feat_only_bin=False, pred_seg_bin=False,
                    add_pred=False, scale_n=0, scope_type='global', bundle_scale=False, init_lr=1e-3,
                    embed_dropout_handler=Dropout, seg_dropout_handler=Dropout, hidden_dropout_handler=Dropout):
    if isinstance(res_shrinkage, float):
        res_shrinkage = [res_shrinkage] * (num_round - 1)
    if isinstance(shift_val, float):
        shift_val = [shift_val] * (num_round - 1)
    inputs = {k: Input(shape=[v.shape[-1] if len(v.shape) > 1 else 1], name=k) for k, v in x.items()}
    extra_inputs = []

    rtnns = []
    for i in range(num_round):
        if ignore_inputs is not None:
            for data_tag in ignore_inputs:
                inputs.pop(data_tag)

        tnn, extra_inputs = get_tnn_block(
            i, inputs, get_output=get_output, cat_in_dims=cat_in_dims, cat_out_dims=cat_out_dims,
            seg_out_dims=seg_out_dims, num_segs=num_segs, seg_type=seg_type, seg_x_val_range=seg_x_val_range,
            seg_flag=seg_flag, add_seg_src=add_seg_src, seg_num_flag=seg_num_flag, x=x, extra_inputs=extra_inputs,
            get_extra_layers=get_extra_layers, embed_dropout=embed_dropout, seg_func=seg_func, seg_dropout=seg_dropout,
            rel_conf=rel_conf, get_last_layers=get_last_layers, hidden_units=hidden_units,
            hidden_activation=hidden_activation, hidden_dropouts=hidden_dropouts, feat_seg_bin=feat_seg_bin,
            feat_only_bin=feat_only_bin, pred_seg_bin=pred_seg_bin, add_pred=add_pred, scale_n=scale_n,
            scope_type=scope_type, bundle_scale=bundle_scale, embed_dropout_handler=embed_dropout_handler,
            seg_dropout_handler=seg_dropout_handler, hidden_dropout_handler=hidden_dropout_handler)
        if i > 0:
            rp_input = Input(shape=[1], name='rp')
            tnn = Add()([tnn, Lambda(lambda ele: res_shrinkage[i - 1] * (ele - shift_val[i - 1]))(rp_input)])
            extra_inputs.append(rp_input)

        tnn = compile_func(inputs, tnn, extra_inputs=extra_inputs, init_lr=init_lr)
        rtnns.append(tnn)
    return rtnns


def gradually_boosting_smote(x, y, p, supervise=False, boost_p_sections=None, p_threshold=0.5, force=0.1,
                             x_val_range=(0, 1), sup_p_sections=([(0, 0.2)], [(0.8, 1.0)])):
    if boost_p_sections is None:
        boost_p_sections = [(0, 1)]

    section_ind = False
    for l, r in boost_p_sections:
        section_ind |= ((p >= l) & (p <= r))
    boost_x_pos_ind = section_ind & ((0 == y) & (p >= p_threshold))
    boost_x_neg_ind = section_ind & ((1 == y) & (p < p_threshold))
    boost_x_ind = boost_x_pos_ind | boost_x_neg_ind

    if 0. != force:
        x = x.copy()

        if supervise:
            sup_neg_ind = False
            for l, r in sup_p_sections[0]:
                sup_neg_ind |= ((p >= l) & (p <= r))
            sup_neg_x = np.max(x[sup_neg_ind], axis=0, keepdims=True)
            sup_pos_ind = False
            for l, r in sup_p_sections[1]:
                sup_pos_ind |= ((p >= l) & (p <= r))
            sup_pos_x = np.min(x[sup_pos_ind], axis=0, keepdims=True)

            x[boost_x_pos_ind] = force * np.maximum(sup_pos_x, x[boost_x_pos_ind]) + (1 - force) * x[boost_x_pos_ind]
            x[boost_x_neg_ind] = force * np.minimum(sup_neg_x, x[boost_x_neg_ind]) + (1 - force) * x[boost_x_neg_ind]
        else:
            pace = force * (x_val_range[1] - x_val_range[0])
            x[boost_x_pos_ind] += pace
            x[boost_x_neg_ind] -= pace

    return x[boost_x_ind], boost_x_ind


def get_gbs_data(x, y, p, rtnns, res_shrinkage=0.1, shift_val=0.01, predict_batch_size=10000, supervise=False,
                 boost_p_sections=None, p_threshold=0.5, force=0.1, x_val_range=(0, 1),
                 sup_p_sections=([(0, 0.2)], [(0.8, 1.0)])):
    gt_segs, gt_ind = gradually_boosting_smote(x['segs'], y, p, supervise=supervise, boost_p_sections=boost_p_sections,
                                               p_threshold=p_threshold, force=force, x_val_range=x_val_range,
                                               sup_p_sections=sup_p_sections)
    gy = y[gt_ind]
    gx = {k: gt_segs if 'segs' == k else v[gt_ind] for k, v in x.items()}
    gp, gx['rp'] = predict(gx, rtnns=rtnns, res_shrinkage=res_shrinkage, shift_val=shift_val,
                           predict_batch_size=predict_batch_size)

    return gx, gy


def make_gbs_handler(supervise=False, boost_p_sections=None, p_threshold=0.5, force=0.1, x_val_range=(0, 1),
                     sup_p_sections=([(0, 0.2)], [(0.8, 1.0)])):
    def _gbs(x, y, p, rtnns, res_shrinkage=0.1, shift_val=0.01, predict_batch_size=10000):
        return get_gbs_data(x, y, p, rtnns, res_shrinkage=res_shrinkage, shift_val=shift_val,
                            predict_batch_size=predict_batch_size, supervise=supervise,
                            boost_p_sections=boost_p_sections, p_threshold=p_threshold, force=force,
                            x_val_range=x_val_range, sup_p_sections=sup_p_sections)

    return _gbs


def train(tx, ty, tnn_models, train_func=simple_train, vx=None, vy=None, res_shrinkage=0.1, shift_val=0.01,
          predict_batch_size=10000, gbs_handler=None, **train_params):
    num_round = len(tnn_models)
    if isinstance(res_shrinkage, float):
        res_shrinkage = [res_shrinkage] * (num_round - 1)
    if isinstance(shift_val, float):
        shift_val = [shift_val] * (num_round - 1)
    rtnns = []
    tp, vp, res_tp, res_vp = None, None, None, None
    for i in range(num_round):
        tnn_model = tnn_models[i]
        if res_tp is not None:
            tx['rp'] = res_tp
        if res_vp is not None:
            vx['rp'] = res_vp
        model_id = 'rtnn'
        if 'model_id' in train_params:
            model_id = train_params['model_id']
        train_params['model_id'] = f'{model_id}_{i}'
        if gbs_handler is not None and tp is not None:
            gtx0, gty0 = gbs_handler(tx, ty, tp, rtnns[:i], res_shrinkage=res_shrinkage, shift_val=shift_val,
                                     predict_batch_size=predict_batch_size)
            gtx, gty = {col: np.hstack([tx[col], gtx0[col]]) if 'rp' == col else np.vstack([tx[col], gtx0[col]]) for col
                        in tx.keys()}, np.hstack([ty, gty0])
            rtnn = train_func(tnn_model, gtx, gty, vx, vy, **train_params)
        else:
            rtnn = train_func(tnn_model, tx, ty, vx, vy, **train_params)
        rtnns.append(rtnn)

        tp = np.squeeze(rtnn.predict(tx, batch_size=predict_batch_size))
        vp = np.squeeze(rtnn.predict(vx, batch_size=predict_batch_size)) if vx is not None else None
        if i < num_round - 1:
            res_tp = tp + res_shrinkage[i - 1] * (1 / res_shrinkage[i] - 1) * (
                res_tp - shift_val[i - 1]) if res_tp is not None else tp
            if vp is not None:
                res_vp = vp + res_shrinkage[i - 1] * (1 / res_shrinkage[i] - 1) * (
                    res_vp - shift_val[i - 1]) if res_vp is not None else vp

    if 'rp' in tx:
        del tx['rp']
    if 'rp' in vx:
        del vx['rp']
    return rtnns


def predict(x, rtnns, res_shrinkage, shift_val=0.01, predict_batch_size=10000):
    num_round = len(rtnns)
    if isinstance(res_shrinkage, float):
        res_shrinkage = [res_shrinkage] * (num_round - 1)
    if isinstance(shift_val, float):
        shift_val = [shift_val] * (num_round - 1)
    res_p, p = None, None
    if rtnns:
        for i in range(num_round):
            rtnn = rtnns[i]
            if res_p is not None:
                x['rp'] = res_p
            p = np.squeeze(rtnn.predict(x, batch_size=predict_batch_size))
            if i < num_round - 1:
                res_p = p + res_shrinkage[i - 1] * (1 / res_shrinkage[i] - 1) * (
                    res_p - shift_val[i - 1]) if res_p is not None else p

    if 'rp' in x:
        del x['rp']
    return p


def merge(tnn_models, res_shrinkage, shift_val=0.01, rs_parameterized=True, rs_learnable=False):
    num_round = len(tnn_models)
    if isinstance(res_shrinkage, float):
        res_shrinkage = [res_shrinkage] * (num_round - 1)
    if isinstance(shift_val, float):
        shift_val = [shift_val] * (num_round - 1)
    for i, rtnn in enumerate(tnn_models):
        for layer in rtnn.layers:
            if not isinstance(layer, InputLayer):
                layer.name = f'block{i}_{layer.name}'

    rt0nn = tnn_models[0]
    inputs = rt0nn.inputs
    rtnn_parts = [rt0nn]
    for i in range(1, num_round):
        rtnn = tnn_models[i]
        out_layer = rtnn.layers[-3]
        for j in range(4):
            rtnn.layers.pop()
        rtnn.layers.append(out_layer)
        rtnn.inputs = inputs
        rtnn_parts.append(rtnn)

    outputs = []
    if rs_parameterized or rs_learnable:
        for i, rtnn in enumerate(rtnn_parts[:-1]):
            rtnn = Lambda(lambda x: x - shift_val[i])(rtnn.layers[-1].output)
            out_dim = bk.int_shape(rtnn)[-1]
            res_factor = Lambda(lambda ele: bk.ones_like(ele) / out_dim)(rtnn)
            res_factor = Dense(out_dim, use_bias=False, kernel_initializer=Constant(res_shrinkage[i]),
                               trainable=rs_learnable, name=f'block{i}_rs')(res_factor)
            outputs.append(Multiply()([rtnn, res_factor]))
    else:
        for i, rtnn in enumerate(rtnn_parts[:-1]):
            outputs.append(Lambda(lambda x: res_shrinkage[i] * (x - shift_val[i]))(rtnn.layers[-1].output))
    outputs.append(rtnn_parts[-1].layers[-1].output)
    output = Add()(outputs)
    rtsnn = Model(inputs, output)
    rtsnn.compile(loss=rt0nn.loss, optimizer=rt0nn.optimizer, metrics=rt0nn.metrics)
    return rtsnn


def freeze_block(rtsnn, block_no, include_rs=True):
    set_block_learnable_state(rtsnn, block_no, False, include_rs)


def unfreeze_block(rtsnn, block_no, include_rs=True):
    set_block_learnable_state(rtsnn, block_no, True, include_rs)


def set_block_learnable_state(rtsnn, block_no, learnable, include_rs=True):
    for layer in rtsnn.layers:
        if layer.name.startswith(f'block{block_no}_') and (f'block{block_no}_rs' != layer.name or include_rs):
            layer.trainable = learnable
    rtsnn.compile(loss=rtsnn.loss, optimizer=rtsnn.optimizer, metrics=rtsnn.metrics)


def freeze_block_rs(rtsnn, block_no):
    set_block_rs_learnable_state(rtsnn, block_no, False)


def unfreeze_block_rs(rtsnn, block_no):
    set_block_rs_learnable_state(rtsnn, block_no, True)


def set_block_rs_learnable_state(rtsnn, block_no, learnable):
    rtsnn.get_layer(f'block{block_no}_rs').trainable = learnable
    rtsnn.compile(loss=rtsnn.loss, optimizer=rtsnn.optimizer, metrics=rtsnn.metrics)
