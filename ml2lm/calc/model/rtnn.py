from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.engine import InputLayer
from keras.initializers import Constant
from keras.layers import Add

from ml2lm.calc.model.tnn import *
from ml2lm.calc.model.units.CVAccelerator import CVAccelerator


def simple_train(tnn_model, tx, ty, vx=None, vy=None, epochs=300, batch_size=1024, model_save_dir='.', model_id='rtnn',
                 lr_patience=30, stop_patience=50):
    checkpointer = ModelCheckpoint(os.path.join(model_save_dir, model_id), monitor='val_loss', verbose=1,
                                   save_best_only=True, save_weights_only=True, period=1)
    checkpointer_reboot = ModelCheckpoint(os.path.join(model_save_dir, f'{model_id}_reboot'), monitor='val_loss',
                                          verbose=1, save_best_only=False, save_weights_only=True, period=1)
    accelerator = CVAccelerator(monitor='val_loss', tol=1e-4, regret_rate=0.1, cooldown=2, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, verbose=1, min_lr=1e-5)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=stop_patience, verbose=1)

    tnn_model.fit(tx, ty, epochs=epochs, batch_size=batch_size, validation_data=(vx, vy) if vx is not None else None,
                  verbose=2, callbacks=[checkpointer_reboot, checkpointer, accelerator, lr_scheduler, early_stopper])

    return read_weights(tnn_model, os.path.join(model_save_dir, model_id))


def get_rtnn_models(x, ignore_inputs=None, num_round=20, res_shrinkage=0.1, get_output=get_linear_output,
                    compile_func=compile_default_mse_output, cat_in_dims=None, cat_out_dims=None, seg_out_dims=None,
                    num_segs=None, seg_type=0, seg_x_val_range=(0, 1), use_fm=False, seg_flag=True, add_seg_src=True,
                    seg_num_flag=True, get_extra_layers=None, embed_dropout=0.2, seg_func=seu, seg_dropout=0.1,
                    fm_dim=320, fm_dropout=0.3, fm_activation='relu', fm_dist_func=lrelu, fm_rel_types='d',
                    fm_exclude_self=False, get_last_layers=get_default_dense_layers, hidden_units=(320, 64),
                    hidden_activation=seu, hidden_dropouts=(0.3, 0.05)):
    if isinstance(res_shrinkage, float):
        res_shrinkage = [res_shrinkage] * (num_round - 1)
    cat_input = Input(shape=[x['cats'].shape[1]], name='cats') if 'cats' in x else None
    seg_input = Input(shape=[x['segs'].shape[1]], name='segs') if 'segs' in x else None
    num_input = Input(shape=[x['nums'].shape[1]], name='nums') if 'nums' in x else None
    extra_inputs = []

    rtnns = []
    for i in range(num_round):
        has_cat, has_seg, has_num = True, True, True
        if ignore_inputs is not None:
            has_cat = 'cats' not in ignore_inputs[i]
            has_seg = 'segs' not in ignore_inputs[i]
            has_num = 'nums' not in ignore_inputs[i]

        tnn, extra_inputs = get_tnn_block(
            i, get_output=get_output, cat_input=cat_input if has_cat else None,
            seg_input=seg_input if has_seg else None, num_input=num_input if has_num else None, cat_in_dims=cat_in_dims,
            cat_out_dims=cat_out_dims, seg_out_dims=seg_out_dims, num_segs=num_segs, seg_type=seg_type,
            seg_x_val_range=seg_x_val_range, use_fm=use_fm, seg_flag=seg_flag, add_seg_src=add_seg_src,
            seg_num_flag=seg_num_flag, x=x, extra_inputs=extra_inputs, get_extra_layers=get_extra_layers,
            embed_dropout=embed_dropout, seg_func=seg_func, seg_dropout=seg_dropout, fm_dim=fm_dim,
            fm_dropout=fm_dropout, fm_activation=fm_activation, fm_dist_func=fm_dist_func, fm_rel_types=fm_rel_types,
            fm_exclude_self=fm_exclude_self, get_last_layers=get_last_layers, hidden_units=hidden_units,
            hidden_activation=hidden_activation, hidden_dropouts=hidden_dropouts)
        if i > 0:
            lp_input = Input(shape=[1], name='lp')
            tnn = Add()([tnn, Lambda(lambda ele: res_shrinkage[i - 1] * ele)(lp_input)])
            extra_inputs.append(lp_input)

        tnn = compile_func(tnn, cat_input=cat_input, seg_input=seg_input, num_input=num_input,
                           other_inputs=extra_inputs)
        rtnns.append(tnn)
    return rtnns


def train(tx, ty, tnn_models, train_func=simple_train, vx=None, vy=None, res_shrinkage=0.1, predict_batch_size=10000,
          **train_params):
    num_round = len(tnn_models)
    if isinstance(res_shrinkage, float):
        res_shrinkage = [res_shrinkage] * (num_round - 1)
    rtnns = []
    last_tp, last_vp = None, None
    for i in range(num_round):
        tnn_model = tnn_models[i]
        if last_tp is not None:
            tx['lp'] = last_tp
        if last_vp is not None:
            vx['lp'] = last_vp
        model_id = 'rtnn'
        if 'model_id' in train_params:
            model_id = train_params['model_id']
        train_params['model_id'] = f'{model_id}_{i}'
        rtnn = train_func(tnn_model, tx, ty, vx, vy, **train_params)
        rtnns.append(rtnn)

        tp = np.squeeze(rtnn.predict(tx, batch_size=predict_batch_size))
        vp = np.squeeze(rtnn.predict(vx, batch_size=predict_batch_size)) if vx is not None else None
        if i < num_round - 1:
            last_tp = tp + res_shrinkage[i - 1] * (1 / res_shrinkage[i] - 1) * last_tp if last_tp is not None else tp
            if vp is not None:
                last_vp = vp + res_shrinkage[i - 1] * (1 / res_shrinkage[i] - 1) * last_vp \
                    if last_vp is not None else vp

    if 'lp' in tx:
        del tx['lp']
    if 'lp' in vx:
        del vx['lp']
    return rtnns


def predict(x, rtnns, res_shrinkage, predict_batch_size=10000):
    num_round = len(rtnns)
    if isinstance(res_shrinkage, float):
        res_shrinkage = [res_shrinkage] * (num_round - 1)
    last_p, p = None, None
    if rtnns:
        for i in range(num_round):
            rtnn = rtnns[i]
            if last_p is not None:
                x['lp'] = last_p
            p = np.squeeze(rtnn.predict(x, batch_size=predict_batch_size))
            if i < num_round - 1:
                last_p = p + res_shrinkage[i - 1] * (1 / res_shrinkage[i] - 1) * last_p if last_p is not None else p

    if 'lp' in x:
        del x['lp']
    return p


def merge(tnn_models, res_shrinkage, rs_parameterized=True, rs_learnable=False):
    num_round = len(tnn_models)
    if isinstance(res_shrinkage, float):
        res_shrinkage = [res_shrinkage] * (num_round - 1)
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
            rtnn = rtnn.layers[-1].output
            out_dim = bk.int_shape(rtnn)[-1]
            res_factor = Lambda(lambda ele: bk.ones_like(ele) / out_dim)(rtnn)
            res_factor = Dense(out_dim, use_bias=False, kernel_initializer=Constant(res_shrinkage[i]),
                               trainable=rs_learnable, name=f'block{i}_rs')(res_factor)
            outputs.append(Multiply()([rtnn, res_factor]))
    else:
        for rtnn in rtnn_parts[:-1]:
            outputs.append(Lambda(lambda x: res_shrinkage[i] * x)(rtnn.layers[-1].output))
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
