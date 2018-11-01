from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
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


def get_rtnn_models(x, num_round=8, res_shrinkage=0.1, get_output=get_simple_linear_output,
                    compile_func=compile_default_mse_output, cat_in_dims=None, cat_out_dims=None, seg_out_dims=None,
                    num_segs=None, seg_type=0, seg_x_val_range=(0, 1), use_fm=False, seg_flag=True, add_seg_src=True,
                    seg_num_flag=True, get_extra_layers=None, embed_dropout=0.2, seg_func=seu, seg_dropout=0.2,
                    fm_dim=320, fm_dropout=0.2, fm_activation=None, hidden_units=320, hidden_activation=seu,
                    hidden_dropout=0.2):
    oh_input = Input(shape=[x['ohs'].shape[1]], name='ohs') if 'ohs' in x else None
    cat_input = Input(shape=[x['cats'].shape[1]], name='cats') if 'cats' in x else None
    seg_input = Input(shape=[x['segs'].shape[1]], name='segs') if 'segs' in x else None
    num_input = Input(shape=[x['nums'].shape[1]], name='nums') if 'nums' in x else None

    rtnns = []
    for i in range(num_round):
        tnn = get_tnn_block(i, get_output=get_output, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input,
                            num_input=num_input, cat_in_dims=cat_in_dims, cat_out_dims=cat_out_dims,
                            seg_out_dims=seg_out_dims, num_segs=num_segs, seg_type=seg_type,
                            seg_x_val_range=seg_x_val_range, use_fm=use_fm, seg_flag=seg_flag, add_seg_src=add_seg_src,
                            seg_num_flag=seg_num_flag, x=x, get_extra_layers=get_extra_layers,
                            embed_dropout=embed_dropout, seg_func=seg_func, seg_dropout=seg_dropout, fm_dim=fm_dim,
                            fm_dropout=fm_dropout, fm_activation=fm_activation, hidden_units=hidden_units,
                            hidden_activation=hidden_activation, hidden_dropout=hidden_dropout)
        other_inputs = None
        if i:
            lp_input = Input(shape=[1], name='lp')
            tnn = Add(name=f'output{i}')([tnn, Lambda(lambda ele: res_shrinkage * ele, name=f'shlp{i}')(lp_input)])
            other_inputs = [lp_input]

        tnn = compile_func(tnn, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input, num_input=num_input,
                           other_inputs=other_inputs)
        rtnns.append(tnn)
    return rtnns


def train(tx, ty, tnn_models, train_func=simple_train, vx=None, vy=None, res_shrinkage=0.1, predict_batch_size=10000,
          **train_params):
    rtnns = []
    last_tp, last_vp = None, None
    for tnn_model in tnn_models:
        if last_tp is not None:
            tx['lp'] = last_tp
        if last_vp is not None:
            vx['lp'] = last_vp
        rtnn = train_func(tnn_model, tx, ty, vx, vy, **train_params)
        rtnns.append(rtnn)

        tp = np.squeeze(rtnn.predict(tx, batch_size=predict_batch_size))
        vp = np.squeeze(rtnn.predict(vx, batch_size=predict_batch_size)) if vx is not None else None
        last_tp = tp + (1 - res_shrinkage) * last_tp if last_tp is not None else tp
        if vp is not None:
            last_vp = vp + (1 - res_shrinkage) * last_vp if last_vp is not None else vp

    if 'lp' in tx:
        del tx['lp']
    if 'lp' in vx:
        del vx['lp']
    return rtnns


def predict(x, rtnns, res_shrinkage, predict_batch_size=10000):
    last_p, p = None, None
    if rtnns:
        for rtnn in rtnns:
            if last_p is not None:
                x['lp'] = last_p
            p = np.squeeze(rtnn.predict(x, batch_size=predict_batch_size))
            last_p = p + (1 - res_shrinkage) * last_p if last_p is not None else p

    if 'lp' in x:
        del x['lp']
    return p


def merge(tnn_models, res_shrinkage):
    rt0nn = tnn_models[0]
    inputs = rt0nn.inputs
    rtnn_parts = [rt0nn]
    for i in range(1, len(tnn_models)):
        rtnn = tnn_models[i]
        out_layer = rtnn.layers[-3]
        for j in range(4):
            rtnn.layers.pop()
        rtnn.layers.append(out_layer)
        rtnn.inputs = inputs
        rtnn_parts.append(rtnn)

    outputs = []
    for rtnn in rtnn_parts[:-1]:
        outputs.append(Lambda(lambda x: res_shrinkage * x)(rtnn.layers[-1].output))
    outputs.append(rtnn_parts[-1].layers[-1].output)
    output = Add()(outputs)
    rtsnn = Model(inputs, output)
    rtsnn.compile(loss=rt0nn.loss, optimizer=rt0nn.optimizer, metrics=rt0nn.metrics)
    return rtsnn
