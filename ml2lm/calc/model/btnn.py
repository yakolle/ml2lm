from ml2lm.calc.model.tnn import *


def get_btnn_model(x, y, get_output=get_linear_output, compile_func=compile_default_mse_output, cat_in_dims=None,
                   cat_out_dims=None, seg_out_dims=None, num_segs=None, seg_type=0, seg_x_val_range=(0, 1), block_num=3,
                   shrink_factor=1.0, seg_y_dim=50, prev_block_weight_files=None, seg_flag=True, add_seg_src=True,
                   seg_num_flag=True, get_extra_layers=None, embed_dropout=0.2, seg_func=seu, seg_dropout=0.1,
                   rel_conf=get_default_rel_conf(), get_last_layers=get_default_dense_layers, hidden_units=(320, 64),
                   hidden_activation=seu, hidden_dropouts=(0.3, 0.05), feat_seg_bin=False, feat_only_bin=False,
                   pred_seg_bin=False, add_pred=False, scale_n=0, scope_type='global', bundle_scale=False, init_lr=1e-3,
                   embed_dropout_handler=Dropout, seg_dropout_handler=Dropout, hidden_dropout_handler=Dropout):
    inputs = {k: Input(shape=[v.shape[-1] if len(v.shape) > 1 else 1], name=k) for k, v in x.items()}
    extra_inputs = None
    seg_y_val_range = (np.min(y), np.max(y))

    params = {'get_output': get_output, 'cat_in_dims': cat_in_dims, 'cat_out_dims': cat_out_dims,
              'seg_out_dims': seg_out_dims, 'num_segs': num_segs, 'seg_type': seg_type,
              'seg_x_val_range': seg_x_val_range, 'seg_y_val_range': seg_y_val_range, 'seg_y_dim': seg_y_dim,
              'shrink_factor': shrink_factor, 'seg_flag': seg_flag, 'add_seg_src': add_seg_src,
              'seg_num_flag': seg_num_flag, 'x': x, 'get_extra_layers': get_extra_layers,
              'embed_dropout': embed_dropout, 'seg_func': seg_func, 'seg_dropout': seg_dropout, 'rel_conf': rel_conf,
              'get_last_layers': get_last_layers, 'hidden_units': hidden_units, 'hidden_activation': hidden_activation,
              'hidden_dropouts': hidden_dropouts, 'feat_seg_bin': feat_seg_bin, 'feat_only_bin': feat_only_bin,
              'pred_seg_bin': pred_seg_bin, 'add_pred': add_pred, 'scale_n': scale_n, 'scope_type': scope_type,
              'bundle_scale': bundle_scale, 'embed_dropout_handler': embed_dropout_handler,
              'seg_dropout_handler': seg_dropout_handler, 'hidden_dropout_handler': hidden_dropout_handler}

    if block_num <= 1:
        btnn, extra_inputs = get_tnn_block(0, inputs, **params)
        btnn = compile_func(inputs, btnn, extra_inputs=extra_inputs, init_lr=init_lr)
    else:
        if prev_block_weight_files is None:
            outputs = []
            prev_output = None
            for i in range(block_num):
                params['pre_output'] = prev_output
                params['extra_inputs'] = extra_inputs
                cur_output, extra_inputs = get_tnn_block(i, **params)
                outputs.append(cur_output)
                prev_output = cur_output
            btnn = compile_func(inputs, outputs, extra_inputs=extra_inputs, loss_weights=[1 / block_num] * block_num,
                                init_lr=init_lr)
        else:
            btnn = get_btnn_model(
                x, y, get_output, compile_func, cat_in_dims, cat_out_dims, seg_out_dims, num_segs, seg_type,
                seg_x_val_range, block_num - 1, shrink_factor, seg_y_dim, prev_block_weight_files, seg_flag,
                add_seg_src, seg_num_flag, get_extra_layers, embed_dropout, seg_func, seg_dropout, rel_conf,
                get_last_layers, hidden_units, hidden_activation, hidden_dropouts, feat_seg_bin, feat_only_bin,
                pred_seg_bin, add_pred, scale_n, scope_type, bundle_scale, init_lr, embed_dropout_handler,
                seg_dropout_handler, hidden_dropout_handler)
            read_weights(btnn, prev_block_weight_files[block_num - 2])
            for layer in btnn.layers:
                layer.trainable = False

            i = len(inputs)
            inputs = btnn.inputs[:i]
            extra_inputs = btnn.inputs[i:]

            outputs = btnn.outputs
            prev_output = outputs[-1]

            params['extra_inputs'] = extra_inputs
            params['pre_output'] = prev_output
            cur_output, extra_inputs = get_tnn_block(block_num - 1, inputs, **params)
            outputs.append(cur_output)

            btnn = compile_func(inputs, outputs, extra_inputs=extra_inputs, loss_weights=[1 / block_num] * block_num,
                                init_lr=init_lr)

    return btnn
