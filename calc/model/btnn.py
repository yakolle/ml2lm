from calc.model.tnn import *


def get_btnn_model(x, y, get_output=get_simple_linear_output, compile_func=compile_default_mse_output, cat_in_dims=None,
                   cat_out_dims=None, seg_out_dims=None, num_segs=None, seg_type=0, seg_x_val_range=(0, 1), block_num=3,
                   shrink_factor=1.0, seg_y_dim=50, prev_block_weight_files=None, use_fm=False, seg_flag=True,
                   add_seg_src=True, seg_num_flag=True, get_extra_layers=None, embed_dropout=0.2, seg_dropout=0.2,
                   fm_dim=320, fm_dropout=0.2, fm_activation=None, hidden_units=320, hidden_dropout=0.2):
    oh_input = Input(shape=[x['ohs'].shape[1]], name='ohs') if 'ohs' in x else None
    cat_input = Input(shape=[x['cats'].shape[1]], name='cats') if 'cats' in x else None
    seg_input = Input(shape=[x['segs'].shape[1]], name='segs') if 'segs' in x else None
    num_input = Input(shape=[x['nums'].shape[1]], name='nums') if 'nums' in x else None
    seg_y_val_range = (np.min(y), np.max(y))

    if block_num <= 1:
        btnn = get_tnn_block(0, get_output=get_output, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input,
                             num_input=num_input, cat_in_dims=cat_in_dims, cat_out_dims=cat_out_dims,
                             seg_out_dims=seg_out_dims, num_segs=num_segs, seg_type=seg_type,
                             seg_x_val_range=seg_x_val_range, use_fm=use_fm, seg_flag=seg_flag, add_seg_src=add_seg_src,
                             seg_num_flag=seg_num_flag, x=x, get_extra_layers=get_extra_layers,
                             embed_dropout=embed_dropout, seg_dropout=seg_dropout, fm_dim=fm_dim, fm_dropout=fm_dropout,
                             fm_activation=fm_activation, hidden_units=hidden_units, hidden_dropout=hidden_dropout)
        btnn = compile_func(btnn, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input, num_input=num_input)
    else:
        if prev_block_weight_files is None:
            outputs = []
            prev_output = None
            for i in range(block_num):
                cur_output = get_tnn_block(
                    i, get_output=get_output, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input,
                    num_input=num_input, pre_output=prev_output, cat_in_dims=cat_in_dims, cat_out_dims=cat_out_dims,
                    seg_out_dims=seg_out_dims, num_segs=num_segs, seg_type=seg_type, seg_x_val_range=seg_x_val_range,
                    seg_y_val_range=seg_y_val_range, seg_y_dim=seg_y_dim, shrink_factor=shrink_factor, use_fm=use_fm,
                    seg_flag=seg_flag, add_seg_src=add_seg_src, seg_num_flag=seg_num_flag, x=x,
                    get_extra_layers=get_extra_layers, embed_dropout=embed_dropout, seg_dropout=seg_dropout,
                    fm_dim=fm_dim, fm_dropout=fm_dropout, fm_activation=fm_activation, hidden_units=hidden_units,
                    hidden_dropout=hidden_dropout)
                outputs.append(cur_output)
                prev_output = cur_output
            btnn = compile_func(outputs, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input,
                                num_input=num_input, loss_weights=[1 / block_num] * block_num)
        else:
            btnn = get_btnn_model(
                x, y, get_output, compile_func, cat_in_dims, cat_out_dims, seg_out_dims, num_segs, seg_type,
                seg_x_val_range, block_num - 1, shrink_factor, seg_y_dim, prev_block_weight_files, use_fm, seg_flag,
                add_seg_src, seg_num_flag, get_extra_layers, embed_dropout, seg_dropout, fm_dim, fm_dropout,
                fm_activation, hidden_units, hidden_dropout)
            read_weights(btnn, prev_block_weight_files[block_num - 2])
            for layer in btnn.layers:
                layer.trainable = False

            i = 0
            if oh_input is not None:
                oh_input = btnn.inputs[i]
                i += 1
            if cat_input is not None:
                cat_input = btnn.inputs[i]
                i += 1
            if seg_input is not None:
                seg_input = btnn.inputs[i]
                i += 1
            if num_input is not None:
                num_input = btnn.inputs[i]

            outputs = btnn.outputs
            prev_output = outputs[-1]
            cur_output = get_tnn_block(
                block_num - 1, get_output=get_output, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input,
                num_input=num_input, pre_output=prev_output, cat_in_dims=cat_in_dims, cat_out_dims=cat_out_dims,
                seg_out_dims=seg_out_dims, num_segs=num_segs, seg_type=seg_type, seg_x_val_range=seg_x_val_range,
                seg_y_val_range=seg_y_val_range, seg_y_dim=seg_y_dim, shrink_factor=shrink_factor, use_fm=use_fm,
                seg_flag=seg_flag, add_seg_src=add_seg_src, seg_num_flag=seg_num_flag, x=x,
                get_extra_layers=get_extra_layers, embed_dropout=embed_dropout, seg_dropout=seg_dropout, fm_dim=fm_dim,
                fm_dropout=fm_dropout, fm_activation=fm_activation, hidden_units=hidden_units,
                hidden_dropout=hidden_dropout)
            outputs.append(cur_output)

            btnn = compile_func(outputs, oh_input=oh_input, cat_input=cat_input, seg_input=seg_input,
                                num_input=num_input, loss_weights=[1 / block_num] * block_num)

    return btnn
