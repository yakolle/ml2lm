import numpy as np
from keras import backend as bk
from keras.engine.topology import Layer, InputSpec
from keras.initializers import Constant
from keras.layers import Dense

from ml2lm.calc.model.units.activations import seu


class SegRightAngleLayer(Layer):
    def __init__(self, seg_num, input_val_range=(0, 1), seg_func=seu, pos_fixed=True, seg_width_fixed=False,
                 include_seg_bin=False, only_seg_bin=False, **kwargs):
        assert seg_num >= 2

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(SegRightAngleLayer, self).__init__(**kwargs)
        self.seg_num = seg_num

        self.left_pos = None
        self.middle_pos = None
        self.right_pos = None
        self.middle_seg_width = None
        self.seg_width = None

        self.input_val_range = input_val_range
        self.seg_func = seg_func
        self.pos_fixed = pos_fixed
        self.seg_width_fixed = seg_width_fixed

        self.include_seg_bin = include_seg_bin
        self.only_seg_bin = only_seg_bin
        if self.only_seg_bin:
            self.include_seg_bin = True
            self.seg_width_fixed = True

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        assert 1 == input_shape[-1]

        self.seg_width = (self.input_val_range[1] - self.input_val_range[0]) / self.seg_num
        left_pos = self.input_val_range[0] + self.seg_width
        right_pos = self.input_val_range[1] - self.seg_width

        self.left_pos = self.add_weight(shape=(1,), initializer=Constant(value=left_pos), name='left_pos',
                                        trainable=not self.pos_fixed)
        if self.seg_num > 2:
            middle_pos = np.linspace(left_pos, right_pos - self.seg_width, self.seg_num - 2)
            self.middle_pos = self.add_weight(shape=(self.seg_num - 2,), name='middle_pos',
                                              initializer=Constant(value=middle_pos), trainable=not self.pos_fixed)
        else:
            self.middle_pos = None
        self.right_pos = self.add_weight(shape=(1,), initializer=Constant(value=right_pos), name='right_pos',
                                         trainable=not self.pos_fixed)

        if self.seg_num > 2 and not self.only_seg_bin:
            self.middle_seg_width = self.add_weight(shape=(self.seg_num - 2,),
                                                    initializer=Constant(value=self.seg_width), name='middle_seg_width',
                                                    trainable=not self.seg_width_fixed)
        else:
            self.middle_seg_width = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: 1})
        self.built = True

    def call(self, inputs, **kwargs):
        outputs = []

        left_out = middle_tmp_out = right_out = None

        if not self.only_seg_bin:
            left_out = self.left_pos - inputs
            middle_tmp_out = None if self.middle_pos is None else self.seg_func(inputs - self.middle_pos)
            middle_out = None if self.middle_pos is None else middle_tmp_out * bk.sign(
                self.middle_pos + self.middle_seg_width - inputs)
            right_out = inputs - self.right_pos

            if self.middle_pos is not None:
                output = bk.concatenate([left_out, middle_out, right_out])
            else:
                output = bk.concatenate([left_out, right_out])
            outputs.append(self.seg_func(output))

        if self.include_seg_bin:
            left_out = self.left_pos - inputs if left_out is None else left_out
            middle_tmp_out = None if self.middle_pos is None else self.seg_func(
                inputs - self.middle_pos) if middle_tmp_out is None else middle_tmp_out
            middle_out = None if self.middle_pos is None else middle_tmp_out * bk.sign(
                self.middle_pos + self.seg_width - inputs)
            right_out = inputs - self.right_pos if right_out is None else right_out

            if self.middle_pos is not None:
                output = bk.concatenate([left_out, middle_out, right_out])
            else:
                output = bk.concatenate([left_out, right_out])
            outputs.append(bk.cast(output > 0, inputs.dtype))

        return bk.concatenate(outputs) if len(outputs) > 1 else outputs[0]

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert 1 == input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.seg_num * (1 + self.include_seg_bin - self.only_seg_bin)
        return tuple(output_shape)

    def get_config(self):
        config = {
            'seg_num': self.seg_num,
            'input_val_range': self.input_val_range,
            'seg_func': self.seg_func,
            'pos_fixed': self.pos_fixed,
            'seg_width_fixed': self.seg_width_fixed,
            'include_seg_bin': self.include_seg_bin,
            'only_seg_bin': self.only_seg_bin
        }
        base_config = super(SegRightAngleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def transform(segment_layer, activation=None, use_bias=False, kernel_initializer='glorot_uniform',
              bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
              kernel_constraint=None, bias_constraint=None):
    return Dense(1, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(segment_layer)
