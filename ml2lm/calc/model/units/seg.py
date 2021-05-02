import numpy as np
from keras import activations
from keras.layers import Dense

from ml2lm.calc.model.units.activations import *


class SegLayer(Layer):
    def get_segs(self, inputs, **kwargs):
        return [inputs]

    def get_segs_size(self):
        return 1

    def call(self, inputs, **kwargs):
        outputs = self.get_segs(inputs, **kwargs)
        return bk.concatenate(outputs) if len(outputs) > 1 else outputs[0]

    @staticmethod
    def calc_seg_out_num(seg_num):
        return seg_num

    @staticmethod
    def calc_seg_num(seg_out_dim):
        return seg_out_dim


class SegTriangleLayer(SegLayer):
    def __init__(self, seg_num, input_val_range=(0, 1), seg_func=seu, win_func=bk.abs, pos_fixed=True,
                 seg_width_fixed=False, include_seg_bin=False, only_seg_bin=False, **kwargs):
        assert seg_num >= 2

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs = {'input_shape': (kwargs['input_dim'],)}
        else:
            kwargs = {}

        super(SegTriangleLayer, self).__init__(**kwargs)
        self.seg_num = seg_num

        self.left_pos = None
        self.middle_pos = None
        self.right_pos = None
        self.middle_seg_width = None
        self.seg_width = None

        self.input_val_range = input_val_range
        self.seg_func = seg_func
        self.win_func = win_func
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

        self.left_pos = self.add_weight(shape=(1,), initializer=Constant(value=left_pos), name=f'{self.name}/left_pos',
                                        trainable=not self.pos_fixed)
        if self.seg_num > 2:
            middle_pos = np.linspace(left_pos, right_pos, self.seg_num - 1)
            self.middle_pos = self.add_weight(shape=(self.seg_num - 1,), name=f'{self.name}/middle_pos',
                                              initializer=Constant(value=middle_pos), trainable=not self.pos_fixed)
        else:
            self.middle_pos = None
        self.right_pos = self.add_weight(shape=(1,), initializer=Constant(value=right_pos),
                                         name=f'{self.name}/right_pos', trainable=not self.pos_fixed)

        if self.seg_num > 2 and not self.only_seg_bin:
            self.middle_seg_width = self.add_weight(shape=(self.seg_num - 1,),
                                                    initializer=Constant(value=self.seg_width),
                                                    name=f'{self.name}/middle_seg_width',
                                                    trainable=not self.seg_width_fixed)
        else:
            self.middle_seg_width = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: 1})
        self.built = True

    def get_segs(self, inputs, **kwargs):
        outputs = []

        left_out = middle_tmp_out = right_out = None

        if not self.only_seg_bin:
            left_out = self.left_pos - inputs
            middle_tmp_out = None if self.middle_pos is None else -self.win_func(inputs - self.middle_pos)
            middle_out = None if self.middle_pos is None else middle_tmp_out + self.middle_seg_width
            right_out = inputs - self.right_pos

            if self.middle_pos is not None:
                output = bk.concatenate([left_out, middle_out, right_out])
            else:
                output = bk.concatenate([left_out, right_out])
            outputs.append(self.seg_func(output))

        if self.include_seg_bin:
            left_out = self.left_pos - inputs if left_out is None else left_out
            middle_tmp_out = None if self.middle_pos is None else -self.win_func(
                inputs - self.middle_pos) if middle_tmp_out is None else middle_tmp_out
            middle_out = None if self.middle_pos is None else middle_tmp_out + self.seg_width
            right_out = inputs - self.right_pos if right_out is None else right_out

            if self.middle_pos is not None:
                output = bk.concatenate([left_out, middle_out, right_out])
            else:
                output = bk.concatenate([left_out, right_out])
            outputs.append(bk.cast(output > 0, inputs.dtype))

        return outputs

    def get_segs_size(self):
        return 1 + self.include_seg_bin - self.only_seg_bin

    @staticmethod
    def calc_seg_out_num(seg_num):
        return seg_num + (seg_num > 2)

    @staticmethod
    def calc_seg_num(seg_out_dim):
        return seg_out_dim - (seg_out_dim > 2)

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert 1 == input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.calc_seg_out_num(self.seg_num) * (1 + self.include_seg_bin - self.only_seg_bin)
        return tuple(output_shape)

    def get_config(self):
        config = {
            'seg_num': self.seg_num,
            'input_val_range': self.input_val_range,
            'seg_func': self.seg_func,
            'win_func': self.win_func,
            'pos_fixed': self.pos_fixed,
            'seg_width_fixed': self.seg_width_fixed,
            'include_seg_bin': self.include_seg_bin,
            'only_seg_bin': self.only_seg_bin
        }
        base_config = super(SegTriangleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SegRightAngleLayer(SegLayer):
    def __init__(self, seg_num, input_val_range=(0, 1), seg_func=seu, dive_height=20., pos_fixed=True,
                 seg_width_fixed=False, include_seg_bin=False, only_seg_bin=False, **kwargs):
        assert seg_num >= 2

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs = {'input_shape': (kwargs['input_dim'],)}
        else:
            kwargs = {}

        super(SegRightAngleLayer, self).__init__(**kwargs)
        self.seg_num = seg_num

        self.left_pos = None
        self.middle_pos = None
        self.right_pos = None
        self.middle_seg_width = None
        self.seg_width = None

        self.input_val_range = input_val_range
        self.seg_func = seg_func
        self.dive_height = dive_height
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

        self.left_pos = self.add_weight(shape=(1,), initializer=Constant(value=left_pos), name=f'{self.name}/left_pos',
                                        trainable=not self.pos_fixed)
        if self.seg_num > 2:
            middle_pos = np.linspace(left_pos, right_pos - self.seg_width, self.seg_num - 2)
            self.middle_pos = self.add_weight(shape=(self.seg_num - 2,), name=f'{self.name}/middle_pos',
                                              initializer=Constant(value=middle_pos), trainable=not self.pos_fixed)
        else:
            self.middle_pos = None
        self.right_pos = self.add_weight(shape=(1,), initializer=Constant(value=right_pos),
                                         name=f'{self.name}/right_pos', trainable=not self.pos_fixed)

        if self.seg_num > 2 and not self.only_seg_bin:
            self.middle_seg_width = self.add_weight(shape=(self.seg_num - 2,),
                                                    initializer=Constant(value=self.seg_width),
                                                    name=f'{self.name}/middle_seg_width',
                                                    trainable=not self.seg_width_fixed)
        else:
            self.middle_seg_width = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: 1})
        self.built = True

    def get_segs(self, inputs, **kwargs):
        outputs = []

        left_out = middle_tmp_out = right_out = None

        if not self.only_seg_bin:
            left_out = self.left_pos - inputs
            middle_tmp_out = None if self.middle_pos is None else self.seg_func(inputs - self.middle_pos)
            middle_out = None if self.middle_pos is None else middle_tmp_out - self.dive_height * self.seg_func(
                inputs - self.middle_pos - self.middle_seg_width)
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

        return outputs

    def get_segs_size(self):
        return 1 + self.include_seg_bin - self.only_seg_bin

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert 1 == input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.calc_seg_out_num(self.seg_num) * (1 + self.include_seg_bin - self.only_seg_bin)
        return tuple(output_shape)

    def get_config(self):
        config = {
            'seg_num': self.seg_num,
            'input_val_range': self.input_val_range,
            'seg_func': self.seg_func,
            'dive_height': self.dive_height,
            'pos_fixed': self.pos_fixed,
            'seg_width_fixed': self.seg_width_fixed,
            'include_seg_bin': self.include_seg_bin,
            'only_seg_bin': self.only_seg_bin
        }
        base_config = super(SegRightAngleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BinSegLayer(SegLayer):
    def __init__(self, seg_num, input_val_range=(0, 1), seg_func=seu, pos_fixed=True, include_seg_bin=False,
                 only_seg_bin=False, **kwargs):
        assert seg_num >= 2

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs = {'input_shape': (kwargs['input_dim'],)}
        else:
            kwargs = {}

        super(BinSegLayer, self).__init__(**kwargs)
        self.seg_num = seg_num
        self.input_val_range = input_val_range
        self.seg_func = seg_func
        self.pos_fixed = pos_fixed
        self.include_seg_bin = include_seg_bin
        self.only_seg_bin = only_seg_bin
        if self.only_seg_bin:
            self.include_seg_bin = True
            self.pos_fixed = True

        self.left_pos = None
        self.right_pos = None

    def build(self, input_shape):
        assert len(input_shape) >= 2
        assert 1 == input_shape[-1]

        pos = np.linspace(self.input_val_range[0], self.input_val_range[1], self.seg_num + 1)
        self.left_pos = self.add_weight(shape=(self.seg_num,), name=f'{self.name}/left_pos',
                                        initializer=Constant(value=pos[1:]), trainable=not self.pos_fixed)
        self.right_pos = self.add_weight(shape=(self.seg_num,), name=f'{self.name}/right_pos',
                                         initializer=Constant(value=pos[:-1]), trainable=not self.pos_fixed)
        self.built = True

    def get_segs(self, inputs, **kwargs):
        outputs = []

        left_out = self.left_pos - inputs
        right_out = inputs - self.right_pos
        if not self.only_seg_bin:
            outputs.append(self.seg_func(left_out))
            outputs.append(self.seg_func(right_out))
        if self.include_seg_bin:
            outputs.append(bk.cast(left_out > 0, inputs.dtype))
            outputs.append(bk.cast(right_out > 0, inputs.dtype))

        return outputs

    def get_segs_size(self):
        return (1 + self.include_seg_bin - self.only_seg_bin) * 2

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert 1 == input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 2 * self.calc_seg_out_num(self.seg_num) * (1 + self.include_seg_bin - self.only_seg_bin)
        return tuple(output_shape)

    def get_config(self):
        config = {
            'seg_num': self.seg_num,
            'input_val_range': self.input_val_range,
            'seg_func': self.seg_func,
            'pos_fixed': self.pos_fixed,
            'include_seg_bin': self.include_seg_bin,
            'only_seg_bin': self.only_seg_bin
        }
        base_config = super(BinSegLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WaveletWrapper(SegLayer):
    def __init__(self, seg_num, input_val_range=(0, 1), seg_func=seu, pos_fixed=True, seg_width_fixed=False,
                 include_seg_bin=False, only_seg_bin=False, seg_class=SegTriangleLayer, scale_n=3, scope_type='global',
                 bundle_scale=True, **kwargs):
        assert seg_num >= 2

        wl_args = {}
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            wl_args['input_shape'] = kwargs['input_shape']
        kwargs.update({'input_val_range': input_val_range, 'seg_func': seg_func, 'pos_fixed': pos_fixed,
                       'seg_width_fixed': seg_width_fixed, 'include_seg_bin': include_seg_bin,
                       'only_seg_bin': only_seg_bin})

        super(WaveletWrapper, self).__init__(**wl_args)
        self.seg_class = seg_class
        self.scale_n = scale_n
        self.scope_type = scope_type
        self.bundle_scale = bundle_scale

        self.base_seg_layer = self.seg_class(seg_num, name=f'seg0', **kwargs)
        self.scales = []
        self.seg_layers = [self.seg_class(self.seg_class.calc_seg_num(self.seg_class.calc_seg_out_num(
            seg_num) * 2 ** i), name=f'seg{i}', **kwargs) for i in range(1, scale_n + 1)]
        bundle_num = int(np.ceil(np.log2(seg_num)))
        self.bundled_seg_layers = [self.seg_class(2 ** i, name=f'seg-{bundle_num - i}', **kwargs) for i in
                                   range(1, bundle_num)] if self.bundle_scale else []

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        self.base_seg_layer.build(input_shape)
        for i, seg_layer in enumerate(self.seg_layers):
            seg_layer.build(input_shape)

            cur_scales = []
            for j in range(seg_layer.get_segs_size()):
                cur_scales.append(self.add_weight(name=f'scale_{i}_{j}', initializer=Constant(value=1.),
                                                  shape=(seg_layer.calc_seg_out_num(seg_layer.seg_num),)))
            self.scales.append(cur_scales)
        for seg_layer in self.bundled_seg_layers:
            seg_layer.build(input_shape)

    def get_segs(self, inputs, **kwargs):
        outpus = self.base_seg_layer.get_segs(inputs, **kwargs)

        for i in range(self.scale_n):
            sub_seg_num = 2 ** (i + 1)
            cur_outputs = self.seg_layers[i].get_segs(inputs, **kwargs)
            seg_out_num = self.base_seg_layer.calc_seg_out_num(self.base_seg_layer.seg_num)
            sample_shape, sample_axis = ([-1, sub_seg_num, seg_out_num], 1) if 'global' == self.scope_type else (
                [-1, seg_out_num, sub_seg_num], -1)
            cur_outputs = [bk.max(bk.reshape(cur_out * cur_scale, sample_shape), axis=sample_axis) for
                           cur_out, cur_scale in zip(cur_outputs, self.scales[i])]
            outpus += cur_outputs
        for seg_layer in self.bundled_seg_layers:
            outpus += seg_layer.get_segs(inputs, **kwargs)

        return outpus

    def get_segs_size(self):
        segs_size = self.base_seg_layer.get_segs_size()
        for i in range(self.scale_n):
            segs_size += self.seg_layers[i].get_segs_size()
        for seg_layer in self.bundled_seg_layers:
            segs_size += seg_layer.get_segs_size()
        return segs_size

    @staticmethod
    def calc_seg_out_num(seg_num):
        return -1.

    @staticmethod
    def calc_seg_num(seg_out_dim):
        return -1

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert 1 == input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = (self.scale_n + 1) * self.base_seg_layer.compute_output_shape(input_shape)[-1]
        for seg_layer in self.bundled_seg_layers:
            output_shape[-1] += seg_layer.compute_output_shape(input_shape)[-1]
        return tuple(output_shape)

    def get_config(self):
        config = {
            'seg_class': self.seg_class.__name__,
            'scale_n': self.scale_n,
            'scope_type': self.scope_type,
            'bundle_scale': self.bundle_scale
        }
        base_config = self.base_seg_layer.get_config()
        return dict(list(base_config.items()) + list(config.items()))


def transform(segment_layer, activation=None, use_bias=False, kernel_initializer='glorot_uniform',
              bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
              kernel_constraint=None, bias_constraint=None):
    return Dense(1, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(segment_layer)


class SegAbsWindowLayer(Layer):
    def __init__(self, seg_num, seg_func=seu, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform', seg_width_initializer='ones', seg_height_initializer='ones',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, seg_width_regularizer=None,
                 seg_height_regularizer=None, kernel_constraint=None, bias_constraint=None, seg_width_constraint=None,
                 seg_height_constraint=None, **kwargs):
        assert seg_num >= 2

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(SegAbsWindowLayer, self).__init__(**kwargs)
        self.seg_num = seg_num
        self.seg_func = seg_func
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.seg_width_initializer = initializers.get(seg_width_initializer)
        self.seg_height_initializer = initializers.get(seg_height_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.seg_width_regularizer = regularizers.get(seg_width_regularizer)
        self.seg_height_regularizer = regularizers.get(seg_height_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.seg_width_constraint = constraints.get(seg_width_constraint)
        self.seg_height_constraint = constraints.get(seg_height_constraint)

        self.left_kernel = None
        self.middle_kernel = None
        self.right_kernel = None
        self.left_bias = None
        self.middle_bias = None
        self.right_bias = None
        self.middle_seg_width = None
        self.middle_seg_height = None

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.left_kernel = self.add_weight(shape=(input_dim, 1),
                                           initializer=self.kernel_initializer,
                                           name=f'{self.name}/left_kernel',
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        if self.seg_num > 2:
            self.middle_kernel = self.add_weight(shape=(input_dim, self.seg_num - 2),
                                                 initializer=self.kernel_initializer,
                                                 name=f'{self.name}/middle_kernel',
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)
        else:
            self.middle_kernel = None
        self.right_kernel = self.add_weight(shape=(input_dim, 1),
                                            initializer=self.kernel_initializer,
                                            name=f'{self.name}/right_kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)

        if self.use_bias:
            self.left_bias = self.add_weight(shape=(1,),
                                             initializer=self.bias_initializer,
                                             name=f'{self.name}/left_bias',
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)
            if self.seg_num > 2:
                self.middle_bias = self.add_weight(shape=(self.seg_num - 2,),
                                                   initializer=self.bias_initializer,
                                                   name=f'{self.name}/middle_bias',
                                                   regularizer=self.bias_regularizer,
                                                   constraint=self.bias_constraint)
            else:
                self.middle_bias = None
            self.right_bias = self.add_weight(shape=(1,),
                                              initializer=self.bias_initializer,
                                              name=f'{self.name}/right_bias',
                                              regularizer=self.bias_regularizer,
                                              constraint=self.bias_constraint)
        else:
            self.left_bias = None
            self.middle_bias = None
            self.right_bias = None

        if self.seg_num > 2:
            self.middle_seg_width = self.add_weight(shape=(self.seg_num - 2,),
                                                    initializer=self.seg_width_initializer,
                                                    name=f'{self.name}/middle_seg_width',
                                                    regularizer=self.seg_width_regularizer,
                                                    constraint=self.seg_width_constraint)
        else:
            self.middle_seg_width = None

        if self.seg_num > 2:
            self.middle_seg_height = self.add_weight(shape=(self.seg_num - 2,),
                                                     initializer=self.seg_height_initializer,
                                                     name=f'{self.name}/middle_seg_height',
                                                     regularizer=self.seg_height_regularizer,
                                                     constraint=self.seg_height_constraint)
        else:
            self.middle_seg_height = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        left_out = bk.dot(inputs, self.left_kernel)
        middle_out = None
        if self.middle_kernel is not None:
            middle_out = bk.dot(inputs, self.middle_kernel)
        right_out = bk.dot(inputs, self.right_kernel)

        if self.use_bias:
            left_out = bk.bias_add(left_out, self.left_bias)
            if self.middle_bias is not None:
                middle_out = bk.bias_add(middle_out, self.middle_bias)
            right_out = bk.bias_add(right_out, self.right_bias)

        if self.activation is not None:
            left_out = self.activation(left_out)
            if self.middle_kernel is not None:
                middle_out = self.activation(middle_out)
            right_out = self.activation(right_out)

        left_out = -left_out
        if self.middle_kernel is not None:
            middle_out = -bk.abs(middle_out) * self.middle_seg_width + self.middle_seg_height

        if self.middle_kernel is not None:
            output = bk.concatenate([left_out, middle_out, right_out])
        else:
            output = bk.concatenate([left_out, right_out])
        return self.seg_func(output)

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.seg_num
        return tuple(output_shape)

    def get_config(self):
        config = {
            'seg_num': self.seg_num,
            'seg_func': self.seg_func,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'seg_width_initializer': initializers.serialize(self.seg_width_initializer),
            'seg_height_initializer': initializers.serialize(self.seg_height_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'seg_width_regularizer': regularizers.serialize(self.seg_width_regularizer),
            'seg_height_regularizer': regularizers.serialize(self.seg_height_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'seg_width_constraint': constraints.serialize(self.seg_width_constraint),
            'seg_height_constraint': constraints.serialize(self.seg_height_constraint)
        }
        base_config = super(SegAbsWindowLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SegSquareWindowLayer(Layer):
    def __init__(self, seg_num, seg_func=seu, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform', seg_width_initializer='ones', seg_height_initializer='ones',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, seg_width_regularizer=None,
                 seg_height_regularizer=None, kernel_constraint=None, bias_constraint=None, seg_width_constraint=None,
                 seg_height_constraint=None, **kwargs):
        assert seg_num >= 2

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(SegSquareWindowLayer, self).__init__(**kwargs)
        self.seg_num = seg_num
        self.seg_func = seg_func
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.seg_width_initializer = initializers.get(seg_width_initializer)
        self.seg_height_initializer = initializers.get(seg_height_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.seg_width_regularizer = regularizers.get(seg_width_regularizer)
        self.seg_height_regularizer = regularizers.get(seg_height_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.seg_width_constraint = constraints.get(seg_width_constraint)
        self.seg_height_constraint = constraints.get(seg_height_constraint)

        self.left_kernel = None
        self.middle_kernel = None
        self.right_kernel = None
        self.left_bias = None
        self.middle_bias = None
        self.right_bias = None
        self.middle_seg_width = None
        self.middle_seg_height = None

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.left_kernel = self.add_weight(shape=(input_dim, 1),
                                           initializer=self.kernel_initializer,
                                           name=f'{self.name}/left_kernel',
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        if self.seg_num > 2:
            self.middle_kernel = self.add_weight(shape=(input_dim, self.seg_num - 2),
                                                 initializer=self.kernel_initializer,
                                                 name=f'{self.name}/middle_kernel',
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)
        else:
            self.middle_kernel = None
        self.right_kernel = self.add_weight(shape=(input_dim, 1),
                                            initializer=self.kernel_initializer,
                                            name=f'{self.name}/right_kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)

        if self.use_bias:
            self.left_bias = self.add_weight(shape=(1,),
                                             initializer=self.bias_initializer,
                                             name=f'{self.name}/left_bias',
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)
            if self.seg_num > 2:
                self.middle_bias = self.add_weight(shape=(self.seg_num - 2,),
                                                   initializer=self.bias_initializer,
                                                   name=f'{self.name}/middle_bias',
                                                   regularizer=self.bias_regularizer,
                                                   constraint=self.bias_constraint)
            else:
                self.middle_bias = None
            self.right_bias = self.add_weight(shape=(1,),
                                              initializer=self.bias_initializer,
                                              name=f'{self.name}/right_bias',
                                              regularizer=self.bias_regularizer,
                                              constraint=self.bias_constraint)
        else:
            self.left_bias = None
            self.middle_bias = None
            self.right_bias = None

        if self.seg_num > 2:
            self.middle_seg_width = self.add_weight(shape=(self.seg_num - 2,),
                                                    initializer=self.seg_width_initializer,
                                                    name=f'{self.name}/middle_seg_width',
                                                    regularizer=self.seg_width_regularizer,
                                                    constraint=self.seg_width_constraint)
        else:
            self.middle_seg_width = None

        if self.seg_num > 2:
            self.middle_seg_height = self.add_weight(shape=(self.seg_num - 2,),
                                                     initializer=self.seg_height_initializer,
                                                     name=f'{self.name}/middle_seg_height',
                                                     regularizer=self.seg_height_regularizer,
                                                     constraint=self.seg_height_constraint)
        else:
            self.middle_seg_height = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        left_out = bk.dot(inputs, self.left_kernel)
        middle_out = None
        if self.middle_kernel is not None:
            middle_out = bk.dot(inputs, self.middle_kernel)
        right_out = bk.dot(inputs, self.right_kernel)

        if self.use_bias:
            left_out = bk.bias_add(left_out, self.left_bias)
            if self.middle_bias is not None:
                middle_out = bk.bias_add(middle_out, self.middle_bias)
            right_out = bk.bias_add(right_out, self.right_bias)

        if self.activation is not None:
            left_out = self.activation(left_out)
            if self.middle_kernel is not None:
                middle_out = self.activation(middle_out)
            right_out = self.activation(right_out)

        left_out = -left_out
        if self.middle_kernel is not None:
            middle_out = -middle_out * middle_out * self.middle_seg_width + self.middle_seg_height

        if self.middle_kernel is not None:
            output = bk.concatenate([left_out, middle_out, right_out])
        else:
            output = bk.concatenate([left_out, right_out])
        return self.seg_func(output)

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.seg_num
        return tuple(output_shape)

    def get_config(self):
        config = {
            'seg_num': self.seg_num,
            'seg_func': self.seg_func,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'seg_width_initializer': initializers.serialize(self.seg_width_initializer),
            'seg_height_initializer': initializers.serialize(self.seg_height_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'seg_width_regularizer': regularizers.serialize(self.seg_width_regularizer),
            'seg_height_regularizer': regularizers.serialize(self.seg_height_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'seg_width_constraint': constraints.serialize(self.seg_width_constraint),
            'seg_height_constraint': constraints.serialize(self.seg_height_constraint)
        }
        base_config = super(SegSquareWindowLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def project(segment_layer, project_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None):
    return Dense(project_dim, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(segment_layer)
