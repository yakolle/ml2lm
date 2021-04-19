from keras import backend as bk, initializers, regularizers, constraints, activations
from keras.engine.topology import Layer, InputSpec


def labs(x):
    return bk.log(1 + bk.abs(x))


def lrelu(x):
    return bk.log(1 + bk.relu(x))


def lsm(x):
    return bk.log(1 + bk.sigmoid(x))


class FMLayer(Layer):
    def __init__(self, factor_rank, dist_func=bk.relu, exclude_selves=(False,), rel_types='d', activation=None,
                 use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMLayer, self).__init__(**kwargs)
        self.supports_masking = True

        self.factor_rank = factor_rank

        self.dist_func = dist_func
        self.exclude_selves = exclude_selves
        self.rel_types = rel_types

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.bias = None

        self.rel_map = {}
        self.exclude_self_map = dict(zip(rel_types, self.exclude_selves))

    def _add_kernel(self, input_dim, name=None):
        return self.add_weight(shape=(input_dim, self.factor_rank), initializer=self.kernel_initializer, name=name,
                               regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if 'v' in self.rel_types:
            self.rel_map['v'] = self._add_kernel(input_dim, 'val_rel')
        else:
            self.rel_map['v'] = None
        if 'd' in self.rel_types:
            self.rel_map['d'] = self._add_kernel(input_dim, 'dist_rel')
        else:
            self.rel_map['d'] = None
        if 'e' in self.rel_types:
            self.rel_map['e'] = self._add_kernel(input_dim, 'ent_rel')
        else:
            self.rel_map['e'] = None

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.factor_rank,), initializer=self.bias_initializer, name='bias',
                                        regularizer=self.bias_regularizer, constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def _call(self, rel_type, inputs):
        rel, self_rel = None, None

        kernel = self.rel_map[rel_type]
        exclude_self = self.exclude_self_map[rel_type]
        kernel_2 = bk.square(kernel) if exclude_self or 'e' != rel_type else None
        val_dot, dist, dist_dot = None, None, None
        val_rel, dist_rel = None, None

        if rel_type in 've':
            val_dot = bk.dot(inputs, kernel)
            val_rel = bk.square(val_dot)
            if 'v' == rel_type:
                rel = val_rel
                self_rel = bk.dot(bk.square(inputs), kernel_2)
        if rel_type in 'de':
            dist = self.dist_func(inputs)
            dist_dot = bk.dot(dist, kernel)
            dist_rel = bk.square(dist_dot)
            if 'd' == rel_type:
                rel = dist_rel
                self_rel = bk.dot(bk.square(dist), kernel_2)
        if 'e' == rel_type:
            dist_val_rel = bk.square(val_dot + dist_dot)
            rel = dist_val_rel - val_rel - dist_rel
            if exclude_self:
                self_rel = 2 * bk.dot(inputs * dist, kernel_2)

        output = rel - self_rel if exclude_self else rel + self_rel if 'e' != rel_type else rel
        return output

    def call(self, inputs, **kwargs):
        outputs = [self._call(rel_type, inputs) for rel_type in self.rel_types]
        output = bk.concatenate(outputs) if len(outputs) > 1 else outputs[0]

        if self.use_bias:
            output = bk.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.factor_rank * len(self.rel_types)
        return tuple(output_shape)

    def get_config(self):
        config = {
            'factor_rank': self.factor_rank,
            'dist_func': self.dist_func,
            'exclude_selves': self.exclude_selves,
            'rel_types': self.rel_types,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(FMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def rel_mul(x, y):
    return x * y


def rel_div(epsilon=1e-3):
    def _rel_div(x, y):
        neg_flag = bk.cast(y < 0, y.dtype)
        clip_flag = bk.cast(bk.abs(y) > epsilon, y.dtype)
        return x / (clip_flag * y + (1. - clip_flag) * epsilon * (1. - 2. * neg_flag))

    return _rel_div


class BiCrossLayer(Layer):
    def __init__(self, factor_rank, trans_func=bk.relu, op_func=rel_mul, rel_types='d', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(BiCrossLayer, self).__init__(**kwargs)
        self.supports_masking = True

        self.factor_rank = factor_rank

        self.trans_func = trans_func
        self.op_func = op_func
        self.rel_types = rel_types

        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.rel_map = {}
        self.bias_map = {}

    def _add_kernel(self, input_dim, name=None):
        return self.add_weight(shape=(input_dim, self.factor_rank), initializer=self.kernel_initializer, name=name,
                               regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)

    def _add_bias(self, name=None):
        return self.add_weight(shape=(self.factor_rank,), initializer=self.bias_initializer, name=name,
                               regularizer=self.bias_regularizer, constraint=self.bias_constraint)

    def build(self, input_shape):
        input_dim1, input_dim2 = input_shape[0][-1], input_shape[1][-1]

        if 'v' in self.rel_types:
            self.rel_map['v'] = (self._add_kernel(input_dim1, 'val_rel1'), self._add_kernel(input_dim2, 'val_rel2'))
            if self.use_bias:
                self.bias_map['v'] = (self._add_bias('val_bias1'), self._add_bias('val_bias2'))
            else:
                self.bias_map['v'] = None
        else:
            self.rel_map['v'] = self.bias_map['v'] = None

        if 'd' in self.rel_types:
            self.rel_map['d'] = (self._add_kernel(input_dim1, 'dist_rel1'), self._add_kernel(input_dim2, 'dist_rel2'))
            if self.use_bias:
                self.bias_map['d'] = (self._add_bias('dist_bias1'), self._add_bias('dist_bias2'))
            else:
                self.bias_map['d'] = None
        else:
            self.rel_map['d'] = self.bias_map['d'] = None

        if 'e' in self.rel_types:
            self.rel_map['e'] = (self._add_kernel(input_dim1, 'ent_rel1'), self._add_kernel(input_dim2, 'ent_rel2'))
            if self.use_bias:
                self.bias_map['e'] = (self._add_bias('ent_bias1'), self._add_bias('ent_bias2'))
            else:
                self.bias_map['e'] = None
        else:
            self.rel_map['e'] = self.bias_map['e'] = None

        self.built = True

    def _call(self, rel_type, inputs):
        kernel1, kernel2 = self.rel_map[rel_type]

        inputs1, inputs2 = inputs
        if 'd' == rel_type:
            inputs1, inputs2 = self.trans_func(inputs1), self.trans_func(inputs2)
        elif 'e' == rel_type:
            inputs2 = self.trans_func(inputs2)

        if self.use_bias:
            bias1, bias2 = self.bias_map[rel_type]
            output = self.op_func(bk.bias_add(bk.dot(inputs1, kernel1), bias1),
                                  bk.bias_add(bk.dot(inputs2, kernel2), bias2))
        else:
            output = self.op_func(bk.dot(inputs1, kernel1), bk.dot(inputs2, kernel2))

        return output

    def call(self, inputs, **kwargs):
        outputs = [self._call(rel_type, inputs) for rel_type in self.rel_types]
        return bk.concatenate(outputs) if len(outputs) > 1 else outputs[0]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape[-1] = self.factor_rank * len(self.rel_types)
        return tuple(output_shape)

    def get_config(self):
        config = {
            'factor_rank': self.factor_rank,
            'trans_func': self.trans_func,
            'op_func': self.op_func,
            'rel_types': self.rel_types,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(BiCrossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BiRelLayer(BiCrossLayer):
    def build(self, input_shape):
        super(BiRelLayer, self).build([input_shape, input_shape])

    def call(self, inputs, **kwargs):
        return super(BiRelLayer, self).call([inputs, inputs], **kwargs)

    def compute_output_shape(self, input_shape):
        return super(BiRelLayer, self).compute_output_shape([input_shape, input_shape])


class ShadowLayer(Layer):
    def __init__(self, trans_func=lsm, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ShadowLayer, self).__init__(**kwargs)
        self.supports_masking = True

        self.trans_func = trans_func

        self.input_spec = InputSpec(min_ndim=2)

    def call(self, inputs, **kwargs):
        return bk.concatenate([inputs, self.trans_func(bk.stop_gradient(inputs))])

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] *= 2
        return tuple(output_shape)

    def get_config(self):
        config = {'trans_func': self.trans_func}
        base_config = super(ShadowLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
