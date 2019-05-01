from keras import backend as bk, initializers, regularizers, constraints, activations
from keras.engine.topology import Layer, InputSpec


def labs(x):
    return bk.log(1 + bk.abs(x))


def lrelu(x):
    return bk.log(1 + bk.relu(x))


def lsm(x):
    return bk.log(1 + bk.sigmoid(x))


class FMLayer(Layer):
    def __init__(self, factor_rank, dist_func=bk.relu, exclude_self=True, rel_types='d', activation='relu',
                 use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMLayer, self).__init__(**kwargs)
        self.supports_masking = True

        self.factor_rank = factor_rank

        self.dist_func = dist_func
        self.exclude_self = exclude_self
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

        self.val_rel = None
        self.dist_rel = None
        self.ent_rel = None
        self.rel_map = {'v': self.val_rel, 'd': self.dist_rel, 'e': self.ent_rel}

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if 'v' in self.rel_types:
            self.val_rel = self.add_weight(shape=(input_dim, self.factor_rank), initializer=self.kernel_initializer,
                                           name='val_rel', regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        else:
            self.val_rel = None
        if 'd' in self.rel_types:
            self.dist_rel = self.add_weight(shape=(input_dim, self.factor_rank), initializer=self.kernel_initializer,
                                            name='dist_rel', regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
        else:
            self.dist_rel = None
        if 'e' in self.rel_types:
            self.ent_rel = self.add_weight(shape=(input_dim, self.factor_rank), initializer=self.kernel_initializer,
                                           name='ent_rel', regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        else:
            self.ent_rel = None
        self.rel_map.update({'v': self.val_rel, 'd': self.dist_rel, 'e': self.ent_rel})

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.factor_rank,), initializer=self.bias_initializer, name='bias',
                                        regularizer=self.bias_regularizer, constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def _call(self, rel_type, inputs, kernel):
        rel, self_rel = None, None

        kernel_2 = bk.square(kernel) if self.exclude_self else None
        val_dot, dist, dist_dot = None, None, None
        val_rel, dist_rel = None, None

        if rel_type in 've':
            val_dot = bk.dot(inputs, kernel)
            val_rel = bk.square(val_dot)
            if 'v' == rel_type:
                rel = val_rel
                if self.exclude_self:
                    self_rel = bk.dot(bk.square(inputs), kernel_2)
        if rel_type in 'de':
            dist = self.dist_func(inputs)
            dist_dot = bk.dot(dist, kernel)
            dist_rel = bk.square(dist_dot)
            if 'd' == rel_type:
                rel = dist_rel
                if self.exclude_self:
                    self_rel = bk.dot(bk.square(dist), kernel_2)
        if 'e' == rel_type:
            dist_val_rel = bk.square(val_dot + dist_dot)
            rel = dist_val_rel - val_rel - dist_rel
            if self.exclude_self:
                self_rel = 2 * bk.dot(inputs * dist, kernel_2)

        output = rel - self_rel if self.exclude_self else rel
        return output

    def call(self, inputs, **kwargs):
        outputs = [self._call(rel_type, inputs, self.rel_map[rel_type]) for rel_type in self.rel_types]
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
            'exclude_self': self.exclude_self,
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
