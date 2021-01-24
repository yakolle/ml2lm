import keras.backend as bk
from keras import initializers, regularizers, constraints
from keras.constraints import Constraint
from keras.engine import Layer, InputSpec
from keras.initializers import Constant


def seu(x):
    return bk.sigmoid(x) * x


def make_lseu(c_val):
    def _lseu(x):
        x1 = bk.sigmoid(x) * x
        x2 = c_val + bk.log(1 + bk.relu(x - c_val))
        return bk.minimum(x1, x2)

    return _lseu


def lseu(x):
    return make_lseu(9.0)(x)


class MinMaxValue(Constraint):
    def __init__(self, min_value=1e-3, max_value=None):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return bk.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


class PLSEU(Layer):
    def __init__(self, alpha_initializer=Constant(3.0), alpha_regularizer=None, alpha_constraint=MinMaxValue(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(PLSEU, self).__init__(**kwargs)
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.alpha = None

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.alpha = self.add_weight(shape=(input_dim,), name='alpha', initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer, constraint=self.alpha_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, mask=None):
        x1 = bk.sigmoid(inputs / self.alpha) * inputs
        x2 = self.alpha * (self.alpha + bk.log(1 + bk.relu(inputs / self.alpha - self.alpha)))
        return bk.minimum(x1, x2)

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint)
        }
        base_config = super(PLSEU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def make_plseu(alpha):
    def _plseu(x):
        return PLSEU(alpha_initializer=Constant(alpha))(x)

    return _plseu


def plseu(x):
    return make_plseu(3.0)(x)


class AdjustableLayer(Layer):
    def adjust(self):
        pass
