from keras import backend as bk
from keras.constraints import Constraint
from keras.engine import Layer


class MinMaxValue(Constraint):
    def __init__(self, min_value=1e-3, max_value=None):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return bk.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


class AdjustableLayer(Layer):
    def adjust(self):
        pass


class DummyLayer(Layer):
    def __init__(self, **kwargs):
        super(DummyLayer, self).__init__(**kwargs)
        self.supports_masking = True

        self.forward_type = self.add_weight(shape=(), name='forward_type', initializer='zeros', trainable=False)
        self.backward_type = self.add_weight(shape=(), name='backward_type', initializer='zeros', trainable=False)

    def dummy(self):
        bk.update(self.forward_type, bk.constant(0.))
        bk.update(self.backward_type, bk.constant(0.))

    def freeze(self):
        bk.update(self.forward_type, bk.constant(1.))
        bk.update(self.backward_type, bk.constant(0.))

    def normal(self):
        bk.update(self.forward_type, bk.constant(1.))
        bk.update(self.backward_type, bk.constant(1.))

    def call(self, inputs, training=None):
        output = inputs
        if 0. == self.forward_type:
            output = bk.zeros_like(output)
        if 0. == self.backward_type:
            output = bk.stop_gradient(output)
        return output
