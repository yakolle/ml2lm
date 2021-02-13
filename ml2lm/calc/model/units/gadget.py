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

        self.forward_type = 'zero'
        self.backward_type = 'stop'

    def dummy(self):
        self.forward_type = 'zero'
        self.backward_type = 'stop'

    def freeze(self):
        self.forward_type = 'one'
        self.backward_type = 'stop'

    def normal(self):
        self.forward_type = 'one'
        self.backward_type = 'normal'

    def call(self, inputs, training=None):
        output = inputs
        if 'zero' == self.forward_type:
            output = bk.zeros_like(output)
        if 'stop' == self.backward_type:
            output = bk.stop_gradient(output)
        return output
