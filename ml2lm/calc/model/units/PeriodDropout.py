from keras import backend as bk
from keras.engine import Layer
from keras.layers import Dropout


class AdjustableLayer(Layer):
    def adjust(self):
        pass


class PeriodDropout(Dropout, AdjustableLayer):
    def __init__(self, rate, noise_shape=None, seed=0, period=10, axis=None, **kwargs):
        super(PeriodDropout, self).__init__(rate, noise_shape, seed, **kwargs)
        self.supports_masking = True

        self.period = period
        self.axis = axis

        self.call_cnt = 0

    def adjust(self):
        self.seed += 1

    def call(self, inputs, training=None):
        self.call_cnt += 1
        if self.period is not None and not self.call_cnt % self.period:
            self.adjust()

        if 0. < self.rate < 1.:
            if self.axis is not None:
                noise_shape = [1] * len(bk.int_shape(inputs))
                noise_shape[self.axis] = noise_shape[self.axis]
                noise_shape = tuple(noise_shape)
            else:
                noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return bk.dropout(inputs, self.rate, noise_shape, seed=self.seed)

            return bk.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs

    def get_config(self):
        config = {
            'period': self.period,
            'axis': self.axis
        }
        base_config = super(PeriodDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
