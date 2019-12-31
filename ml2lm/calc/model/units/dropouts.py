from keras import backend as bk
from keras.engine import Layer
from keras.layers import Dropout


class AdjustableLayer(Layer):
    def adjust(self):
        pass


def cut_norm_max(inputs):
    return bk.max(bk.stop_gradient(bk.abs(inputs)), axis=-1, keepdims=True)


def cut_norm_moments(inputs, sigma=2.5):
    abs_inputs = bk.stop_gradient(bk.abs(inputs))
    return bk.mean(abs_inputs, axis=-1, keepdims=True) + sigma * bk.std(abs_inputs, axis=-1, keepdims=True)


class Cutoff(AdjustableLayer):
    def __init__(self, off_val=1e-3, dist_rate=0.3, cut_norm_func=None, keep_cut_amp=False, keep_drop_amp=True,
                 cut_in_test=False, seed=None, period=None, axis=None, epsilon=1e-3, **kwargs):
        super(Cutoff, self).__init__(**kwargs)
        self.supports_masking = True

        self.off_val = off_val
        self.dist_rate = min(1., max(0., dist_rate))

        self.cut_norm_func = cut_norm_func
        self.keep_cut_amp = keep_cut_amp
        self.keep_drop_amp = keep_drop_amp
        self.cut_in_test = cut_in_test

        self.seed = seed
        self.period = period
        self.axis = axis
        self.epsilon = epsilon

        self.call_cnt = 0

    def adjust(self):
        self.seed += 1

    def call(self, inputs, training=None):
        self.call_cnt += 1
        if self.period is not None and not self.call_cnt % self.period:
            self.adjust()

        data_type = inputs.dtype
        if self.axis is not None:
            noise_shape = [1] * len(bk.int_shape(inputs))
            noise_shape[self.axis] = noise_shape[self.axis]
            noise_shape = tuple(noise_shape)
        else:
            noise_shape = bk.shape(inputs)

        def _cut():
            abs_inputs = bk.stop_gradient(bk.abs(inputs))
            off_inputs = inputs
            if self.off_val > 0:
                off_val = self.off_val if self.cut_norm_func is None else self.off_val * self.cut_norm_func(off_inputs)
                off_inputs = inputs * bk.cast(abs_inputs > off_val, data_type)

                if self.keep_cut_amp:
                    old_amps = bk.sum(abs_inputs, axis=-1, keepdims=True)
                    cur_amps = bk.sum(bk.stop_gradient(bk.abs(off_inputs)), axis=-1, keepdims=True)
                    off_inputs = off_inputs * old_amps / (cur_amps + self.epsilon)

            return off_inputs

        def _drop():
            off_inputs = _cut()
            keep_prob = 1. - self.dist_rate
            cutoff_tensor = bk.random_uniform(noise_shape, seed=self.seed, dtype=data_type)
            cutoff_tensor = bk.cast(cutoff_tensor >= self.dist_rate, data_type)

            ret = cutoff_tensor * off_inputs
            if self.keep_drop_amp:
                ret /= keep_prob
            return ret

        cut_func = _drop if 0. < self.dist_rate < 1. else _cut
        alt_func = _cut if self.cut_in_test else inputs
        return bk.in_train_phase(cut_func, alt_func, training=training)

    def get_config(self):
        config = {
            'off_val': self.off_val,
            'dist_rate': self.dist_rate,
            'cut_norm_func': self.cut_norm_func,
            'keep_cut_amp': self.keep_cut_amp,
            'keep_drop_amp': self.keep_drop_amp,
            'cut_in_test': self.cut_in_test,
            'seed': self.seed,
            'period': self.period,
            'axis': self.axis,
            'epsilon': self.epsilon
        }
        base_config = super(Cutoff, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


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
