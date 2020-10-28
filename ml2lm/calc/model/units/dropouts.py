import tensorflow as tf
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
            input_shape = bk.int_shape(inputs)
            noise_shape = [1] * len(input_shape)
            noise_shape[self.axis] = input_shape[self.axis]
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
    def __init__(self, rate, period=10, axis=None, seed=0, **kwargs):
        super(PeriodDropout, self).__init__(rate, seed=seed, **kwargs)
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
                input_shape = bk.int_shape(inputs)
                noise_shape = [1] * len(input_shape)
                noise_shape[self.axis] = input_shape[self.axis]
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


class SegDropout(Dropout, AdjustableLayer):
    def __init__(self, rate, anneal=0.1, agg_method='mean', smooth_rate=0., noise_type='gaussian', keep_amp_type='abs',
                 epsilon=1e-6, period=None, axis=None, seed=None, **kwargs):
        super(SegDropout, self).__init__(rate, seed=seed, **kwargs)
        self.supports_masking = True

        self.anneal = 0.5 + anneal
        self.agg_method = agg_method
        self.smooth_rate = max(min(smooth_rate, 1.), 0.)
        self.noise_type = noise_type
        self.keep_amp_type = keep_amp_type
        self.epsilon = epsilon

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
            input_shape = bk.int_shape(inputs)
            if self.axis is not None:
                noise_shape = [1] * len(input_shape)
                noise_shape[self.axis] = input_shape[self.axis]
                noise_shape = tuple(noise_shape)
            else:
                noise_shape = bk.shape(inputs)

            def dropped_inputs():
                if 'max' == self.agg_method:
                    x_agg = bk.max(inputs, axis=0)
                    if self.smooth_rate > 0:
                        x_agg = self.smooth_rate * bk.mean(inputs, axis=0) + (1 - self.smooth_rate) * x_agg
                elif 'extreme' == self.agg_method:
                    x_mean = bk.mean(inputs, axis=0)
                    x_agg = tf.where(x_mean >= 0, bk.max(inputs, axis=0), bk.min(inputs, axis=0))
                    if self.smooth_rate > 0:
                        x_agg = self.smooth_rate * x_mean + (1 - self.smooth_rate) * x_agg
                else:
                    x_agg = bk.mean(inputs, axis=0)

                x_min, x_max = bk.min(x_agg), bk.max(x_agg)
                x_agg_int = bk.cast(input_shape[-1] * (x_agg - x_min) / (x_max - x_min), 'int32')
                y, idx, counts = tf.unique_with_counts(x_agg_int)
                dr = self.rate ** (1. / (self.anneal * bk.cast(counts, inputs.dtype)))
                dr = tf.where(1 == counts, self.rate * bk.ones_like(dr), dr)

                if 'gaussian' == self.noise_type:
                    sigma = (dr / (1. - dr)) ** .5
                    return inputs * bk.map_fn(lambda i: bk.random_normal((1,), 1., sigma[i], dtype=inputs.dtype)[0],
                                              idx, dtype=inputs.dtype)
                else:
                    dr_tensor = bk.random_uniform(noise_shape, seed=self.seed, dtype=inputs.dtype)
                    ret = inputs * bk.cast(dr_tensor >= bk.gather(dr, idx), inputs.dtype)

                    if 'abs' == self.keep_amp_type:
                        old_amps = bk.sum(bk.abs(inputs), axis=-1, keepdims=True)
                        cur_amps = bk.sum(bk.stop_gradient(bk.abs(ret)), axis=-1, keepdims=True)
                        ret = ret * old_amps / (cur_amps + self.epsilon)
                    elif self.keep_amp_type is not None:
                        old_amps = bk.sum(inputs, axis=-1, keepdims=True)
                        cur_amps = bk.sum(bk.stop_gradient(ret), axis=-1, keepdims=True)
                        ret = ret * old_amps / (cur_amps + self.epsilon)

                    return ret

            return bk.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs

    def get_config(self):
        config = {
            'anneal': self.anneal,
            'agg_method': self.agg_method,
            'smooth_rate': self.smooth_rate,
            'noise_type': self.noise_type,
            'keep_amp_type': self.keep_amp_type,
            'epsilon': self.epsilon,
            'period': self.period,
            'axis': self.axis
        }
        base_config = super(SegDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
