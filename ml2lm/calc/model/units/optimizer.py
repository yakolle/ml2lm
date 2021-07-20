import numpy as np
import tensorflow as tf
from keras import backend as bk

from ml2lm.calc.model.units.embed import TargetEmbedding, SegEmbedding, TargetEmbedding4CPU, TargetEmbedding4TPU
from ml2lm.calc.model.units.norm import ScaleLayer


def diff(x, direction='both'):
    x = bk.transpose(x)
    xl, xr = None, None
    xm = x[1:] - x[:-1]
    if direction in ['left', 'both']:
        xl = bk.transpose(bk.concatenate([xm[0:1], xm], axis=0))
    if direction in ['right', 'both']:
        xr = bk.transpose(bk.concatenate([xm, xm[-1:]], axis=0))

    if 'left' == direction:
        return xl
    if 'right' == direction:
        return xr
    return xl, xr


def diff_by_col_num(x, col_num, direction='both'):
    x_shape = bk.shape(x)
    y = bk.reshape(x, [-1, col_num])
    y = diff(y, direction=direction)
    if 'both' == direction:
        return bk.reshape(y[0], x_shape), bk.reshape(y[1], x_shape)
    return bk.reshape(y, x_shape)


class DeltaDifferenceDelegate(TargetEmbedding):
    def __init__(self, delta_x_num=1000, grad_ease=1., **kwargs):
        kwargs['seg_num'] = delta_x_num
        kwargs['noise_scale'] = 0.
        kwargs['mask_zero'] = False
        kwargs['embed_only'] = True
        super(DeltaDifferenceDelegate, self).__init__(**kwargs)
        self.supports_masking = True

        self.grad_ease = grad_ease

    def call(self, inputs, training=None):
        if training is None:
            training = bk.learning_phase()
        training = bk.get_value(training)

        if training:
            dtype = self.embedding.dtype
            bk.update(self.call_cnt, self.call_cnt + 1)
            if self.period is not None and self.call_cnt % self.period == 0:
                self.adjust()

            @tf.custom_gradient
            def __delegate(_x, _y):
                x = bk.cast(_x, dtype)
                if 1 == self._target_dim:
                    y = bk.cast(_y, dtype) * bk.ones_like(x)
                else:
                    y = bk.reshape(bk.expand_dims(bk.cast(_y, dtype), 1) * bk.ones_like(bk.expand_dims(x, -1)),
                                   (-1, self.input_dim * self._target_dim))

                def _grad(dy):
                    seg_indices = self._calc_seg_indices(x, self.cur_min, self.cur_max)
                    seg_embeddings = bk.gather(self.embedding, seg_indices)
                    self._update_embedding(x, y, seg_indices, seg_embeddings)

                    dys = diff_by_col_num(self.embedding, col_num=self.seg_num, direction='both')
                    cur_dy = bk.gather((dys[0] + dys[1]) / 2, seg_indices)
                    if 1 == self._target_dim:
                        cur_dy *= dy
                    else:
                        cur_dy = bk.reshape(cur_dy, (-1, self.input_dim, self._target_dim)) * bk.expand_dims(dy, 1)
                        cur_dy = bk.sum(cur_dy, axis=-1) / self.input_dim

                    return cur_dy * self.grad_ease, dy

                return _y, _grad

            return __delegate(inputs[0], inputs[1])
        return inputs[-1]

    def get_config(self):
        config = {'delta_x_num': self.seg_num, 'grad_ease': self.grad_ease}
        base_config = super(DeltaDifferenceDelegate, self).get_config()
        base_config.pop('delta_x_num')
        config.update(base_config)
        return config

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        return input_shape[-1]


class DeltaDifferenceDelegate4CPU(DeltaDifferenceDelegate, TargetEmbedding4CPU):
    pass


class DeltaDifferenceDelegate4TPU(DeltaDifferenceDelegate, TargetEmbedding4TPU):
    pass


def node_embedding(node, ne_seg_num, dd_delegate=None):
    ne_units = node.shape[1]
    node = ScaleLayer(min_trainable=False, max_trainable=False)(node)
    y = SegEmbedding([ne_seg_num] * ne_units, [np.expand_dims(np.array(range(ne_units)), -1)])(node)[0]
    if dd_delegate is None:
        dd_delegate = DeltaDifferenceDelegate(delta_x_num=100, grad_ease=1., io_aligned=False)
    return dd_delegate([node, y])


def make_unit_step(step_point=0., grad_range=(-0.01, 0.01), grad_ease=1.):
    @tf.custom_gradient
    def _unit_step(_x):
        y = bk.cast(_x >= step_point, _x.dtype)

        def __grad(dy):
            l, r = grad_range
            h = 1. / (r - l)
            k = h / (r - l)
            cur_dy = h - bk.abs(l + r - 2 * _x) * k
            cur_dy = bk.cast((_x >= l) & (_x <= r), _x.dtype) * cur_dy
            return dy * cur_dy * grad_ease

        return y, __grad

    return _unit_step


def make_unit_box(box_range=(-0.5, 0.5), grad_range=((-0.75, -0.25), (0.25, 0.75)), delta_x_num=(10, 10), grad_ease=1.):
    @tf.custom_gradient
    def _unit_box(_x):
        y = bk.cast((_x >= box_range[0]) & (_x <= box_range[1]), _x.dtype)

        def __grad(dy):
            (l1, r1), (l2, r2) = grad_range
            m1, m2 = (l1 + r1) / 2, (l2 + r2) / 2
            n1, n2 = delta_x_num

            mask1 = (_x >= l1) & (_x <= r1)
            dy1 = 1. / (r1 - l1)
            seg_scale = n1 / (r1 - m1)
            seg_indices = n1 - bk.clip(bk.cast(seg_scale * bk.abs(_x - m1), 'int32'), 0, n1 - 1)
            dy1 = bk.cast(seg_indices, _x.dtype) * dy1 / n1

            mask2 = (_x >= l2) & (_x <= r2)
            dy2 = 1. / (r2 - l2)
            seg_scale = n2 / (r2 - m2)
            seg_indices = n2 - bk.clip(bk.cast(seg_scale * bk.abs(_x - m2), 'int32'), 0, n2 - 1)
            dy2 = bk.cast(seg_indices, _x.dtype) * dy2 / n2

            cur_dy = dy1 * bk.cast(mask1, _x.dtype) + dy2 * bk.cast(mask2, _x.dtype)
            return dy * cur_dy * grad_ease

        return y, __grad

    return _unit_box


def make_gather_in_flow(embed_in_dim=None):
    def _gather_in_flow(_embed, _indices):
        y = bk.gather(_embed,
                      bk.cast(_indices if embed_in_dim is None else bk.clip(_indices, 0, embed_in_dim), 'int32'))

        @tf.custom_gradient
        def _gather(__indices):
            in_dim = bk.shape(__indices)
            in_dim = 1 if 1 == len(in_dim) else in_dim[-1]

            def __grad(dy):
                dy = bk.sum(dy, axis=-1) / bk.cast(in_dim, dy.dtype)

            return y, __grad

        return (y + _gather(_indices)) / 2

    return _gather_in_flow
