import tensorflow as tf
from keras import backend as bk

from ml2lm.calc.model.units.activations import AdjustableLayer


class TargetEmbedding(AdjustableLayer):
    def __init__(self, seg_num=200, noise_scale=0.2, period=100, pave_momentum=0.5, mask_zero=False,
                 target_momentum=0.99, val_epsilon=1e-3, val_inf=1e10, embed_only=True, **kwargs):
        super(TargetEmbedding, self).__init__(**kwargs)
        self.supports_masking = True

        self.seg_num = seg_num
        self.noise_scale = noise_scale
        self.period = period
        self.pave_momentum = pave_momentum
        self.mask_zero = mask_zero
        self.target_momentum = target_momentum
        self.val_epsilon = val_epsilon
        self.val_inf = val_inf
        self.embed_only = embed_only

        self.input_dim = None
        self.embedding = None
        self.update_cnt = None

        self.val_momentum = self.val_epsilon ** (1 / self.period)
        self.cur_min = None
        self.cur_max = None
        self.moving_min = None
        self.moving_max = None

        self.call_cnt = None

    def build(self, input_shape):
        self.input_dim = input_shape[0][-1]

        ele_num = self.input_dim * (self.seg_num + self.mask_zero)
        self.embedding = self.add_weight(shape=(ele_num,), name='embedding', initializer='zeros', trainable=False)
        self.update_cnt = self.add_weight(shape=(ele_num,), name='update_cnt', initializer='zeros', trainable=False,
                                          dtype='int32')

        self.cur_min = self.add_weight(shape=(self.input_dim,), name='cur_min', initializer='zeros', trainable=False)
        self.cur_max = self.add_weight(shape=(self.input_dim,), name='cur_max', initializer='ones', trainable=False)
        self.moving_min = self.add_weight(shape=(self.input_dim,), name='moving_min', initializer='zeros',
                                          trainable=False)
        self.moving_max = self.add_weight(shape=(self.input_dim,), name='moving_max', initializer='ones',
                                          trainable=False)
        self.call_cnt = self.add_weight(shape=(), name='call_cnt', initializer='zeros', trainable=False, dtype='int32')

        self.built = True

    def _calc_seg_indices(self, val, min_val, max_val):
        seg_scale = (self.seg_num - 1) / (max_val - min_val)
        seg_indices = bk.clip(bk.cast(seg_scale * (val - min_val), 'int32'), 0, self.seg_num - 1) + self.mask_zero
        if self.mask_zero:
            seg_indices *= bk.cast(val != 0, 'int32')
        seg_indices += bk.arange(0, self.input_dim * (self.seg_num + self.mask_zero), self.seg_num + self.mask_zero)

        return seg_indices

    def _ungather(self, seg_indices, vals):
        return tf.sparse.to_dense(tf.sparse.reorder(tf.SparseTensor(bk.cast(bk.reshape(bk.stack(
            [bk.repeat_elements(bk.expand_dims(bk.arange(0, bk.shape(seg_indices)[0])), seg_indices.shape[1],
                                axis=-1), seg_indices], axis=-1), (-1, 2)), 'int64'),
            bk.flatten(vals), [bk.shape(seg_indices)[0]] + list(self.embedding.shape))))

    def _sum_seg_embeddings(self, seg_indices, seg_embeddings):
        return bk.sum(self._ungather(seg_indices, seg_embeddings), axis=0), bk.sum(
            self._ungather(seg_indices, bk.ones_like(seg_embeddings, dtype='int32')), axis=0)

    def _pave_embedding(self, _embedding):
        def __pave(i):
            if i < start:
                return _embedding[i::col_num]
            if i == start:
                return _embedding[start + 1::col_num]
            if col_num - 1 == i:
                return _embedding[col_num - 2::col_num]
            return (_embedding[i - 1::col_num] + _embedding[i + 1::col_num]) / 2

        start, col_num = int(self.mask_zero), self.seg_num + self.mask_zero
        return bk.flatten(bk.transpose(bk.map_fn(__pave, bk.arange(0, col_num), dtype=_embedding.dtype)))

    def adjust(self):
        dtype = self.embedding.dtype

        unchanged_mask = bk.cast(0 == self.update_cnt, dtype)
        paved_embedding = self._pave_embedding(self.embedding) * unchanged_mask
        unpaved_embedding = self.embedding * unchanged_mask
        bk.update_add(self.embedding, (1 - self.pave_momentum) * (paved_embedding - unpaved_embedding))

        inv_seg_scale = (self.cur_max - self.cur_min) / (self.seg_num - 1)
        x = bk.expand_dims(bk.arange(0, self.seg_num, dtype=dtype), axis=-1) * inv_seg_scale + (
            self.cur_min + self.val_epsilon * inv_seg_scale)
        x = bk.concatenate([x, x + (1. - 2 * self.val_epsilon) * inv_seg_scale], axis=0)
        y = bk.transpose(bk.reshape(self.embedding, (self.input_dim, -1)))[int(self.mask_zero):]
        y = bk.concatenate([y, y], axis=0)
        seg_indices = self._calc_seg_indices(x, self.moving_min, self.moving_max)
        tmp_embedding, tmp_cnt = self._sum_seg_embeddings(seg_indices, y)

        bk.update(self.embedding, tmp_embedding / (bk.cast(tmp_cnt, dtype) + bk.cast(0 == tmp_cnt, dtype=dtype)))
        unmatched_mask = bk.cast(0 == self.embedding, dtype)
        bk.update_add(self.embedding, self._pave_embedding(self.embedding) * unmatched_mask)

        bk.update(self.update_cnt, bk.zeros_like(self.update_cnt))
        bk.update(self.cur_min, self.moving_min)
        bk.update(self.cur_max, self.moving_max)

    def call(self, inputs, training=None):
        if training is None:
            training = bk.learning_phase()

        (x, y), dtype = inputs, self.embedding.dtype
        if training:
            bk.update(self.call_cnt, self.call_cnt + 1)
            y = bk.cast(y, dtype)
            if self.mask_zero:
                y *= bk.cast(x != 0, dtype)
        x = bk.cast(x, dtype)

        seg_indices = self._calc_seg_indices(x, self.cur_min, self.cur_max)
        seg_embeddings = bk.gather(self.embedding, seg_indices)
        output = seg_embeddings

        if training:
            output = seg_embeddings * (1. + bk.random_uniform(bk.shape(seg_embeddings), minval=-self.noise_scale,
                                                              maxval=self.noise_scale, dtype=dtype))

            delta_embeddings = (1 - self.target_momentum) * (y - seg_embeddings)
            tmp_embedding, tmp_cnt = self._sum_seg_embeddings(seg_indices, delta_embeddings)

            bk.update_add(self.embedding,
                          tmp_embedding / (bk.cast(tmp_cnt, dtype=dtype) + bk.cast(0 == tmp_cnt, dtype=dtype)))
            bk.update_add(self.update_cnt, tmp_cnt)

            if self.mask_zero:
                min_val = bk.min(x + bk.constant(self.val_inf, dtype=dtype) * bk.cast(0 == x, dtype), axis=0)
                max_val = bk.max(x + bk.constant(-self.val_inf, dtype=dtype) * bk.cast(0 == x, dtype), axis=0)
            else:
                min_val, max_val = bk.min(x, axis=0), bk.max(x, axis=0)
            bk.moving_average_update(self.moving_min, min_val, self.val_momentum)
            bk.moving_average_update(self.moving_max, max_val, self.val_momentum)

            if self.period is not None and self.call_cnt % self.period == 0:
                self.adjust()

        if self.embed_only:
            return bk.cast(output, inputs[0].dtype)
        return bk.concatenate([inputs[0], bk.cast(output, inputs[0].dtype)])

    def get_config(self):
        config = {
            'seg_num': self.seg_num,
            'noise_scale': self.noise_scale,
            'period': self.period,
            'pave_momentum': self.pave_momentum,
            'mask_zero': self.mask_zero,
            'target_momentum': self.target_momentum,
            'val_epsilon': self.val_epsilon,
            'val_inf': self.val_inf,
            'embed_only': self.embed_only
        }
        base_config = super(TargetEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape[-1] *= (2 - self.embed_only)
        return tuple(output_shape)
