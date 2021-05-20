import numpy as np
import tensorflow as tf
from keras import backend as bk, initializers, regularizers, constraints
from keras.layers import Layer

from ml2lm.calc.model.units.gadget import AdjustableLayer


class TargetEmbedding(AdjustableLayer):
    def __init__(self, seg_num=200, noise_scale=0.2, period=100, pave_momentum=0.5, mask_zero=False,
                 target_momentum=0.99, val_epsilon=1e-3, val_inf=1e10, embed_only=True, io_aligned=True, **kwargs):
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
        self.io_aligned = io_aligned

        self.input_dim = None
        self.target_dim = None
        self._target_dim = None
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
        self.target_dim = input_shape[1][-1]
        self._target_dim = self.target_dim
        if self.io_aligned:
            assert (self.input_dim == self.target_dim or 1 == self.target_dim)
            self._target_dim = 1

        ele_num = self.input_dim * self._target_dim * (self.seg_num + self.mask_zero)
        self.embedding = self.add_weight(shape=(ele_num,), name='embedding', initializer='zeros', trainable=False)
        self.update_cnt = self.add_weight(shape=(ele_num,), name='update_cnt', initializer='zeros', trainable=False)

        self.cur_min = self.add_weight(shape=(self.input_dim,), name='cur_min', initializer='zeros', trainable=False)
        self.cur_max = self.add_weight(shape=(self.input_dim,), name='cur_max', initializer='ones', trainable=False)
        self.moving_min = self.add_weight(shape=(self.input_dim,), name='moving_min', initializer='zeros',
                                          trainable=False)
        self.moving_max = self.add_weight(shape=(self.input_dim,), name='moving_max', initializer='ones',
                                          trainable=False)
        self.call_cnt = self.add_weight(shape=(), name='call_cnt', initializer='zeros', trainable=False)

        self.built = True

    def _calc_seg_indices(self, val, min_val, max_val):
        gap = max_val - min_val
        gap_ind = gap < self.val_epsilon
        seg_scale = self.seg_num / (bk.cast(gap_ind, val.dtype) * self.val_epsilon + bk.cast(~gap_ind, val.dtype) * gap)
        seg_indices = bk.clip(bk.cast(seg_scale * (val - min_val), 'int32'), 0, self.seg_num - 1) + self.mask_zero
        if self.mask_zero:
            seg_indices *= bk.cast(val != 0, 'int32')

        seg_num = self.seg_num + self.mask_zero
        ele_num = self.input_dim * seg_num * self._target_dim
        if self._target_dim > 1:
            seg_indices = bk.repeat_elements(seg_indices, self._target_dim, axis=-1)
        seg_indices += bk.cast(bk.arange(0, ele_num, seg_num), dtype='int32')

        return seg_indices

    def _ungather(self, seg_indices, vals):
        return tf.sparse.to_dense(tf.sparse.reorder(tf.SparseTensor(bk.cast(bk.reshape(bk.stack(
            [bk.repeat_elements(bk.expand_dims(bk.arange(0, bk.shape(seg_indices)[0])), seg_indices.shape[1],
                                axis=-1), seg_indices], axis=-1), (-1, 2)), 'int64'),
            bk.flatten(vals), [bk.shape(seg_indices)[0]] + list(self.embedding.shape))))

    def _sum_seg_embeddings(self, seg_indices, seg_embeddings):
        return bk.sum(self._ungather(seg_indices, seg_embeddings), axis=0), bk.sum(
            self._ungather(seg_indices, bk.ones_like(seg_embeddings)), axis=0)

    def _pave_embedding(self, _embedding):
        dtype = _embedding.dtype
        start, seg_num = int(self.mask_zero), self.seg_num + self.mask_zero
        ele_num = self.input_dim * self._target_dim * seg_num
        indices = bk.arange(0, ele_num) % seg_num
        _embedding1 = bk.concatenate([_embedding[1:], _embedding[-1:]])
        _embedding_1 = bk.concatenate([_embedding[0:1], _embedding[:-1]])

        return (bk.cast(indices < start, dtype) * _embedding + bk.cast(indices == start, dtype) * _embedding1
                + bk.cast(indices == seg_num - 1, dtype) * _embedding_1 + bk.cast(
            (indices > start) & (indices < seg_num - 1), dtype) * (_embedding1 + _embedding_1) / 2)

    def adjust(self):
        dtype = self.embedding.dtype

        unchanged_mask = bk.cast(0 == self.update_cnt, dtype)
        paved_embedding = self._pave_embedding(self.embedding) * unchanged_mask
        unpaved_embedding = self.embedding * unchanged_mask
        bk.update_add(self.embedding, (1 - self.pave_momentum) * (paved_embedding - unpaved_embedding))

        inv_seg_scale = (self.cur_max - self.cur_min) / self.seg_num
        x = bk.expand_dims(bk.arange(0, self.seg_num, dtype=dtype), axis=-1) * inv_seg_scale + (
            self.cur_min + self.val_epsilon * inv_seg_scale)
        x = bk.concatenate([x, x + (1. - 2 * self.val_epsilon) * inv_seg_scale], axis=0)
        y = bk.transpose(bk.reshape(self.embedding, (self.input_dim * self._target_dim, -1)))[int(self.mask_zero):]
        y = bk.concatenate([y, y], axis=0)
        seg_indices = self._calc_seg_indices(x, self.moving_min, self.moving_max)
        tmp_embedding, tmp_cnt = self._sum_seg_embeddings(seg_indices, y)

        bk.update(self.embedding, tmp_embedding / (tmp_cnt + bk.cast(0 == tmp_cnt, dtype=dtype)))
        unmatched_mask = bk.cast(0 == self.embedding, dtype)
        bk.update_add(self.embedding, self._pave_embedding(self.embedding) * unmatched_mask)

        bk.update(self.update_cnt, bk.zeros_like(self.update_cnt))
        bk.update(self.cur_min, self.moving_min)
        bk.update(self.cur_max, self.moving_max)

    def _update_embedding(self, x, y, seg_indices, seg_embeddings):
        dtype = self.embedding.dtype
        delta_embeddings = (1 - self.target_momentum) * (y - seg_embeddings)
        tmp_embedding, tmp_cnt = self._sum_seg_embeddings(seg_indices, delta_embeddings)

        bk.update_add(self.embedding, tmp_embedding / (tmp_cnt + bk.cast(0 == tmp_cnt, dtype=dtype)))
        bk.update_add(self.update_cnt, tmp_cnt)

        if self.mask_zero:
            min_val = bk.min(x + bk.constant(self.val_inf, dtype=dtype) * bk.cast(0 == x, dtype), axis=0)
            max_val = bk.max(x + bk.constant(-self.val_inf, dtype=dtype) * bk.cast(0 == x, dtype), axis=0)
        else:
            min_val, max_val = bk.min(x, axis=0), bk.max(x, axis=0)
        bk.moving_average_update(self.moving_min, min_val, self.val_momentum)
        bk.moving_average_update(self.moving_max, max_val, self.val_momentum)

    def call(self, inputs, training=None):
        if training is None:
            training = bk.learning_phase()
        training = bk.get_value(training)

        (x, y), dtype = inputs, self.embedding.dtype
        x = bk.cast(x, dtype)

        seg_indices = self._calc_seg_indices(x, self.cur_min, self.cur_max)
        seg_embeddings = bk.gather(self.embedding, seg_indices)
        output = seg_embeddings

        if training:
            bk.update(self.call_cnt, self.call_cnt + 1)
            if 1 == self._target_dim:
                y = bk.cast(y, dtype) * (bk.cast(x != 0, dtype) if self.mask_zero else bk.ones_like(x))
            else:
                _x = bk.expand_dims(x, -1)
                y = bk.reshape(bk.expand_dims(bk.cast(y, dtype), 1) * (bk.cast(
                    _x != 0, dtype) if self.mask_zero else bk.ones_like(_x)), (-1, self.input_dim * self._target_dim))
            if self.noise_scale > 0:
                output = seg_embeddings * (1. + bk.random_uniform(bk.shape(seg_embeddings), minval=-self.noise_scale,
                                                                  maxval=self.noise_scale, dtype=dtype))

            self._update_embedding(x, y, seg_indices, seg_embeddings)
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
            'embed_only': self.embed_only,
            'io_aligned': self.io_aligned
        }
        base_config = super(TargetEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[0])
        output_shape[-1] = (2 - self.embed_only) * self.input_dim * self._target_dim
        return tuple(output_shape)


class TargetEmbedding4CPU(TargetEmbedding):
    def _ungather(self, val, seg_indices):
        return tf.sparse.to_dense(tf.sparse.reorder(
            tf.SparseTensor(bk.expand_dims(bk.cast(seg_indices, 'int64'), -1), val, self.embedding.shape)))

    def _sum_seg_embeddings(self, seg_indices, seg_embeddings):
        def __merge(ret, ele):
            _tmp_embedding, _tmp_cnt = ret
            _seg_ind, _seg_embedding = ele
            return _tmp_embedding + self._ungather(_seg_embedding, _seg_ind), _tmp_cnt + self._ungather(
                bk.ones_like(_seg_embedding), _seg_ind)

        tmp_embedding = bk.zeros_like(self.embedding)
        tmp_cnt = bk.zeros_like(self.embedding)
        return bk.foldl(__merge, (seg_indices, seg_embeddings), initializer=(tmp_embedding, tmp_cnt))


class TargetEmbedding4TPU(TargetEmbedding):
    def _ungather(self, seg_indices, vals):
        return tf.sparse.to_dense(tf.SparseTensor(bk.cast(bk.reshape(bk.stack(
            [bk.repeat_elements(bk.expand_dims(bk.arange(0, bk.shape(seg_indices)[0])), seg_indices.shape[1],
                                axis=-1), seg_indices], axis=-1), (-1, 2)), 'int64'),
            bk.flatten(vals), [bk.shape(seg_indices)[0]] + list(self.embedding.shape)))


def sqrt_out_dim_calcor(seg_nums, max_param_num=None):
    in_dim = np.prod(seg_nums)
    out_dim = int(in_dim ** 0.5)
    param_num = out_dim ** 3

    if max_param_num is not None and param_num > max_param_num:
        sn_size = len(seg_nums)
        max_dim = int(max_param_num ** (2 / (3 * sn_size)))
        seg_nums = np.clip(seg_nums, 0, max_dim)
        in_dim = np.prod(seg_nums)
        out_dim = int(in_dim ** 0.5)

    return in_dim, out_dim, seg_nums


def log_out_dim_calcor(seg_nums, max_param_num=None):
    feat_num = len(seg_nums)
    param_num_l = np.sum(np.log(seg_nums))
    if max_param_num is not None and param_num_l > np.log(max_param_num):
        max_dim = int(max_param_num ** (1 / feat_num))
        seg_nums = np.clip(seg_nums, 0, max_dim)

    in_dim = np.prod(seg_nums)
    out_dim = feat_num * int(np.log(in_dim))
    return in_dim, max(2, out_dim), seg_nums


class SegEmbedding(AdjustableLayer):
    def __init__(self, seg_nums, feat_idx_list, embed_trainable_list=None, input_val_range=(0., 1.),
                 out_dim_calcor=log_out_dim_calcor, max_param_num=int(1e6), period=100, pave_momentum=0.5,
                 embeddings_initializer='uniform', embeddings_regularizer=None, embeddings_constraint=None, **kwargs):
        super(SegEmbedding, self).__init__(**kwargs)
        self.supports_masking = True

        self.seg_nums = np.array(seg_nums)
        self.feat_idx_list = feat_idx_list
        self.embed_trainable_list = embed_trainable_list
        self.input_val_range = input_val_range
        self.out_dim_calcor = out_dim_calcor
        self.max_param_num = max_param_num
        self.period = period
        self.pave_momentum = pave_momentum

        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)

        self.phase = len(self.feat_idx_list)
        self.embedding_list = [[] for i in range(self.phase)]
        if not self.embed_trainable_list:
            self.embed_trainable_list = [True] * self.phase
        self.update_cnt_list = [[] for i in range(self.phase)]
        self.seg_num_list = [[] for i in range(self.phase)]
        self.seg_num_mul_list = [[] for i in range(self.phase)]

        feats_num = self.seg_nums.shape[0]
        if isinstance(self.input_val_range[0], (int, float)):
            self.min_vals = bk.constant([self.input_val_range[0]] * feats_num)
        else:
            self.min_vals = bk.constant(self.input_val_range[0])
        if isinstance(self.input_val_range[1], (int, float)):
            self.max_vals = bk.constant([self.input_val_range[1]] * feats_num)
        else:
            self.max_vals = bk.constant(self.input_val_range[1])
        self.min_val_list = [[] for i in range(self.phase)]
        self.max_val_list = [[] for i in range(self.phase)]

        self.call_cnt = None

    def _add_embeddings(self, phase):
        feats_indices = self.feat_idx_list[phase]
        for i, feats_idx in enumerate(feats_indices):
            in_dim, out_dim, seg_nums = self.out_dim_calcor(self.seg_nums[feats_idx], self.max_param_num)
            self.seg_num_list[phase].append(bk.constant(seg_nums, dtype='int32'))
            self.seg_num_mul_list[phase].append(
                bk.expand_dims(bk.constant(np.append(np.cumprod(seg_nums[:0:-1])[::-1], [1]), dtype='int32')))
            self.min_val_list[phase].append(bk.gather(self.min_vals, feats_idx))
            self.max_val_list[phase].append(bk.gather(self.max_vals, feats_idx))

            self.embedding_list[phase].append(self.add_weight(
                shape=(in_dim, out_dim), name=f'seg_embed_{phase}_{i}', initializer=self.embeddings_initializer,
                regularizer=self.embeddings_regularizer, constraint=self.embeddings_constraint,
                trainable=self.embed_trainable_list[phase]))
            self.update_cnt_list[phase].append(self.add_weight(
                shape=(in_dim,), name=f'se_update_cnt_{phase}_{i}', initializer='zeros', trainable=False))

    def build(self, input_shape):
        for i in range(self.phase):
            self._add_embeddings(i)
        self.call_cnt = self.add_weight(shape=(), name='call_cnt', initializer='zeros', trainable=False)
        self.built = True

    @staticmethod
    def _calc_indices(val, min_val, max_val, seg_num, seg_num_mul):
        seg_scale = bk.cast(seg_num, val.dtype) / (max_val - min_val)
        seg_indices = bk.clip(bk.cast(seg_scale * (val - min_val), 'int32'), 0, seg_num - 1)
        seg_indices = bk.squeeze(bk.dot(seg_indices, seg_num_mul), -1)

        return seg_indices

    @staticmethod
    def _pave_embedding(_embedding, row_num):
        dtype = _embedding.dtype
        indices = bk.expand_dims(bk.arange(0, _embedding.shape[0])) % row_num
        _embedding1 = bk.concatenate([_embedding[1:], _embedding[-1:]], axis=0)
        _embedding_1 = bk.concatenate([_embedding[0:1], _embedding[:-1]], axis=0)

        return (bk.cast(0 == indices, dtype) * _embedding1 + bk.cast(indices == row_num - 1, dtype) * _embedding_1
                + bk.cast((indices > 0) & (indices < row_num - 1), dtype) * (_embedding1 + _embedding_1) / 2)

    def adjust(self):
        for i in range(self.phase):
            if self.embed_trainable_list[i]:
                for j, embedding in enumerate(self.embedding_list[i]):
                    update_cnt = self.update_cnt_list[i][j]
                    unchanged_mask = bk.expand_dims(bk.cast(0 == update_cnt, embedding.dtype))
                    paved_embedding = self._pave_embedding(embedding, self.seg_num_list[i][j][-1]) * unchanged_mask
                    unpaved_embedding = embedding * unchanged_mask

                    bk.update_add(embedding, (1 - self.pave_momentum) * (paved_embedding - unpaved_embedding))
                    bk.update(update_cnt, bk.zeros_like(update_cnt))

    def call(self, inputs, training=None):
        if training is None:
            training = bk.learning_phase()
        training = bk.get_value(training)

        indices, outputs = [[] for i in range(self.phase)], []
        for i in range(self.phase):
            trainable = self.embed_trainable_list[i]
            cur_outputs = []
            for j, feats_idx in enumerate(self.feat_idx_list[i]):
                inds = self._calc_indices(
                    bk.transpose(bk.gather(bk.transpose(inputs), feats_idx)), self.min_val_list[i][j],
                    self.max_val_list[i][j], self.seg_num_list[i][j], self.seg_num_mul_list[i][j])
                if trainable:
                    indices[i].append(inds)
                cur_outputs.append(bk.gather(self.embedding_list[i][j], inds))
            outputs.append(bk.concatenate(cur_outputs) if len(cur_outputs) > 1 else cur_outputs[0])

        if training:
            bk.update(self.call_cnt, self.call_cnt + 1)

            for i in range(self.phase):
                for j, inds in enumerate(indices[i]):
                    ind, _, cnts = tf.unique_with_counts(bk.flatten(inds))

                    update_cnt = self.update_cnt_list[i][j]
                    tmp_cnt = tf.sparse.to_dense(tf.sparse.reorder(tf.SparseTensor(bk.expand_dims(
                        bk.cast(ind, 'int64'), -1), bk.cast(cnts, update_cnt.dtype), update_cnt.shape)))
                    bk.update_add(update_cnt, tmp_cnt)

            if self.period is not None and self.call_cnt % self.period == 0:
                self.adjust()

        return outputs

    def get_config(self):
        config = {
            'seg_nums': self.seg_nums,
            'feat_idx_list': self.feat_idx_list,
            'embed_trainable_list': self.embed_trainable_list,
            'input_val_range': self.input_val_range,
            'out_dim_calcor': self.out_dim_calcor,
            'max_param_num': self.max_param_num,
            'period': self.period,
            'pave_momentum': self.pave_momentum,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': constraints.serialize(self.embeddings_constraint)
        }
        base_config = super(SegEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        output_shape = [(input_shape[0], bk.sum([embedding.shape[-1] for embedding in self.embedding_list[i]]))
                        for i in range(self.phase)]
        return output_shape
