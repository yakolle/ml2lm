import keras.backend as bk
import tensorflow as tf
from keras import initializers, regularizers, constraints
from keras.engine import Layer, InputSpec
from keras.initializers import Constant

from ml2lm.calc.model.units.gadget import MinMaxValue


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


@tf.custom_gradient
def out_of_flow_sigmoid(logits):
    pred = bk.sigmoid(logits)

    def _grad(dy):
        return dy

    return pred, _grad


def calc_section_mask(x, sections):
    mask = False
    for l, r in sections:
        mask |= ((x >= l) & (x < r))
    return mask


def make_seg_sigmoid(scale=200, round_type='ceil', gradient_type='sm', pos_p_sections=None, neg_p_sections=None,
                     threshold=0.5):
    @tf.custom_gradient
    def _seg_sigmoid(logits):
        pos_handler, neg_handler = (tf.math.ceil, tf.math.floor) if 'ceil' == round_type else (
            tf.math.floor, tf.math.ceil)
        pred = bk.sigmoid(logits)

        pos_mask = pred >= threshold
        pos_mask_c = (pos_mask & calc_section_mask(pred, pos_p_sections)) if pos_p_sections is not None else None
        pos_mask_nc = (pos_mask & ~pos_mask_c) if pos_mask_c is not None else None
        neg_mask = pred < threshold
        neg_mask_c = (neg_mask & calc_section_mask(pred, neg_p_sections)) if neg_p_sections is not None else None
        neg_mask_nc = (neg_mask & ~neg_mask_c) if neg_mask_c is not None else None

        s_pred = pred * scale
        if pos_mask_c is None:
            pred_pos = pos_handler(bk.cast(pos_mask, logits.dtype) * s_pred) / scale
        else:
            pred_pos_c = pos_handler(bk.cast(pos_mask_c, logits.dtype) * s_pred) / scale
            pred_pos_nc = bk.cast(pos_mask_nc, logits.dtype) * pred
            pred_pos = pred_pos_c + pred_pos_nc
        if neg_mask_c is None:
            pred_neg = neg_handler(bk.cast(neg_mask, logits.dtype) * s_pred) / scale
        else:
            pred_neg_c = neg_handler(bk.cast(neg_mask_c, logits.dtype) * s_pred) / scale
            pred_neg_nc = bk.cast(neg_mask_nc, logits.dtype) * pred
            pred_neg = pred_neg_c + pred_neg_nc
        s_pred = pred_pos + pred_neg

        def __grad(dy):
            grad = 1.
            if 'sm' == gradient_type:
                grad = pred * (1. - pred)
            elif 'ssm' == gradient_type:
                grad = s_pred * (1. - s_pred)
            return dy * grad

        return s_pred, __grad

    return _seg_sigmoid
