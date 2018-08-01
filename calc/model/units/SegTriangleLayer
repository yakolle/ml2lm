from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.initializers import Constant, RandomUniform
from keras.layers import Dense


class SegTriangleLayer(Layer):
    def __init__(self, seg_num, input_val_range=(0, 1), relu_alpha=0, **kwargs):
        assert seg_num >= 2

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(SegTriangleLayer, self).__init__(**kwargs)
        self.seg_num = seg_num

        self.left_pos = None
        self.middle_pos = None
        self.right_pos = None
        self.middle_seg_width = None

        self.input_val_range = input_val_range
        self.relu_alpha = relu_alpha

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        assert 1 == input_shape[-1]

        seg_width = (self.input_val_range[1] - self.input_val_range[0]) / self.seg_num
        left_pos = self.input_val_range[0] + seg_width
        right_pos = self.input_val_range[1] - seg_width

        self.left_pos = self.add_weight(shape=(1,), initializer=Constant(value=left_pos), name='left_pos')
        if self.seg_num > 2:
            self.middle_pos = self.add_weight(shape=(self.seg_num - 2,), name='middle_pos',
                                              initializer=RandomUniform(minval=left_pos, maxval=right_pos - seg_width))
        else:
            self.middle_pos = None
        self.right_pos = self.add_weight(shape=(1,), initializer=Constant(value=right_pos), name='right_pos')

        if self.seg_num > 2:
            self.middle_seg_width = self.add_weight(shape=(self.seg_num - 2,), initializer=Constant(value=seg_width),
                                                    name='middle_seg_width')
        else:
            self.middle_seg_width = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: 1})
        self.built = True

    def call(self, inputs, **kwargs):
        left_out = self.left_pos - inputs
        middle_out = None if self.middle_pos is None else -K.abs(inputs - self.middle_pos) + self.middle_seg_width
        right_out = inputs - self.right_pos

        if self.middle_pos is not None:
            output = K.concatenate([left_out, middle_out, right_out])
        else:
            output = K.concatenate([left_out, right_out])
        return K.relu(output, alpha=self.relu_alpha)

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert 1 == input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.seg_num
        return tuple(output_shape)

    def get_config(self):
        config = {
            'seg_num': self.seg_num,
            'input_val_range': self.input_val_range,
            'relu_alpha': self.relu_alpha
        }
        base_config = super(SegTriangleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def transform(segment_layer, activation=None, use_bias=False, kernel_initializer='glorot_uniform',
              bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
              kernel_constraint=None, bias_constraint=None):
    return Dense(1, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(segment_layer)
