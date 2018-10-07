from keras import backend as K, initializers, regularizers, constraints, activations
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense


class SegSquareWindowLayer(Layer):
    def __init__(self, seg_num,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',
                 seg_width_initializer='ones',
                 seg_height_initializer='ones',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 seg_width_regularizer=None,
                 seg_height_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seg_width_constraint=None,
                 seg_height_constraint=None,
                 **kwargs):
        assert seg_num >= 2

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(SegSquareWindowLayer, self).__init__(**kwargs)
        self.seg_num = seg_num
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.seg_width_initializer = initializers.get(seg_width_initializer)
        self.seg_height_initializer = initializers.get(seg_height_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.seg_width_regularizer = regularizers.get(seg_width_regularizer)
        self.seg_height_regularizer = regularizers.get(seg_height_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.seg_width_constraint = constraints.get(seg_width_constraint)
        self.seg_height_constraint = constraints.get(seg_height_constraint)

        self.left_kernel = None
        self.middle_kernel = None
        self.right_kernel = None
        self.left_bias = None
        self.middle_bias = None
        self.right_bias = None
        self.middle_seg_width = None
        self.middle_seg_height = None

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.left_kernel = self.add_weight(shape=(input_dim, 1),
                                           initializer=self.kernel_initializer,
                                           name='left_kernel',
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        if self.seg_num > 2:
            self.middle_kernel = self.add_weight(shape=(input_dim, self.seg_num - 2),
                                                 initializer=self.kernel_initializer,
                                                 name='middle_kernel',
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)
        else:
            self.middle_kernel = None
        self.right_kernel = self.add_weight(shape=(input_dim, 1),
                                            initializer=self.kernel_initializer,
                                            name='right_kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)

        if self.use_bias:
            self.left_bias = self.add_weight(shape=(1,),
                                             initializer=self.bias_initializer,
                                             name='left_bias',
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)
            if self.seg_num > 2:
                self.middle_bias = self.add_weight(shape=(self.seg_num - 2,),
                                                   initializer=self.bias_initializer,
                                                   name='middle_bias',
                                                   regularizer=self.bias_regularizer,
                                                   constraint=self.bias_constraint)
            else:
                self.middle_bias = None
            self.right_bias = self.add_weight(shape=(1,),
                                              initializer=self.bias_initializer,
                                              name='right_bias',
                                              regularizer=self.bias_regularizer,
                                              constraint=self.bias_constraint)
        else:
            self.left_bias = None
            self.middle_bias = None
            self.right_bias = None

        if self.seg_num > 2:
            self.middle_seg_width = self.add_weight(shape=(self.seg_num - 2,),
                                                    initializer=self.seg_width_initializer,
                                                    name='middle_seg_width',
                                                    regularizer=self.seg_width_regularizer,
                                                    constraint=self.seg_width_constraint)
        else:
            self.middle_seg_width = None

        if self.seg_num > 2:
            self.middle_seg_height = self.add_weight(shape=(self.seg_num - 2,),
                                                     initializer=self.seg_height_initializer,
                                                     name='middle_seg_height',
                                                     regularizer=self.seg_height_regularizer,
                                                     constraint=self.seg_height_constraint)
        else:
            self.middle_seg_height = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        left_out = K.dot(inputs, self.left_kernel)
        middle_out = None
        if self.middle_kernel is not None:
            middle_out = K.dot(inputs, self.middle_kernel)
        right_out = K.dot(inputs, self.right_kernel)

        if self.use_bias:
            left_out = K.bias_add(left_out, self.left_bias)
            if self.middle_bias is not None:
                middle_out = K.bias_add(middle_out, self.middle_bias)
            right_out = K.bias_add(right_out, self.right_bias)

        if self.activation is not None:
            left_out = self.activation(left_out)
            if self.middle_kernel is not None:
                middle_out = self.activation(middle_out)
            right_out = self.activation(right_out)

        left_out = -left_out
        if self.middle_kernel is not None:
            middle_out = -middle_out * middle_out * self.middle_seg_width + self.middle_seg_height

        if self.middle_kernel is not None:
            output = K.concatenate([left_out, middle_out, right_out])
        else:
            output = K.concatenate([left_out, right_out])
        return K.relu(output)

    def compute_output_shape(self, input_shape):
        assert input_shape and 2 == len(input_shape)
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.seg_num
        return tuple(output_shape)

    def get_config(self):
        config = {
            'seg_num': self.seg_num,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'seg_width_initializer': initializers.serialize(self.seg_width_initializer),
            'seg_height_initializer': initializers.serialize(self.seg_height_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'seg_width_regularizer': regularizers.serialize(self.seg_width_regularizer),
            'seg_height_regularizer': regularizers.serialize(self.seg_height_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'seg_width_constraint': constraints.serialize(self.seg_width_constraint),
            'seg_height_constraint': constraints.serialize(self.seg_height_constraint)
        }
        base_config = super(SegSquareWindowLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def project(segment_layer, project_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None):
    return Dense(project_dim, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(segment_layer)
