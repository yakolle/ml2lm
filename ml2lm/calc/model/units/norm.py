from keras import backend as bk, initializers, regularizers, constraints
from keras.engine import Layer, InputSpec


class ScaleLayer(Layer):
    def __init__(self, momentum=0.99, min_trainable=True, max_trainable=True, epsilon=1e-3,
                 moving_min_initializer='zeros', moving_max_initializer='ones', min_initializer='zeros',
                 max_initializer='ones', min_regularizer=None, max_regularizer=None, min_constraint=None,
                 max_constraint=None, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.supports_masking = True

        self.momentum = momentum
        self.min_trainable = min_trainable
        self.max_trainable = max_trainable
        self.epsilon = epsilon
        self.moving_min_initializer = initializers.get(moving_min_initializer)
        self.moving_max_initializer = initializers.get(moving_max_initializer)
        self.min_initializer = initializers.get(min_initializer)
        self.max_initializer = initializers.get(max_initializer)
        self.min_regularizer = regularizers.get(min_regularizer)
        self.max_regularizer = regularizers.get(max_regularizer)
        self.min_constraint = constraints.get(min_constraint)
        self.max_constraint = constraints.get(max_constraint)

        self.moving_min = None
        self.moving_max = None
        self.min_val = None
        self.max_val = None

    def build(self, input_shape):
        self.input_spec = InputSpec(min_ndim=2)
        shape = (input_shape[-1],)

        self.moving_min = self.add_weight(shape=shape, name='moving_min', initializer=self.moving_min_initializer,
                                          trainable=False)
        self.moving_max = self.add_weight(shape=shape, name='moving_max', initializer=self.moving_max_initializer,
                                          trainable=False)
        self.min_val = self.add_weight(shape=shape, name='min_val', initializer=self.min_initializer,
                                       regularizer=self.min_regularizer, constraint=self.min_constraint,
                                       trainable=self.min_trainable)
        self.max_val = self.add_weight(shape=shape, name='max_val', initializer=self.max_initializer,
                                       regularizer=self.max_regularizer, constraint=self.max_constraint,
                                       trainable=self.max_trainable)
        self.built = True

    def call(self, inputs, training=None):
        if training is None:
            training = bk.learning_phase()
        training = bk.get_value(training)

        if training:
            bk.moving_average_update(self.moving_min, bk.min(inputs, axis=0), self.momentum)
            bk.moving_average_update(self.moving_max, bk.max(inputs, axis=0), self.momentum)

        scale = (self.max_val - self.min_val) / (self.moving_max - self.moving_min + self.epsilon)
        output = bk.clip((inputs - self.moving_min) * scale + self.min_val, self.min_val, self.max_val)
        return output

    def get_config(self):
        config = {
            'momentum': self.momentum,
            'min_trainable': self.min_trainable,
            'max_trainable': self.max_trainable,
            'epsilon': self.epsilon,
            'moving_min_initializer': initializers.serialize(self.moving_min_initializer),
            'moving_max_initializer': initializers.serialize(self.moving_max_initializer),
            'min_initializer': initializers.serialize(self.min_initializer),
            'max_initializer': initializers.serialize(self.max_initializer),
            'min_regularizer': regularizers.serialize(self.min_regularizer),
            'max_regularizer': regularizers.serialize(self.max_regularizer),
            'min_constraint': constraints.serialize(self.min_constraint),
            'max_constraint': constraints.serialize(self.max_constraint)
        }
        base_config = super(ScaleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
