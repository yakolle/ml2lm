from keras import backend as bk, initializers
from keras.engine import Layer, InputSpec


class BatchScale(Layer):
    def __init__(self, min_val=0, max_val=1, momentum=0.5, epsilon=1e-3, moving_min_initializer='zeros',
                 moving_max_initializer='ones', **kwargs):
        super(BatchScale, self).__init__(**kwargs)
        self.supports_masking = True

        self.min_val = min_val
        self.max_val = max_val
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_min_initializer = initializers.get(moving_min_initializer)
        self.moving_max_initializer = initializers.get(moving_max_initializer)

        self.moving_min = None
        self.moving_max = None

        self.scale_range = self.max_val - self.min_val
        self.scale = None
        self.min = None

    def build(self, input_shape):
        self.input_spec = InputSpec(min_ndim=2)
        shape = (input_shape[-1],)

        self.moving_min = self.add_weight(shape=shape, name='moving_min', initializer=self.moving_min_initializer,
                                          trainable=False)
        self.moving_max = self.add_weight(shape=shape, name='moving_max', initializer=self.moving_max_initializer,
                                          trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        def scale_inference():
            return bk.clip(inputs * self.scale + self.min, self.min_val, self.max_val)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return scale_inference()

        cur_min = bk.min(inputs, axis=0)
        cur_max = bk.max(inputs, axis=0)
        self.moving_min = self.momentum * self.moving_min + (1 - self.momentum) * cur_min
        self.moving_max = self.momentum * self.moving_max + (1 - self.momentum) * cur_max

        self.scale = self.scale_range / (self.moving_max - self.moving_min + self.epsilon)
        self.min = self.min_val - self.moving_min * self.scale

        scaled_training = scale_inference()

        # Pick the normalized form corresponding to the training phase.
        return bk.in_train_phase(scaled_training, scale_inference, training=training)

    def get_config(self):
        config = {
            'min_val': self.min_val,
            'max_val': self.max_val,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'moving_min_initializer': initializers.serialize(self.moving_min_initializer),
            'moving_max_initializer': initializers.serialize(self.moving_max_initializer)
        }
        base_config = super(BatchScale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
