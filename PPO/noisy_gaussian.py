from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
from keras.layers import GaussianNoise
import numpy as np
import tensorflow as tf

class NoisyGaussian(Layer):

    def __init__(self, stddev,interval,**kwargs):
        super(NoisyGaussian, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        self.interval = interval
        self.counter = 0

    def call(self, inputs,training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        self.counter += 1
        if(self.counter % self.interval == 0) :
            res = K.in_train_phase(noised, inputs, training)
            return res
        return inputs

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

