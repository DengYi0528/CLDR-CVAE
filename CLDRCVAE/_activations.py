import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Activation, ReLU, Lambda, LeakyReLU


def mean_activation(x):
    return tf.clip_by_value(K.exp(x), 1e-5, 1e6)


def disp_activation(x):
    return tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


ACTIVATIONS = {
    "relu": ReLU(name='reconstruction_output'),
    'leaky_relu': LeakyReLU(name="reconstruction_output"),
    'linear': Activation("linear", name='reconstruction_output'),
    'mean_activation': Activation(mean_activation, name="decoder_mean"),
    'disp_activation': Activation(disp_activation, name="decoder_disp"),
    'sigmoid': Activation('sigmoid', name='decoder_pi'),
    'softplus': Activation('softplus', name='reconstruction_output')    # 新增
}
