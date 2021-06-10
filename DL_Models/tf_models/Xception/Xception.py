import tensorflow as tf
from config import config
from utils.utils import *
import logging
from DL_Models.tf_models.ConvNet import ConvNet


class XCEPTION(ConvNet):
    """
    The Xception architecture. This is inspired by Xception paper, which describes how 'extreme' convolutions can be represented
    as separable convolutions and can achieve better accuracy then the Inception architecture. It is made of modules in a specific depth.
    Each module, in our implementation, consists of a separable convolution followed by batch normalization and a ReLu activation layer.
    """
    def __init__(self, loss, model_number, batch_size, input_shape, output_shape, kernel_size=40, nb_filters=128, verbose=True, epochs=1,
                        use_residual=True, depth=6):
        super(XCEPTION, self).__init__(input_shape=input_shape, output_shape=output_shape, loss=loss, kernel_size=kernel_size, nb_filters=nb_filters,
                        verbose=verbose, epochs=epochs, batch_size=batch_size, use_residual=use_residual, depth=depth,
                        model_number=model_number, preprocessing=False)

    def _module(self, input_tensor, current_depth):
        """
        The module of Xception. Consists of a separable convolution followed by batch normalization and a ReLu activation function.
        """
        x = tf.keras.layers.SeparableConv1D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='same', use_bias=False, depth_multiplier=1)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x