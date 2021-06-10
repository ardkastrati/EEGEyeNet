import tensorflow as tf
from config import config
from utils.utils import *
import logging
from DL_Models.tf_models.ConvNet import ConvNet


class INCEPTION(ConvNet):
    """
    The InceptionTime architecture used as baseline. This is the architecture explained in the paper

    'InceptionTime: Finding AlexNet for Time Series Classification' with authors
    Hassan Ismail Fawaz, Benjamin Lucas, Germain Forestier, Charlotte Pelletier,
    Daniel F. Schmidt, Jonathan Weber, Geoffrey I. Webb, Lhassane Idoumghar, Pierre-Alain Muller, François Petitjean
    """

    def __init__(self, loss, model_number, batch_size, input_shape, output_shape, kernel_size=64, epochs = 50, nb_filters=16, verbose=True,
                            use_residual=True, depth=12, bottleneck_size=16):
        self.bottleneck_size = bottleneck_size
        super(INCEPTION, self).__init__(input_shape=input_shape, output_shape=output_shape, loss=loss, kernel_size=kernel_size, epochs=epochs, 
                            nb_filters=nb_filters, verbose=verbose, batch_size=batch_size, 
                            use_residual=use_residual, depth=depth, model_number=model_number)
        logging.info('--------------- bottleneck_size : ' + str(self.bottleneck_size))
        
    def _module(self, input_tensor, current_depth):
        """
        The module of InceptionTime (Taken from the implementation of InceptionTime paper).
        It is made of a bottleneck convolution that reduces the number of channels from 128 -> 32.
        Then it uses 3 filters with different kernel sizes (Default values are 40, 20, and 10)
        In parallel it uses a simple convolution with kernel size 1 with max pooling for stability during training.
        The outputs of each convolution are concatenated, followed by batch normalization and a ReLu activation.
        """
        if int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1, padding='same', use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i], padding='same', use_bias=False)(input_inception))

        max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(input_tensor)
        conv_6 = tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1, padding='same', use_bias=False)(max_pool_1)

        conv_list.append(conv_6)
        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x
