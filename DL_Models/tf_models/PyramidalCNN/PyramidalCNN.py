import tensorflow as tf
from config import config
from utils.utils import *
import logging
from tf_models.ConvNet import ConvNet
from tensorflow.keras.constraints import max_norm


class PyramidalCNN(ConvNet):
    """
    The Classifier_PyramidalCNN is one of the simplest classifiers. It implements the class ConvNet, which is made of modules with a
    specific depth, where for each depth the number of filters is increased.
    """
    def __init__(self, input_shape, kernel_size=16, epochs = 50, nb_filters=16, verbose=True, batch_size=64, 
                    use_residual=False, depth=6, model_number=0):

        super(PyramidalCNN, self).__init__(input_shape, kernel_size=kernel_size, epochs=epochs, nb_filters=nb_filters,
                    verbose=verbose, batch_size=batch_size, use_residual=use_residual, depth=depth, model_number=model_number)

    def __str__(self):
        return self.__class__.__name__
        
    def _module(self, input_tensor, current_depth):
        """
        The module of CNN is made of a simple convolution with batch normalization and ReLu activation. Finally, MaxPooling is also used.
        """
        x = tf.keras.layers.Conv1D(filters=self.nb_filters*(current_depth + 1), kernel_size=self.kernel_size, padding='same', use_bias=False)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
        return x
