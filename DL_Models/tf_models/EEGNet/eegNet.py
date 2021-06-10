import tensorflow as tf
import tensorflow.keras as keras
from config import config
from utils.utils import *
from DL_Models.tf_models.BaseNet import BaseNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
import numpy as np
import logging


class EEGNet(BaseNet):
    """
    The EEGNet architecture used as baseline. This is the architecture explained in the paper

    'EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces' with authors
    Vernon J. Lawhern, Amelia J. Solon, Nicholas R. Waytowich, Stephen M. Gordon, Chou P. Hung, Brent J. Lance
    """

    def __init__(self, loss, model_number, batch_size, input_shape, output_shape, epochs=50, 
                F1 = 16, F2 = 256, verbose = True, D=4, kernel_size=250, dropout_rate = 0.5,
                norm_rate = 0.5, dropoutType = 'Dropout', X = None):

        #self.nb_classes = nb_classes
        self.chans = input_shape[1]
        self.samples = input_shape[0]
        self.dropoutRate = dropout_rate
        self.kernLength = kernel_size
        self.loss = loss 
        self.batch_size = batch_size
        self.output_shape = output_shape
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate
        self.dropoutType = dropoutType
        super(EEGNet, self).__init__(loss=loss, input_shape=input_shape, output_shape=output_shape, epochs=epochs, verbose=verbose, model_number=model_number)
        logging.info('Parameters...')
        logging.info('--------------- chans            : ' + str(self.chans))
        logging.info('--------------- samples          : ' + str(self.samples))
        logging.info('--------------- dropoutRate      : ' + str(self.dropoutRate))
        logging.info('--------------- kernLength       : ' + str(self.kernLength))
        logging.info('--------------- F1               : ' + str(self.F1))
        logging.info('--------------- D                : ' + str(self.D))
        logging.info('--------------- F2               : ' + str(self.F2))
        logging.info('--------------- norm_rate        : ' + str(self.norm_rate))
        
    def _split_model(self):
        """
        This method is added to make use of clustering idea in EEGNet as well. It divides the input into different clusters.
        Then it builds a model of EEGNet for each cluster, concatenates the extracted featers and uses a Dense layer to finally
        classify the data.
        """
        input_layer = keras.layers.Input((config['eegnet']['channels'] , config['eegnet']['samples'] ))
        output=[]

        # run inception over the cluster
        for c in config['cluster'].keys():
            output.append(self._build_model(X = tf.expand_dims(tf.transpose(tf.nn.embedding_lookup(
            tf.transpose(input_layer,(1,0,2)),config['cluster'][c]),(1,0,2)),axis=-1), c = c))

        # append the results and perform 1 dense layer with last_channel dimension and the output layer

        x = tf.keras.layers.Concatenate(axis=1)(output)
        dense=tf.keras.layers.Dense(32, activation='relu')(x)
        output_layer=tf.keras.layers.Dense(1,activation='sigmoid')(dense)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model


    def _build_model(self, X = None, c = None):
        """
        The model of EEGNet (Taken from the implementation of EEGNet paper).
        """
        if self.dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif self.dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
        
        input1 = Input(shape=(self.chans, self.samples, 1))

        block1 = Conv2D(self.F1, (1, self.kernLength), padding='same',
                        input_shape=(self.chans, self.samples, 1),
                        use_bias=False)(input1)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D((self.chans, 1), use_bias=False,
                                 depth_multiplier=self.D,
                                 depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 16))(block1)
        block1 = dropoutType(self.dropoutRate)(block1)

        block2 = SeparableConv2D(self.F2, (1, 64),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization()(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 6))(block2)
        block2 = dropoutType(self.dropoutRate)(block2)

        flatten = Flatten()(block2)

        # Create output layer depending on task
        if self.loss == 'bce':
            output_layer = tf.keras.layers.Dense(self.output_shape, activation='sigmoid')(flatten)
        elif self.loss == 'mse' or 'angle-loss':
            output_layer = tf.keras.layers.Dense(self.output_shape, activation='linear')(flatten)
        else:
            raise ValueError("Choose a valid task")

        return Model(inputs=input1, outputs=output_layer)
