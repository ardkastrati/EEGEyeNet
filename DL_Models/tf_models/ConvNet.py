from abc import ABC, abstractmethod
from tf_models.BaseNet import BaseNet
import tensorflow as tf
import tensorflow.keras as keras
from config import config
import logging


class ConvNet(ABC, BaseNet):
    def __init__(self, input_shape, output_shape, loss, kernel_size=32, nb_filters=32, verbose=True, batch_size=64, 
                use_residual=False, depth=6, epochs=2, preprocessing = False, model_number=0):
        self.use_residual = use_residual
        self.depth = depth
        self.callbacks = None
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.preprocessing = preprocessing
        self.input_shape = input_shape
        super(ConvNet, self).__init__(output_shape=output_shape, loss=loss, epochs=epochs, verbose=verbose, model_number=model_number)
        logging.info('Parameters of {}, model number {}: '.format(self, model_number))
        logging.info('--------------- use residual : ' + str(self.use_residual))
        logging.info('--------------- depth        : ' + str(self.depth))
        logging.info('--------------- batch size   : ' + str(self.batch_size))
        logging.info('--------------- kernel size  : ' + str(self.kernel_size))
        logging.info('--------------- nb filters   : ' + str(self.nb_filters))
        logging.info('--------------- preprocessing: ' + str(self.preprocessing))

    def _split_model(self):
        input_layer = keras.layers.Input(self.input_shape)
        output = []

        for c in config['cluster'].keys():
            a = [self.input_shape[0]]
            a.append(len(config['cluster'][c]))
            input_shape = tuple(a)
            output.append(self._build_model(X=tf.transpose(tf.nn.embedding_lookup(tf.transpose(input_layer), config['cluster'][c]))))

        # append the results and perform 1 dense layer with last_channel dimension and the output layer
        x = tf.keras.layers.Concatenate()(output)
        dense = tf.keras.layers.Dense(32, activation='relu')(x)
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model

    # abstract method
    def _preprocessing(self, input_tensor):
        pass

    # abstract method
    def _module(self, input_tensor, current_depth):
        pass

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1, padding='same', use_bias=False)(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)
        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def _build_model(self, X=[]):
        if config['split']:
            input_layer = X
        else:
            input_layer = tf.keras.layers.Input(self.input_shape)

        if self.preprocessing:
            preprocessed = self._preprocessing(input_layer)
            x = preprocessed
            input_res = preprocessed
        else:
            x = input_layer
            input_res = input_layer

        for d in range(self.depth):
            x = self._module(x, d)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)
        if config['split']:
            return gap_layer
            
        if self.loss == 'prosaccade_clf':
            output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(gap_layer)
        elif config['task'] == 'gaze-reg':
            output_layer = tf.keras.layers.Dense(2, activation='linear')(gap_layer)
        else: #elif config['task'] == 'angle-reg':
            output_layer = tf.keras.layers.Dense(1, activation='linear')(gap_layer) 
        #else:
         #   pass #TODO: implement for event detection task

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model
