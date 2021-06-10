import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import CSVLogger
import numpy as np
import logging


class prediction_history(tf.keras.callbacks.Callback):
    """
    Prediction history for model ensembles=
    """
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.predhis = []
        self.targets = validation_data[1]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        self.predhis.append(y_pred)


class BaseNet:
    def __init__(self, loss, input_shape, output_shape, epochs=50, verbose=True, model_number=0):
        self.epochs = epochs
        self.verbose = verbose
        self.model_number = model_number
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss = loss 
        self.nb_channels = input_shape[1]
        self.timesamples = input_shape[0]
        self.model = self._build_model()

        # Compile the model depending on the task 
        if self.loss == 'bce':
            self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        elif self.loss == 'mse':
            self.model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['mean_squared_error'])
        elif self.loss == 'angle_loss':
            from DL_Models.tf_models.utils.losses import angle_loss
            self.model.compile(loss=angle_loss, optimizer=keras.optimizers.Adam())
        else:
            raise ValueError("Choose valid loss for your task")
            
        # if self.verbose:
            # self.model.summary()

        logging.info(f"Number of trainable parameters: {np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])}")
        logging.info(f"Number of non-trainable parameters: {np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])}")

    # abstract method
    def _split_model(self):
        pass

    # abstract method
    def _build_model(self):
        pass

    def get_model(self):
        return self.model

    def save(self, path):
        self.model.save(path)


    def fit(self, X_train, y_train, X_val, y_val):
        #csv_logger = CSVLogger(config['batches_log'], append=True, separator=';')
        #ckpt_dir = config['model_dir'] + '/best_models/' + config['model'] + '_nb_{}_'.format(self.model_number) + 'best_model.h5'
        #ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_dir, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        hist = self.model.fit(X_train, y_train, verbose=2, batch_size=self.batch_size, validation_data=(X_val, y_val),
                                  epochs=self.epochs, callbacks=[early_stop])


    def predict(self, testX):
        return self.model.predict(testX)
