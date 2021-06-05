import tensorflow as tf
import tensorflow.keras as keras
from config import config
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from utils.utils import train_val_split
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
    def __init__(self, loss, epochs=50, verbose=True, model_number=0):
        self.epochs = epochs
        self.verbose = verbose
        self.model_number = model_number

        if config['split']:
            self.model = self._split_model()
        else:
            self.model = self._build_model()

        # Compile the model depending on the task 
        if loss == 'bce':
            self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        elif loss == 'mse':
            self.model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['mean_squared_error'])
        elif loss == 'angle_loss':
            from DL_Models.tf_models.utils.losses import angle_loss
            self.model.compile(loss=angle_loss, optimizer=keras.optimizers.Adam())
        else:
            raise ValueError("Choose valid loss")
            
        if self.verbose:
            self.model.summary()

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

    def fit(self, X_train, y_train, X_val, y_val):
        csv_logger = CSVLogger(config['batches_log'], append=True, separator=';')
        ckpt_dir = config['model_dir'] + '/best_models/' + config['model'] + '_nb_{}_'.format(self.model_number) + 'best_model.h5'
        ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_dir, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        if config['model'] == 'eegnet':
            hist = self.model.fit(X_train, y_train, verbose=1, validation_data=(X_val, y_val),
                                  epochs=config['epochs'], callbacks=[csv_logger, ckpt, early_stop])
        else:
            hist = self.model.fit(X_train, y_train, verbose=2, batch_size=self.batch_size, validation_data=(X_val, y_val),
                                  epochs=config['epochs'], callbacks=[csv_logger, ckpt, early_stop])


    def predict(self, testX):
        return self.model.predict(testX)
