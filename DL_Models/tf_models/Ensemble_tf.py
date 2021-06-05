import tensorflow as tf
from config import config
from utils.utils import *
import keras
import logging
import os


class Ensemble_tf:
    """
    The Ensemble is a model itself, which contains a number of models that are averaged on prediction. 
    Default: nb_models of the model_type model
    Optional: Initialize with a model list, create a more versatile Ensemble model
    """
    def __init__(self, model_name='CNN', nb_models='5', loss='mse', batch_size=64, **model_params):
        """
        model_name: the model that the ensemble uses
        nb_models: Number of models to run in the ensemble
        model_list: optional, give a list of models that should be contained in the Ensemble
        ...
        """
        self.model_name = model_name
        self.nb_models = nb_models
        self.loss = loss
        self.batch_size = batch_size
        self.model_params = model_params

        if self.model_name == 'CNN':
            from DL_Models.tf_models.CNN.CNN import CNN
            self.model = CNN
        elif self.model_name == 'EEGNet':
            from DL_Models.tf_models.EEGNet.eegNet import EEGNet
            self.model = EEGNet
        elif self.model_name == 'InceptionTime':
            from DL_Models.tf_models.InceptionTime.Inception import INCEPTION
            self.model = INCEPTION
        elif self.model_name == 'PyramidalCNN':
            from DL_Models.tf_models.PyramidalCNN.PyramidalCNN import PyramidalCNN
            self.model = PyramidalCNN
        elif self.model_name == 'Xception':
            from DL_Models.tf_models.Xception.Xception import XCEPTION
            self.model = XCEPTION


    def fit(self, trainX, trainY, validX, validY):
        """
        Fit all the models in the ensemble and save them to the run directory 
        """
        # Fit the models 
        for i in range(config['ensemble']):
            print("------------------------------------------------------------------------------------")
            logging.info('Start fitting model number {}/{} ...'.format(i+1, self.nb_models))
            model = self.model(loss=self.loss, batch_size=self.batch_size, **self.model_params)
            model.fit(trainX, trainY, validX, validY)
            logging.info('Finished fitting model number {}/{} ...'.format(i+1, self.nb_models))

    def predict(self, testX):
        # Load models from the directory
        path = config['model_dir'] + '/best_models/'
        for i, file in enumerate(os.listdir(path)):
            logging.info(f"Loading model nb {i}from file {file} and predict with it")
            model = keras.models.load_model(file)
            if i == 0:
                pred = model.predict(testX)
            else:
                pred += model.predict(testX)
        res = pred / config['ensemble']
        logging.info(f"tf ensemble predicts {res}")
        return res # TODO: this might have to be rounded for majority decision in LR task 



