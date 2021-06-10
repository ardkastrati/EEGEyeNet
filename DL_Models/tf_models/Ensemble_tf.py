import tensorflow as tf
from config import config
from utils.utils import *
import keras
import logging
import os
import re 
import numpy as np 

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
        self.model_params = model_params
        self.batch_size = batch_size
        self.loss = loss
        self.model_instance = None
        self.load_file_pattern = re.compile(self.model_name[:3] +  '.*_nb_._best_model.pth', re.IGNORECASE)
        self.models = []

        logging.info(f"Instantiated Ensemble of {self.model_name} models")

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
        print(f"trainX shape before transpose: {trainX.shape}")
        if self.model_name == 'EEGNet':
            trainX = np.transpose(trainX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
            validX = np.transpose(validX, (0,2,1))
        print(f"trainX shape after transpose: {trainX.shape}")

        self.models = []
        # Fit the models 
        for i in range(self.nb_models):
            logging.info("------------------------------------------------------------------------------------")
            logging.info('Start fitting model number {}/{} ...'.format(i+1, self.nb_models))
            model = self.model(loss=self.loss, model_number=i, batch_size=self.batch_size, **self.model_params)
            self.models.append(model )
            model.fit(trainX, trainY, validX, validY)
            logging.info('Finished fitting model number {}/{} ...'.format(i+1, self.nb_models))

    def predict(self, testX):
        if self.model_name == 'EEGNet':
            testX = np.transpose(testX, (0, 2, 1)) 
        pred = None
        for model in self.models:
            if pred is not None:
                pred += model.predict(testX)
            else:
                pred = model.predict(testX)
        return pred / len(self.models)

    def save(self, path):
        for i, model in enumerate(self.models):
            ckpt_dir = path + self.model_name + '_nb_{}_'.format(i)
            model.save(ckpt_dir)

    def load(self, path):
        self.models = []
        for file in os.listdir(path):
            if not self.load_file_pattern.match(file):
                continue
            logging.info(f"Loading model nb from file {file} and predict with it")
            self.models.append(keras.models.load_model(file))