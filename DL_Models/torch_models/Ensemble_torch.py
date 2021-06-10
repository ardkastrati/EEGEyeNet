from config import config
import logging
import torch
import os
import numpy as np
import re

from DL_Models.torch_models.torch_utils.dataloader import create_dataloader
from DL_Models.torch_models.torch_utils.training import test_loop


class Ensemble_torch:
    """
    The Ensemble is a model itself, which contains a number of models whose prediction is averaged (majority decision in case of a classifier). 
    """

    def __init__(self, model_name='CNN', nb_models=5, loss='bce', batch_size=64, **model_params):
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

        if self.model_name == 'CNN':
            from DL_Models.torch_models.CNN.CNN import CNN
            self.model = CNN
        elif self.model_name == 'EEGNet':
            from DL_Models.torch_models.EEGNet.eegNet import EEGNet
            self.model = EEGNet
        elif self.model_name == 'InceptionTime':
            from DL_Models.torch_models.InceptionTime.InceptionTime import Inception
            self.model = Inception
        elif self.model_name == 'PyramidalCNN':
            from DL_Models.torch_models.PyramidalCNN.PyramidalCNN import PyramidalCNN
            self.model = PyramidalCNN
        elif self.model_name == 'Xception':
            from DL_Models.torch_models.Xception.Xception import XCEPTION
            self.model = XCEPTION

    
    def fit(self, trainX, trainY, validX, validY):
        """
        Fit an ensemble of models. They will be saved by BaseNet into the model dir
        """
        # Create dataloaders
        trainX = np.transpose(trainX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
        validX = np.transpose(validX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
        train_dataloader = create_dataloader(trainX, trainY, self.batch_size, self.model_name)
        validation_dataloader = create_dataloader(validX, validY, self.batch_size, self.model_name)
        # Fit the models 
        for i in range(self.nb_models):
            logging.info("------------------------------------------------------------------------------------")
            logging.info('Start fitting model number {}/{} ...'.format(i+1, self.nb_models))
            model = self.model(loss = self.loss, model_number=i, batch_size=self.batch_size, **self.model_params)
            model.fit(train_dataloader, validation_dataloader)
            self.models.append(model)
            logging.info('Finished fitting model number {}/{} ...'.format(i+1, self.nb_models))


    def predict(self, testX):
        testX = np.transpose(testX, (0, 2, 1))  # (batch_size, samples, channels) to (bs, ch, samples) as torch conv layers want it
        a,b,c = testX.shape
        a = self.batch_size - a % self.batch_size
        dummy = np.zeros((a,b,c))
        #print(dummy.shape)
        testX = np.concatenate((testX, dummy)) # TO ADD batch_size - testX.shape[0]%batch_size
        test_dataloader = create_dataloader(testX, testX, self.batch_size, self.model_name, drop_last=False)
        pred = None
        #print(f"self models len {len(self.models)}")
        for model in self.models:
            if torch.cuda.is_available():
                model.cuda()
            if pred is not None:
                pred += test_loop(dataloader=test_dataloader, model=model)
            else:
                pred = test_loop(dataloader=test_dataloader, model=model)
        pred = pred[:-a]
        return pred / len(self.models) 

    def save(self, path):
        for i, model in enumerate(self.models):
            ckpt_dir = path + self.model_name + '_nb_{}_'.format(i) + '.pth'
            torch.save(model.state_dict(), ckpt_dir)

    def load(self, path):
        #print(f"cuda avail {torch.cuda.is_available()}")
        self.models = []
        for file in os.listdir(path):
            if not self.load_file_pattern.match(file):
                continue
            # These 3 lines are needed for torch to load
            logging.info(f"Loading model nb from file {file} and predict with it")
            model = self.model(loss=self.loss, model_number=0, batch_size=self.batch_size,
                               **self.model_params)  # model = TheModelClass(*args, **kwargs)
            #print(path + file)
            model.load_state_dict(torch.load(path + file))  # model.load_state_dict(torch.load(PATH))
            model.eval()  # needed before prediction
            self.models.append(model)