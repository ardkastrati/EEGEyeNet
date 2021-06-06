import torch
import sys 
from torch import nn
import numpy as np
import pandas as pd 
from config import config 
import logging
from DL_Models.torch_models.torch_utils.training import train_loop, validation_loop


class Prediction_history:
    """
    Collect predictions on the dataset in the given dataloader
    Called after each epoch by BaseNet 
    """
    def __init__(self, dataloader, model) -> None:
        """
        predhis: a list of lists (one for each epoch) of tensors (one for each batch of length batch_size)
        """
        self.dataloader = dataloader
        self.predhis = []
        self.model = model

    def on_epoch_end(self):
        with torch.no_grad():
            y_pred = []
            for batch, (x, y) in enumerate(self.dataloader):
                # Move batch to GPU
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                pred = self.model(x)
                y_pred.append(pred) 
                # Remove batch from GPU 
                del x
                del y 
                torch.cuda.empty_cache()
            self.predhis.append(y_pred)

class BaseNet(nn.Module):
    """
    BaseNet class for ConvNet and EEGnet to inherit common functionality 
    """
    def __init__(self, loss, input_shape, output_shape, epochs=50, verbose=True, model_number=0):
        """
        Initialize common variables of models based on BaseNet, e.g. ConvNet or EEGNET 
        Create the common output layer dependent on the task to run 
        """
        super().__init__()
        self.input_shape = input_shape
        self.epochs = epochs
        self.verbose = verbose
        self.model_number = model_number
        self.timesamples = self.input_shape[0]
        self.nb_channels = self.input_shape[1]
        self.early_stopped = False
        self.loss = loss

        # Create output layer depending on task
        if loss == 'bce':
            self.loss_fn = nn.BCELoss()
            self.output_layer = nn.Sequential(
                nn.Linear(in_features=self.get_nb_features_output_layer(), out_features=output_shape),
                nn.Sigmoid()
            )
        elif loss == 'mse':
            self.loss_fn = nn.MSELoss()
            self.output_layer = nn.Sequential(
                nn.Linear(in_features=self.get_nb_features_output_layer(), out_features=output_shape) 
            )
        elif loss == 'angle-loss':
            from DL_Models.torch_models.torch_utils.custom_losses import angle_loss
            self.loss_fn = angle_loss
            self.output_layer = nn.Sequential(
                nn.Linear(in_features=self.get_nb_features_output_layer(), out_features=output_shape) 
            )
        else:
            raise ValueError("Choose a valid task")

        if verbose and self.model_number == 0:
            logging.info(f"Using loss fct: {self.loss_fn}")

    # abstract method 
    def forward(self, x):
        """
        Implements a forward pass of the network 
        This method has to be implemented by models based on BaseNet 
        """
        pass
    
    # abstract method 
    def get_nb_features_output_layer(self):
        """
        Return the number of features that the output layer should take as input
        This method has to be implemented by models based on BaseNet to compute the number of hidden neurons that the output layer takes as input. 
        """
        pass

    # abstract method
    def _split_model(self):
        pass    
    
    #@profile
    #@timing_decorator
    def fit(self, train_dataloader, validation_dataloader):
        """
        Fit the model on the dataset defined by data x and labels y 

        Inputs:
        training, validation and test dataloader containing the respective datasets

        Output:
        prediction_ensemble containing the predictions on the test data, prediction on test data when model stops is used to compute the ensemble metrics 
        """
        # Move the model to GPU
        if torch.cuda.is_available():
            self.cuda()
            logging.info(f"Model moved to cuda")
        # Create the optimizer
        optimizer = torch.optim.Adam(list(self.parameters()), lr=config['learning_rate'])
        # Create a history to track ensemble performance, similar to tensorflow.keras callbacks
        #prediction_ensemble = Prediction_history(dataloader=test_dataloader, model=self)
        # Create datastructures to collect metrics and implement early stopping
        epochs = self.epochs
        #metrics = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]} if config['task'] == 'prosaccade-clf' else {'train_loss':[], 'val_loss':[]}
        best_val_loss = sys.maxsize # For early stopping 
        patience = 0
        # Train the model 
        for t in range(epochs):
            logging.info("-------------------------------")
            logging.info(f"Epoch {t+1}")
            # Run through training and validation set  
            if not self.early_stopped:
                train_loss_epoch, train_acc_epoch = train_loop(train_dataloader, self.float(), self.loss, self.loss_fn, optimizer)
                val_loss_epoch, val_acc_epoch = validation_loop(validation_dataloader, self.float(), self.loss, self.loss_fn)
                #metrics['train_loss'].append(train_loss_epoch)
                #metrics['val_loss'].append(val_loss_epoch)
                #if config['task'] == 'prosaccade-clf':
                    #metrics['train_acc'].append(train_acc_epoch)
                    #metrics['val_acc'].append(val_acc_epoch) 
            # Add the predictions on the validation set, even if model was early stopped, later we will compute ensemble metrics on them 
            #prediction_ensemble.on_epoch_end()
            # Impementation of early stopping 
            if config['early_stopping'] and not self.early_stopped:
                if patience > config['patience']:
                    logging.info(f"Early stopping the model after {t} epochs")
                    self.early_stopped = True
                if val_loss_epoch >= best_val_loss:
                    logging.info(f"Validation loss did not improve, best was {best_val_loss}")
                    patience +=1 
                else:
                    best_val_loss = val_loss_epoch
                    logging.info(f"Improved validation loss to: {best_val_loss}")
                    patience = 0
                    
    def predict(self, X):
        tensor_X = torch.tensor(X).float()
        if torch.cuda.is_available():
            tensor_X.cuda()
        return self(tensor_X).detach().numpy()
