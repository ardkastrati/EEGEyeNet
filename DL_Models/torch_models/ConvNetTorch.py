from abc import ABC, abstractmethod
from re import X

from DL_Models.torch_models.BaseNetTorch import BaseNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
import logging
from DL_Models.torch_models.Modules import Pad_Conv, Pad_Pool

class ConvNet(ABC, BaseNet):
    """
    This class defines all the common functionality for convolutional nets
    Inherit from this class and only implement _module() and _get_nb_features_output_layer() methods
    Modules are then stacked in the forward() pass of the model 
    """
    def __init__(self, loss, model_number, batch_size, input_shape, output_shape, kernel_size=32, epochs=2, nb_filters=32, verbose=True,
                use_residual=False, depth=6,  preprocessing = False):
        """
        We define the layers of the network in the __init__ function
        """
        self.batch_size = batch_size
        super().__init__(loss=loss, input_shape=input_shape, output_shape=output_shape, epochs=epochs, verbose=verbose,
                            model_number=model_number)

        self.use_residual = use_residual
        self.depth = depth
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.preprocessing = preprocessing

        # Define all the convolutional and shortcut modules that we will need in the model 
        self.conv_blocks = nn.ModuleList([self._module(d) for d in range(self.depth)])
        if self.use_residual:
            self.shortcuts = nn.ModuleList([self._shortcut(d) for d in range(int(self.depth / 3))])
        self.gap_layer = nn.AvgPool1d(kernel_size=2, stride=1)
        self.gap_layer_pad = Pad_Pool(left=0, right=1, value=0)

        if self.verbose and model_number == 0:
            logging.info(f"Number of model parameters: {sum(p.numel() for p in self.parameters())}")
            logging.info(f"Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
            logging.info('--------------- use residual : ' + str(self.use_residual))
            logging.info('--------------- depth        : ' + str(self.depth))
            logging.info('--------------- batch_size        : ' + str(self.batch_size))
            logging.info('--------------- kernel size  : ' + str(self.kernel_size))
            logging.info('--------------- nb filters   : ' + str(self.nb_filters))
            logging.info('--------------- preprocessing: ' + str(self.preprocessing))
        

    def forward(self, x):
        """
        Implements the forward pass of the network
        Modules defined in a class implementing ConvNet are stacked and shortcut connections are used if specified. 
        """
        if self.preprocessing:
            x = self._preprocessing(x)
        input_res = x # set for the residual shortcut connection   
        # Stack the modules and residual connection 
        shortcut_cnt = 0 
        for d in range(self.depth):
            x = self.conv_blocks[d](x)            
            if self.use_residual and d % 3 == 2:
                res = self.shortcuts[shortcut_cnt](input_res)
                shortcut_cnt += 1 
                x = torch.add(x, res)
                x = nn.functional.relu(x)
                input_res = x
        x = self.gap_layer_pad(x) 
        x = self.gap_layer(x)
        x = x.view(self.batch_size, -1)
        output = self.output_layer(x) # Defined in BaseNet 
        return output 
    
    def _shortcut(self, depth):
        """
        Implements a shortcut with a convolution and batch norm 
        This is the same for all models implementing ConvNet, therefore defined here 
        Padding before convolution for constant tensor shape, similar to tensorflow.keras padding=same
        """
        return nn.Sequential(
                Pad_Conv(kernel_size=self.kernel_size, value=0),
                nn.Conv1d(in_channels=self.nb_channels if depth==0 else self.nb_features,
                             out_channels=self.nb_features, kernel_size=self.kernel_size),
                nn.BatchNorm1d(num_features=self.nb_features)
            )

    def get_nb_features_output_layer(self):
        """
        Return number of features passed into the output layer of the network 
        nb.features has to be defined in a model implementing ConvNet
        """
        return self.nb_features * self.timesamples
        
    # abstract method
    def _preprocessing(self, input_tensor):
        pass

    # abstract method
    def _module(self, input_tensor, current_depth):
        pass
    
    def _split_model(self):
        pass