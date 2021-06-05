from torch.nn.modules import batchnorm
from torch.nn.modules.batchnorm import BatchNorm1d
from config import config
from DL_Models.torch_models.ConvNetTorch import ConvNet
import torch
import torch.nn as nn
import logging 
from DL_Models.torch_models.Modules import Pad_Conv, Pad_Pool

class Inception(ConvNet):
    """
    The InceptionTime architecture used as baseline. This is the architecture explained in the paper
    'InceptionTime: Finding AlexNet for Time Series Classification' with authors
    Hassan Ismail Fawaz, Benjamin Lucas, Germain Forestier, Charlotte Pelletier,
    Daniel F. Schmidt, Jonathan Weber, Geoffrey I. Webb, Lhassane Idoumghar, Pierre-Alain Muller, François Petitjean
    
    """
    def __init__(self, loss, model_number, batch_size, input_shape, output_shape, kernel_size=64, epochs = 50, nb_filters=16, verbose=True,
                            use_residual=True, depth=12, bottleneck_size=16):
        """
        nb_features: specifies number of channels before the output layer 
        """
        self.bottleneck_size = bottleneck_size
        self.nb_features = 4 * nb_filters # these are the 4 concatenated parallel convolutions, width of the inner tensort passed through network 
        logging.info('--------------- bottleneck_size : ' + str(self.bottleneck_size))
        super().__init__(loss=loss, model_number=model_number, batch_size=batch_size, input_shape=input_shape, output_shape=output_shape, kernel_size=kernel_size, epochs=epochs,
                            nb_filters=nb_filters, verbose=verbose,
                            use_residual=use_residual, depth=depth)

    def _module(self, depth):
        """
        The module of InceptionTime (Taken from the implementation of InceptionTime paper).
        It is made of a bottleneck convolution that reduces the number of channels from 128 -> 32.
        Then it uses 3 filters with different kernel sizes (Default values are 40, 20, and 10)
        In parallel it uses a simple convolution with kernel size 1 with max pooling for stability during training.
        The outputs of each convolution are concatenated, followed by batch normalization and a ReLu activation.

        Padding=same 
        """
        return Inception_module(self.kernel_size, self.nb_features, self.nb_channels, 
                                self.nb_filters, self.bottleneck_size, depth)

class Inception_module(nn.Module):
    """
    This class implements one inception module descirbed above as torch.nn.Module, which can then be stacked into a model by ConvNet 
    """
    def __init__(self, kernel_size, nb_features, nb_channels, nb_filters, bottleneck_size, depth):
        super().__init__()
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        # Define all the layers and modules we need in the forward pass: first the initial convolution and the parallel maxpooling 
        self.pad_conv_in = Pad_Conv(kernel_size=kernel_size)
        # This is the bottleneck convolution 
        self.conv_in = nn.Conv1d(in_channels=nb_channels if depth==0 else nb_features, 
                            out_channels=bottleneck_size, 
                            kernel_size=kernel_size, 
                            bias=False)
        self.pad_pool_in = Pad_Pool(left=1, right=1)
        self.maxpool_in = nn.MaxPool1d(kernel_size=3, stride=1)
        # 3 parallel convolutions taking the bottleneck as input
        self.conv1 = nn.Conv1d(in_channels=bottleneck_size, 
                            out_channels=nb_filters, 
                            kernel_size=kernel_size_s[0], 
                            bias=False)
        self.pad1 = Pad_Conv(kernel_size=kernel_size_s[0])
        self.conv2 = nn.Conv1d(in_channels=bottleneck_size, 
                            out_channels=nb_filters, 
                            kernel_size=kernel_size_s[1], 
                            bias=False)
        self.pad2 = Pad_Conv(kernel_size=kernel_size_s[1])
        self.conv3 = nn.Conv1d(in_channels=bottleneck_size, 
                            out_channels=nb_filters, 
                            kernel_size=kernel_size_s[2], 
                            bias=False)
        self.pad3 = Pad_Conv(kernel_size=kernel_size_s[2])
        # and the 4th parallel convolution following the maxpooling, no padding needed since 1x1 convolution 
        self.conv4 = nn.Conv1d(in_channels=nb_channels if depth==0 else nb_features,
                            out_channels=nb_filters,
                            kernel_size=1, 
                            bias=False)
        self.batchnorm = nn.BatchNorm1d(num_features=nb_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Implements a forward pass through one inception module
        """
        # Implement the left convolution 
        x_left = self.pad_conv_in(x)
        x_left = self.conv_in(x_left)
        # Implement the 3 parallel convolutions afterwards
        x_left1 = self.pad1(x_left)
        x_left1 = self.conv1(x_left1)
        x_left2 = self.pad2(x_left)
        x_left2 = self.conv2(x_left2)
        x_left3 = self.pad1(x_left)
        x_left3 = self.conv1(x_left3)
        # Implement the right maxpooling followed by a conv
        x_right = self.pad_pool_in(x)
        x_right = self.maxpool_in(x_right)
        x_right = self.conv4(x_right)
        # Concatenate the 4 outputs        
        x = torch.cat(tensors=(x_left1, x_left2, x_left3, x_right), dim=1) # concatenate along the feature dimension 
        x = self.batchnorm(x)
        return self.activation(x)