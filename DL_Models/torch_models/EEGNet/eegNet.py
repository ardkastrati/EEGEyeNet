import torch.nn as nn
from DL_Models.torch_models.BaseNetTorch import BaseNet
import logging 
from DL_Models.torch_models.Modules import Pad_Conv2d

class EEGNet(BaseNet):
    """
    The EEGNet architecture used as baseline. This is the architecture explained in the paper
    'EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces' with authors
    Vernon J. Lawhern, Amelia J. Solon, Nicholas R. Waytowich, Stephen M. Gordon, Chou P. Hung, Brent J. Lance

    In our implementation it is built on BaseNet and can therefore use the same interface for training as models based on ConvNet. 
    We only define the layers we need, the forward pass, and a method that returns the number of hidden units before the output layer, which is accessed by BaseNet to create the same output layer as for ConvNet models. 

    
    """
    def __init__(self, loss, model_number, batch_size, input_shape, output_shape, epochs=50,
                F1=16, F2=256, verbose=True, D=4, kernel_size=256,
                dropout_rate=0.5):
        """
        nb_features: specifies number of channels before the output layer 
        Padding=same 
        """
        self.kernel_size = kernel_size
        self.timesamples = input_shape[0]
        self.channels = input_shape[1]
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

        super().__init__(loss, input_shape=input_shape, output_shape=output_shape, epochs=epochs, model_number=model_number, verbose=verbose)
        
        # Block 1: 2dconv and depthwise conv
        self.padconv1 = Pad_Conv2d(kernel=(1,self.kernel_size))
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=self.F1,
            kernel_size=(1, self.kernel_size),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(self.F1, False)
        self.depthwise_conv1 = nn.Conv2d(
            in_channels=self.F1,
            out_channels=self.F1 * self.D,
            groups=self.F1,
            kernel_size=(self.channels, 1),
            bias=False
        )
        self.batchnorm1_2 = nn.BatchNorm2d(self.F1 * self.D)
        self.activation1 = nn.ELU()
        self.padpool1 = Pad_Conv2d(kernel=(1,16))
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1,16), stride=1)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        # Block 2: separable conv = depthwise + pointwise 
        self.pad_depthwise2 = Pad_Conv2d(kernel=(1,64))
        self.depthwise_conv2 = nn.Conv2d(
            in_channels=self.F1 * self.D,
            out_channels=self.F2,
            groups=self.F1*self.D,
            kernel_size=(1,64),
            bias=False 
        )
        self.pointwise_conv2 = nn.Conv2d( # no need for padding or pointwise 
            in_channels=self.F2,
            out_channels=4,
            kernel_size=1,
            bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.activation2 = nn.ELU()
        self.padpool2 = Pad_Conv2d(kernel=(1,8))
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1,8), stride=1)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        logging.info(f"Number of model parameters: {sum(p.numel() for p in self.parameters())}")
        logging.info(f"Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

        if verbose and self.model_number == 0:
            #print(self)
            pass

    def forward(self, x):
        """
        Implements a forward pass of the eegnet.
        """
        # Block 1
        x = self.padconv1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv1(x)
        x = self.batchnorm1_2(x)
        x = self.activation1(x)
        x = self.padpool1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        # Block2
        x = self.pad_depthwise2(x)
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.padpool2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        # Reshape
        x = x.view(self.batch_size, -1)
        # Output
        x = self.output_layer(x)

        return x

    def get_nb_features_output_layer(self):
        """
        Return number of features passed into the output layer of the network 
        """
        # 4 channels before the output layer 
        return 4 * self.timesamples 