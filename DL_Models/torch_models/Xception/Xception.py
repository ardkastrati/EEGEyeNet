from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from DL_Models.torch_models.ConvNetTorch import ConvNet
import torch.nn as nn
from DL_Models.torch_models.Modules import Pad_Conv, Pad_Pool, TCSConv1d


class XCEPTION(ConvNet):
    """
    The Xception architecture. This is inspired by Xception paper, which describes how 'extreme' convolutions can be represented
    as separable convolutions and can achieve better accuracy than the Inception architecture. It is made of modules in a specific depth.
    Each module, in our implementation, consists of a separable convolution followed by batch normalization and a ReLu activation layer.
    """
    def __init__(self, loss, model_number, batch_size, input_shape, output_shape, kernel_size=40, nb_filters=64, verbose=True, epochs=1,
                        use_residual=True, depth=6):
        """
        nb_features: specifies number of channels before the output layer 
        """
        self.nb_features = nb_filters # Exception passes a tensor of shape (timesamples, nb_filters) through the network
        super(XCEPTION, self).__init__(loss=loss, model_number=model_number, batch_size=batch_size, input_shape=input_shape, output_shape=output_shape, kernel_size=kernel_size,
                                       nb_filters=nb_filters, verbose=verbose, epochs=epochs, use_residual=use_residual, depth=depth, preprocessing=False)

    def _module(self, depth):
        """
        The module of Xception. Consists of a separable convolution followed by batch normalization and a ReLu activation function.
        Padding=same 
        """
        return nn.Sequential(
            TCSConv1d(mother=self, depth=depth, bias=False),
            nn.BatchNorm1d(num_features=self.nb_features),
            nn.ReLU()
        )