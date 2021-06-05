from DL_Models.torch_models.ConvNetTorch import ConvNet
import torch.nn as nn
from DL_Models.torch_models.Modules import Pad_Conv, Pad_Pool


class PyramidalCNN(ConvNet):
    """
    The Classifier_PyramidalCNN is one of the simplest classifiers. It implements the class ConvNet, which is made of modules with a
    specific depth, where for each depth the number of filters is increased.
    """
    def __init__(self, loss, model_number, batch_size, input_shape, output_shape, kernel_size=16, epochs = 50, nb_filters=16, verbose=True,
                    use_residual=False, depth=6):
        """
        nb_features: specifies number of channels before the output layer 
        """
        self.nb_features = depth * nb_filters # For pyramidal we increase the nbfilters each depth layer
        super().__init__(loss=loss, model_number=model_number, batch_size=batch_size, input_shape=input_shape, output_shape=output_shape, kernel_size=kernel_size, epochs=epochs, nb_filters=nb_filters,
                            verbose=verbose, use_residual=use_residual, depth=depth)
        
    def _module(self, depth):
        """
        The module of CNN is made of a simple convolution with batch normalization and ReLu activation. Finally, MaxPooling is also used.
        The number of filters / output channels is increases with the depth of the model.
        Padding=same 
        """
        return nn.Sequential(
            Pad_Conv(kernel_size=self.kernel_size),
            nn.Conv1d(in_channels=129 if depth==0 else depth*self.nb_filters,
                        out_channels=(depth+1)*self.nb_filters, 
                        kernel_size=self.kernel_size, 
                        bias=False),
            nn.BatchNorm1d(num_features=(depth+1)*self.nb_filters),
            nn.ReLU(),
            Pad_Pool(left=0, right=1),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )