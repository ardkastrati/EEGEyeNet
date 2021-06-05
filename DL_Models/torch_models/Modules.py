import torch.nn as nn
import math 

class Pad_Pool(nn.Module):
    """
    Implements a padding layer in front of pool1d layers used in our architectures to achieve padding=same output shape 
    Pads 0 to the left and 1 to the right side of x 
    """
    def __init__(self, left=0, right=1, value=0):
        super().__init__()
        self.left = left
        self.right = right
        self.value = value 

    def forward(self, x):
        return nn.ConstantPad1d(padding=(self.left, self.right), value=self.value)(x)

class Pad_Pool2d(nn.Module):
    """
    Implements a padding layer in front of pool1d layers used in our architectures to achieve padding=same output shape 
    Pads 0 to the left and 1 to the right side of x 
    """
    def __init__(self, left=0, right=1, top=0, bottom=1, value=0):
        super().__init__()
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.value = value 

    def forward(self, x):
        return nn.ConstantPad2d(padding=(self.left, self.right, self.top, self.bottom), value=self.value)(x)

class Pad_Conv(nn.Module):
    """
    Implements a padding layer in front of conv1d layers used in our architectures to achieve padding=same output shape 
    Pads 0 to the left and 1 to the right side of x 
    """
    def __init__(self, kernel_size, value=0):
        super().__init__()
        self.value = value 
        self.left = max(math.floor(kernel_size/2)-1, 0)
        self.right = max(math.floor(kernel_size/2), 0)

    def forward(self, x):
        return nn.ConstantPad1d(padding=(self.left, self.right), value=self.value)(x)

class Pad_Conv2d(nn.Module):
    """
    Implements a padding layer in front of conv2d layers used in our architectures to achieve padding=same output shape 
    Pads 0 to the left and 1 to the right side of x 
    Input:
    kernel as a tuple (kx, ky)
    Output:
    Padded tensor for the following convolution s.t. padding=same
    """
    def __init__(self, kernel, value=0):
        super().__init__()
        kernel_x, kernel_y = kernel 
        self. value = value 
        self.left = max(math.floor(kernel_y/2)-1, 0)
        self.right = max(math.floor(kernel_y/2), 0)
        self.top = max(math.floor(kernel_x/2)-1, 0)
        self.bottom = max(math.floor(kernel_x/2), 0)

    def forward(self, x):
        return nn.ConstantPad2d((self.left, self.right, self.top, self.bottom), self.value)(x)


class TCSConv1d(nn.Module):
    """
    Implements a 1D separable convolution with constant tensor shape, similar to padding='same' in keras 
    """
    def __init__(self, mother, depth, bias=False):
        super(TCSConv1d, self).__init__()
        self.pad_depthwise = Pad_Conv(mother.kernel_size)
        # groups=in_channels makes it separable
        self.depthwise = nn.Conv1d(in_channels=mother.nb_channels if depth==0 else mother.nb_features, 
                            out_channels=mother.nb_channels if depth==0 else mother.nb_features,
                            groups=mother.nb_channels if depth==0 else mother.nb_features,
                            kernel_size=mother.kernel_size, 
                            bias=bias) 
        self.pointwise = nn.Conv1d(in_channels=mother.nb_channels if depth==0 else mother.nb_features,
                            out_channels=mother.nb_features, 
                            kernel_size=1,
                            bias=bias)

    def forward(self, x):
        x = self.pad_depthwise(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SeparableConv2d(nn.Module):
    """
    Implements a 2d separable convolution
    Not used in our default models 
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, 
                            in_channels, 
                            kernel_size=kernel_size, 
                            groups=in_channels, 
                            bias=bias, 
                            padding=padding)
        self.pointwise = nn.Conv2d(in_channels, 
                            out_channels, 
                            kernel_size=1, 
                            bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out