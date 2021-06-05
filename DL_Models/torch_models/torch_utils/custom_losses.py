"""
Definition of custum loss functions that can be used in our models
"""
import torch 

def angle_loss(a, b):
    """
    Custom loss function for models that predict the angle on the fix-sacc-fix dataset
    Angles -pi and pi should lead to 0 loss, since this is actually the same angle on the unit circle
    Angles pi/2 and -pi/2 should lead to a large loss, since this is a difference by pi on the unit circle
    Therefore we compute the absolute error of the "shorter" direction on the unit circle

    Inputs:
    a: predicted angle in rad
    b: target angle in rad 

    Output: reduced angle diff MSE of the batch 
    """
    return torch.mean(torch.square(torch.abs(torch.atan2(torch.sin(a - b), torch.cos(a - b)))))