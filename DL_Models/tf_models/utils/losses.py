"""
Definition of custum loss functions that can be used in our models
"""

import tensorflow as tf

def angle_loss(a, b):
    """
    Custom loss function for models that predict the angle on the fix-sacc-fix dataset
    Angles -pi and pi should lead to 0 loss, since this is actually the same angle on the unit circle
    Angles pi/2 and -pi/2 should lead to a large loss, since this is a difference by pi on the unit circle
    Therefore we compute the absolute error of the "shorter" direction on the unit circle
    """
    return tf.reduce_mean(tf.math.square(tf.abs(tf.atan2(tf.sin(a - b), tf.cos(a - b)))))