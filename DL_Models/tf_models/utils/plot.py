import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf 
from matplotlib.ticker import FormatStrFormatter
import os
from config import config 


def plot_batches_log_loss(model_name):
    """
    Create loss and validation loss plots from the batches.log file
    """
    dir = './runs/' # must be correct relative to caller
    path = dir + model_name + 'batches.log'
    df = pd.read_csv(path, sep=';')
    nparr = df.to_numpy()
    epochs = nparr[:, 0]
    loss = nparr[:, 1]
    val_loss= nparr[:, 3]

    plt.plot(epochs, loss, label='loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.ylabel("mse loss")
    plt.title("mse")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    save_path = '../images/'
    plt.savefig(fname=save_path+model_name)

def plot_model(model, dir=config['model_dir'], show_shapes=True):
    """
    Plot the model as graph and save it
    """
    pathname = dir + "/model_plot.png"
    keras.utils.plot_model(model, to_file=pathname, show_shapes=show_shapes)






"""
Following some functionality for the gradient ascent method to compute inputs that maximize specific kernel activations 
"""
def compute_loss(input_image, filter_index, feature_extractor):
    """
    Part of the gradient ascent algorithm to maximize filter activation
    """ 
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, filter_index]
    return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(img, filter_index, learning_rate, feature_extractor):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index, feature_extractor)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def initialize_image(img_width, img_height):
    # We start from a gray image with some random noise 
    img = tf.random.uniform((1, img_width, img_height))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25

def visualize_filter(filter_index, img_width, img_height, feature_extractor):
    # We run gradient ascent for 30 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image(img_width, img_height)
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate, feature_extractor)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img

def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    #img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img