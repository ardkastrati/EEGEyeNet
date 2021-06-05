import numpy as np
import matplotlib.pyplot as plt
from config import config
import logging 

def plot_from_csv(file_path, output_dir, metric, savefig=True):
    """
    Plot the metric saved in the file_path file 
    """
    logging.info("Plotting metrics...")
    x = np.loadtxt(file_path, delimiter=',')
    epochs = np.arange(len(x))
    plt.figure()
    if config['pretrained']:
        plt.title("Pretrained " + config['model'] + ' loss')
    else:
        plt.title(config['model'] + ' loss')
    plt.plot(epochs, x, 'b-', label='validation')
    plt.legend()
    plt.xlabel('epochs')

    if config['task'] == 'gaze-reg':
        plt.ylabel("MSE")
    elif config['task'] == 'angle-reg':
        plt.ylabel("Mean Absolute Angle Error")
    else:
        plt.ylabel('Binary Cross Entropy Loss')
    if savefig:
        plt.savefig(output_dir + '/plots/' + config['model'] + '_val_' + metric + '.png')

def plot_array(x, output_dir, metric, savefig=True):
    """
    Plot the metric saved in the file_path file 
    """
    logging.info("Plotting metrics...")
    epochs = np.arange(len(x))
    plt.figure()
    if config['pretrained']:
        plt.title("Pretrained " + config['model'] + " " + metric)
    else:
        plt.title(config['model'] + " " + metric)
    plt.plot(epochs, np.array(x), 'b-', label='validation')
    plt.legend()
    plt.xlabel('epochs')

    if config['task'] == 'gaze-reg':
        plt.ylabel("MSE")
    elif metric == 'accuracy':
        plt.ylabel("Accuracy")
    elif config['task'] == 'angle-reg':
        plt.ylabel("Mean Absolute Angle Error")
    else:
        plt.ylabel('Binary Cross Entropy Loss')
    if savefig:
        plt.savefig(output_dir + '/plots/' + config['model'] + '_val_' + metric + '.png')

def plot_metrics(train, val, output_dir, metric, model_number=0 ,savefig=True):
    """
    Plot the training and validation metric together in one image 
    """
    logging.info("Plotting training and validation metrics ...")
    epochs = np.arange(len(train))
    plt.figure()
    if config['pretrained']:
        plt.title("Pretrained " + config['model'] + " " + metric)
    else:
        plt.title(config['model'] + " " + str(model_number) + " " + metric)
    plt.plot(epochs, np.array(train), 'b-', label='train')
    plt.plot(epochs, np.array(val), 'g-', label='validation')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel(metric)
    
    if savefig:
        plt.savefig(output_dir + '/plots/' + config['model'] + '_' +str(model_number) + "_" + metric + '.png')