import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from config import config
import seaborn as sns
import numpy as np
#import torch
import pandas as pd
import math 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from subprocess import call
import operator
import shutil
#import main cyclic import!

sns.set_style('darkgrid')
import logging

#TODO: Remove what we really don't need and document the others.
def plot_acc(hist, output_directory, model, val=False):
    '''
    plot the accuracy against the epochs during training
    '''
    epochs = len(hist.history['accuracy'])
    epochs = np.arange(epochs)
    plt.figure()
    plt.title(model + ' accuracy')
    plt.plot(epochs, hist.history['accuracy'],'b-',label='training')
    if val:
        plt.plot(epochs, hist.history['val_accuracy'],'g-',label='validation')

    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.savefig(output_directory + '/' + model + '_accuracy.png')
    # plt.show()
    logging.info(10*'*'+'\n')


def plot_loss(hist, output_directory, model, val=False, savefig=True):
    """
    Plot loss function of the trained model over the epochs 
    Works for both classification and regression, set config.py accordingly
    """
    
    """
    if config['task'] != 'prosaccade-clf':
        metric = "loss"
    else:
        metric = 'accuracy'
    """

    epochs = len(hist.history['loss'])
    epochs = np.arange(epochs)
    
    plt.figure()
    if config['pretrained']:
        plt.title("Pretrained " + model + ' loss')
    else:
        plt.title(model + ' loss')
    # plot the training curve
    plt.plot(epochs, np.array(hist.history['loss']), 'b-', label='training')
    # plot the validation curve 
    if val:
        plt.plot(epochs, np.array(hist.history['val_loss']),'g-',label='validation')
    plt.legend()
    plt.xlabel('epochs')

    if config['task'] == 'gaze-reg':
        plt.ylabel("MSE")
    elif config['task'] == 'angle-reg':
        plt.ylabel("Mean Absolute Angle Error")
    else:
        plt.ylabel('Binary Cross Entropy Loss')
        

    if savefig:
        plt.savefig(output_directory + '/' + model + '_loss.png')
    #plt.show()

def plot_loss_torch(loss, output_directory, model):
    epochs=np.arange(len(loss))
    plt.figure()
    plt.title(model + ' loss')
    plt.plot(epochs, loss, 'b-', label='training')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Binary Cross Entropy')
    plt.savefig(output_directory + '/' + model + 'loss.png')
    # plt.show()


def cp_dir(source, target):
    call(['cp', '-a', source, target])

def comparison_plot_accuracy():

    run_dir = './results/OHBM/'
    print(run_dir)
    plt.figure()
    plt.title('Comparison of the validation accuracy' )
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('accuracy (%)')
    
    for experiment in os.listdir(run_dir):
        name = experiment
        print(name)
        if(name != 'eegnet'):
            summary = pd.read_csv(run_dir+experiment+'/'+name+'_history.csv')
            acc = 100 * summary['val_accuracy']
            plt.plot(acc, '-' , label=name)

    plt.legend()
    plt.savefig(run_dir+'/comparison_accuracy.png')


def comparison_plot_loss():
    run_dir = './results/OHBM/'
    print(run_dir)
    plt.figure()
    plt.title('Comparison of the validation loss')
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('loss')

    for experiment in os.listdir(run_dir):
        name = experiment
        print(name)
        if (name != 'eegnet'):
            summary = pd.read_csv(run_dir + experiment + '/' + name + '_history.csv')
            acc = summary['val_loss']
            plt.plot(acc, '-', label=name)

    plt.legend()
    plt.savefig(run_dir + '/comparison_loss.png')


def select_best_model():
    results = {}
    model = {}
    run_dir = config['log_dir']
    #get best model in runs for all model_name
    for experiment in os.listdir(run_dir):
        number,name = experiment.split('_',1)
        if os.path.isdir(run_dir+experiment):
            try:
                summary = pd.read_csv(run_dir+experiment+'/'+name+'_df_best_model.csv')
                acc = float(summary['val_accuracy'])
                if not (name in results.keys()):

                    results[name] = acc
                    model[name] = number
                else:
                    if acc > results[name]:
                        results[name] = acc
                        model[name] = number
            except FileNotFoundError:
                pass

    # update the best model in th results folder with the ones in runs
    for name in model.keys():
        if os.path.exists(os.getcwd()+'/results/'+'best_'+name) and os.path.isdir(os.getcwd()+'/results/'+'best_'+name):
            acc = float(pd.read_csv(os.getcwd()+'/results/'+'best_'+name+'/'+name+'_df_best_model.csv')['val_accuracy'])
            if acc < results[name]:
                shutil.rmtree(os.getcwd()+'/results/'+'best_'+name)
                cp_dir(run_dir+model[name]+'_'+name,os.getcwd()+'/results/')
                os.rename(os.getcwd()+'/results/'+model[name]+'_'+name, os.getcwd()+'/results/'+'best_'+name)
            else:
                pass


# Save the logs
def save_logs(hist, output_directory, model, pytorch=False):
    # os.mkdir(output_directory)
    if pytorch:
        try:
            hist_df = pd.DataFrame(hist)
            hist_df.to_csv(output_directory + '/' + model + '_' + 'history.csv', index=False)
        except:
            return

    else:
        try:
            hist_df = pd.DataFrame(hist.history)
            hist_df.to_csv(output_directory + '/' + model + '_' + 'history.csv', index=False)

            #df_metrics = {'Accuracy': hist_df['accuracy'], 'Loss': hist_df['loss']}
            #df_metrics = pd.DataFrame(df_metrics)
            #df_metrics.to_csv(output_directory + '/' + model + '_' + 'df_metrics.csv', index=False)

            index_best_model = hist_df['val_accuracy'].idxmax()
            row_best_model = hist_df.loc[index_best_model]

            df_best_model = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                    columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc', 'best_model_val_acc'])

            df_best_model['best_model_train_loss'] = row_best_model['loss']
            df_best_model['best_model_val_loss'] = row_best_model['val_loss']
            df_best_model['best_model_train_acc'] = row_best_model['accuracy']
            df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']

            df_best_model.to_csv(output_directory + '/' + model + '_' + 'df_best_model.csv', index=False)
        except:
            return 

def train_val_split(x, y, val_size, subjectID):
    # subjectIDs = np.genfromtxt('subjects.csv', delimiter=',')
    subjectIDs = y[:, 1]
    y = y[:, 0]
    # Arrays must be reshaped from (n,) to (n, 1)
    subjectIDs = subjectIDs.reshape(-1, 1)
    y = y.reshape(-1, 1)

    if config['within_subjects']:
        # Get indices of desired subject
        indices = np.where(subjectIDs == subjectID)
        first_index = indices[0][0]
        last_index = indices[0][-1]
        # Cut the data and do usual train_test_split
        x = x[first_index: last_index + 1, :]
        y = y[first_index: last_index + 1]
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=val_size, random_state=42)
    elif config['across_subjects_v2']:
        # Make sure that all data of the subject at train_val_split is in training set
        index = math.ceil((1 - val_size) * subjectIDs.shape[0])
        # index = math.ceil(val_size * subjectIDs.shape[0])  # Have validation set at beginning
        subjectID = subjectIDs[index]
        while subjectIDs[index] == subjectID:
            index += 1
        # Data must be shuffled after the train_test_split
        X_train, X_val, y_train, y_val = train_test_split(x, y, train_size=index, shuffle=False)
        # X_val, X_train, y_val, y_train = train_test_split(x, y, train_size=index, shuffle=False)  # Have validation set at beginning
        shuffle(X_train, y_train, random_state=42)
        shuffle(X_val, y_val, random_state=42)
        # Cut the training data if desired
        X_train = X_train[:math.ceil(main.args.percentage * X_train.shape[0]), :]
        y_train = y_train[:math.ceil(main.args.percentage * y_train.shape[0])]
        # shuffle(X_train, y_train, random_state=42)
    else:
        # Simple across_subjects_v1
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=val_size, random_state=42)
        # Cut the training data if desired
        X_train = X_train[:math.ceil(main.args.percentage * X_train.shape[0]), :]
        y_train = y_train[:math.ceil(main.args.percentage * y_train.shape[0])]

    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    print('X_val: ', X_val.shape)
    print('y_val: ', y_val.shape)

    return X_train, X_val, y_train, y_val


# Save the model parameters (newly added without debugging)
# def save_model_param(classifier, output_directory, model, pytorch=False):
#     try:
#         if pytorch:
#             torch.save(classifier.state_dict(), output_directory + '/' + model + '_' + 'model.pth')
#         else:
#             classifier.save(output_directory + '/' + model + '_' + 'model.h5')
#     except:
#         return
