from config import config
import numpy as np
import scipy.io as sio
import os
import pickle
import logging

def get_npz_data(data_dir, verbose=True):
    if verbose:
        logging.info('Loading ' + config['all_EEG_file'])
    with np.load(data_dir + config['all_EEG_file']) as f:
        X = f[config['trainX_variable']]
        if verbose:
            logging.info("X training loaded.")
            logging.info(X.shape)
        y = f[config['trainY_variable']]
        if verbose:
            logging.info("y training loaded.")
            logging.info(y.shape)
    return X, y

def store(x, y, clip=True):
    if clip:
        x = x[:10000]
        y = y[:10000]
    output_x = open('x_clip.pkl', 'wb')
    pickle.dump(x, output_x)
    output_x.close()

    output_y = open('y_clip.pkl', 'wb')
    pickle.dump(y, output_y)
    output_y.close()