from config import config
import numpy as np
import math
import logging

from Victor_files.sklearnclassifier import validate_classifiers, try_sklearn_classifiers
from Victor_files.sklearnregressor import validate_regressors, try_sklearn_regressors
from Victor_files.sklearndummy import try_sklearn_dummy

# return boolean arrays with length corresponding to n_samples
# the split is done based on the number of IDs
def split(trainY, train, val, test):
    assert (train+val+test == 1)
    IDs = np.unique(trainY[:,0])
    num_ids = len(IDs)

    # priority given to the test/val sets
    test_split = math.ceil(test * num_ids)
    val_split = math.ceil(val * num_ids)
    train_split = num_ids - val_split - test_split

    train = np.isin(trainY[:,0], IDs[:train_split])
    val = np.isin(trainY[:,0], IDs[train_split:train_split+val_split])
    test = np.isin(trainY[:,0], IDs[train_split+val_split:])

    return train, val, test

def crossvalidate_models(trainX, trainY, train, val, test, scale=False):
    logging.info('ML - ' + config['ml'])
    train, val, test = split(trainY, train, val, test)
    trainX = trainX.reshape((-1, 258))
    X = trainX[train | val]
    lab = config['label']
    y = trainY[:,lab][train | val]
    t = train[train | val] # reduce the length of train array

    if config['ml'] == 'c':
        validate_classifiers(X, y, t, scale=scale)
    elif config['ml'] == 'r':
        validate_regressors(X, y, t, scale=scale)