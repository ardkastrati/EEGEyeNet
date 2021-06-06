import numpy as np
import logging
import time
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from config import config
from hyperparameters import allmodels
import os


# return boolean arrays with length corresponding to n_samples
# the split is done based on the number of IDs
def split(ids, train, val, test):
    assert (train+val+test == 1)
    IDs = np.unique(ids)
    num_ids = len(IDs)

    # priority given to the test/val sets
    test_split = math.ceil(test * num_ids)
    val_split = math.ceil(val * num_ids)
    train_split = num_ids - val_split - test_split

    train = np.isin(ids, IDs[:train_split])
    val = np.isin(ids, IDs[train_split:train_split+val_split])
    test = np.isin(ids, IDs[train_split+val_split:])

    return train, val, test


def try_models(trainX, trainY, ids, models, N=5, scoring=None, scale=False, save_trail='', save=False):

    logging.info("Training the models")
    train, val, test = split(ids, 0.7, 0.15, 0.15)
    X_train, y_train = trainX[train], trainY[train]
    X_val, y_val = trainX[val], trainY[val]
    X_test, y_test = trainX[test], trainY[test]


    if scale:
        logging.info('Standard Scaling')
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    all_runs = []
    statistics = []

    for name, model in models.items():
        logging.info("Training of " + name)

        model_runs = []

        for i in range(N):
            # create the model with the corresponding parameters
            trainer = model[0](**model[1])
            logging.info(trainer)
            start_time = time.time()

            # Taking care of saving and loading
            path = config['checkpoint_dir'] + 'run' + str(i + 1) + '/'
            if not os.path.exists(path):
                os.makedirs(path)

            if config['retrain']:
                trainer.fit(X_train, y_train, X_val, y_val)
            else:
                trainer.load(path)

            if config['save_models']:
                trainer.save(path)

            #print(y_test.ravel().shape)
            #print(trainer.predict(X_test).shape)
            score = scoring(y_test, trainer.predict(X_test))

            runtime = (time.time() - start_time)
            all_runs.append([name, score, runtime])
            model_runs.append([score, runtime])

            logging.info("--- Score: %s " % score)
            logging.info("--- Runtime: %s for seconds ---" % runtime)
        
        model_runs = np.array(model_runs)
        model_scores, model_runtimes = model_runs[:,0], model_runs[:,1]
        statistics.append([name, model_scores.mean(), model_scores.std(), model_runtimes.mean(), model_runtimes.std()])

    np.savetxt(config['model_dir']+'/runs'+save_trail+'.csv', all_runs, fmt='%s', delimiter=',', header='Model,Score,Runtime', comments='')
    np.savetxt(config['model_dir']+'/statistics'+save_trail+'.csv', statistics, fmt='%s', delimiter=',', header='Model,Mean_score,Std_score,Mean_runtime,Std_runtime', comments='')

def benchmark(trainX, trainY):
    np.savetxt(config['model_dir']+'/config.csv', [config['task'], config['dataset'], config['preprocessing']], fmt='%s')
    models = allmodels[config['task']][config['dataset']][config['preprocessing']]

    ids = trainY[:, 0]

    if config['task'] == 'LR_task':
        if config['dataset'] == 'antisaccade':
            scoring = (lambda y, y_pred: accuracy_score(y.ravel(), y_pred))  # Subject to change to mean euclidean distance.
            y = trainY[:,1] # The first column are the Id-s, we take the second which are labels
            try_models(trainX=trainX, trainY=y, ids=ids, models=models, scoring=scoring)
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Direction_task':
        if config['dataset'] == 'dots':
            scoring = (lambda y, y_pred : np.sqrt(mean_squared_error(y.ravel(), y_pred)))
            y1 = trainY[:,1] # The first column are the Id-s, we take the second which are amplitude labels
            try_models(trainX=trainX, trainY=y1, ids=ids, models=models['amplitude'], scoring=scoring, save_trail='_amplitude')
            y2 = trainY[:,2] # The first column are the Id-s, second are the amplitude labels, we take the third which are the angle labels
            try_models(trainX=trainX, trainY=y2, ids=ids, models=models['angle'], scoring=scoring, save_trail='_angle')
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Position_task':
        if config['dataset'] == 'dots':
            scoring = (lambda y, y_pred : np.sqrt(mean_squared_error(y, y_pred))) # Subject to change to mean euclidean distance.
            scoring2 = (lambda y, y_pred: np.linalg.norm(y - y_pred, axis=1).mean()) # Euclidean distance
            y = trainY[:,1:] # The first column are the Id-s, the second and third are position x and y which we use
            try_models(trainX=trainX, trainY=y, ids=ids, models=models, scoring=scoring2)
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")
    else:
        raise NotImplementedError(f"Task {config['task']} is not implemented yet.")