import numpy as np
import logging
import time
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from config import config

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


def try_models(X, y, train, models, N=5, scoring=None, scale=False, save_trail=''):
    logging.info("Training the models")
    
    if scale:
        logging.info('Standard Scaling')
        scaler = StandardScaler()
        scaler.fit(X[train])
        X = scaler.transform(X)

    X_train, y_train = X[train], y[train]
    X_test, y_test = X[~train], y[~train]

    all_runs = []
    statistics = []

    for name, model in models.items():
        logging.info("Training of " + name)

        model_runs = []

        for i in range(N):
            # create the model with the corresponding parameters
            classifier = model[0](**model[1])

            start_time = time.time()
            classifier.fit(X_train, y_train.ravel())
            if scoring is None:
                score = classifier.score(X_test, y_test.ravel())
            else:
                score = scoring(y_test.ravel(), classifier.predict(X_test))
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

    train, val, test = split(trainY, 0.7, 0.15, 0.15)
    models = config['models']

    # TODO: Because the datasets are (nb_samples, 1, 258) and it changes it to (nb_samples, 258) -> NEEDS TO BE ADAPTED
    trainX = trainX.reshape((-1, 258))

    X = trainX[train | test]
    t = train[train | test] # TODO Not so understandable, but works :) -> t has indexes of train data to true and false to test (validation is removed).

    if config['task'] == 'LR_task':
        if config['dataset'] == 'antisaccade':
            y = trainY[:,1][train | test] # The first column are the Id-s, we take the second which are labels
            try_models(X, y, t, models)
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Direction_task':
        if config['dataset'] == 'dots':
            scoring = (lambda y, y_pred : np.sqrt(mean_squared_error(y, y_pred)))
            y = trainY[:,1][train | test] # The first column are the Id-s, we take the second which are amplitude labels
            try_models(X, y, t, models['amplitude'], scoring=scoring, save_trail='_amplitude')
            y = trainY[:,2][train | test] # The first column are the Id-s, secon are the amplitude labels, we take the third which are the angle labels
            try_models(X, y, t, models['angle'], scoring=scoring, save_trail='_angle')
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Position_task':
        if config['dataset'] == 'dots':
            scoring = (lambda y, y_pred : np.sqrt(mean_squared_error(y, y_pred)))
            y = trainY[:,1][train | test]
            try_models(X, y, t, models['x'], scoring=scoring, save_trail='_x')
            y = trainY[:,2][train | test]
            try_models(X, y, t, models['y'], scoring=scoring, save_trail='_y')
        else:
            raise ValueError("This task cannot be predicted (is not implemented yet) with the given dataset.")

    else:
        raise NotImplementedError(f"Task {config['task']} is not implemented yet.")