import numpy as np
import logging
import time
import csv
import sys

from sklearn.preprocessing import StandardScaler

from config import config

from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

train_models_c = {
    "Stratified" : DummyClassifier(),
    "MostFrequent" : DummyClassifier(),
    "Prior" : DummyClassifier(),
    "Uniform" : DummyClassifier()
    }

train_models_r = {
    "Mean" : DummyRegressor(strategy='mean'),
    "Median" : DummyRegressor(strategy='median'),
    "Constant" : DummyRegressor(strategy='constant', constant=300)
    }

train_models = {
    "c" : train_models_c,
    "r" : train_models_r
}

def try_sklearn_dummy(X, y, train, ml_type, scale=False):
    logging.info("Training the simple classifiers")
    
    if scale:
        logging.info('Standard Scaling')
        scaler = StandardScaler()
        scaler.fit(X[train])
        X = scaler.transform(X)

    X_train, y_train = X[train], y[train]
    X_test, y_test = X[~train], y[~train]

    tm = train_models[ml_type]

    models = []
    scores = []
    runtimes = []

    for model in tm:
        logging.info("Trainning of " + model)

        models.append(model)
        classifier = tm[model]

        start_time = time.time()
        classifier.fit(X_train, y_train.ravel())
        if ml_type == 'c':
            score = classifier.score(X_test, y_test.ravel())
        elif ml_type == 'r':
            score = mean_squared_error(y_test.ravel(), classifier.predict(X_test))
            #score = classifier.score(X_test, y_test.ravel())
        runtime = (time.time() - start_time)

        scores.append(score)
        runtimes.append(runtime)

        logging.info("--- Score: %s " % score)
        logging.info("--- Runtime: %s for seconds ---" % runtime)
        
    export_dict(models, scores, runtimes, first_row=('Model', 'Score', 'Runtime'), file_name='results.csv')

def export_dict(*columns, first_row, file_name):
    rows = zip(*columns)
    file = config['model_dir'] + '/' + file_name
    with open(file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(first_row)
        for row in rows:
            writer.writerow(row)