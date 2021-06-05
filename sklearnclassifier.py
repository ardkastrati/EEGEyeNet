# Some baselines to test

import numpy as np
import logging
import time
import csv
import sys
from config import config

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# default models for cross_validation
val_models = {
    "KNN" : KNeighborsClassifier(weights='uniform', algorithm='auto', n_jobs=-1),
    "GaussianNB" : GaussianNB(),
    "LinearSVC" : LinearSVC(tol=1e-2, max_iter=500),
    "RBF_SVC" : SVC(tol=1e-2, max_iter=500),
    "DTC" : DecisionTreeClassifier(random_state=42),
    "RFC" : RandomForestClassifier(max_features='auto', random_state=42, n_jobs=-1),
    "GBC" : GradientBoostingClassifier(max_features='auto', random_state=42),
    "ABC" : AdaBoostClassifier(random_state=42),
    "XGB" : XGBClassifier(objective="binary:logistic", eval_metric='logloss', random_state=42, use_label_encoder=False)
    }

# parameters for cross validation
params = {
    "KNN" : {'n_neighbors': [5, 10, 15, 25, 50, 75, 100], 'leaf_size': [10, 20, 35, 50]},
    "GaussianNB" : {'var_smoothing' : np.logspace(0, -9, num=50)},
    "LinearSVC" : {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    "RBF_SVC" : {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10, 100]},
    "DTC" : {'max_depth' : [3, 5, 7, 10, 15, 20, 30, 50]},
    "RFC" : {'n_estimators': [10, 50, 100, 250], 'max_depth': [5, 10, 50, 100, 500]},
    "GBC" : {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 15], 'learning_rate' : [0.01, 0.05, 0.1, 0.5]},
    "ABC" : {'n_estimators': [10, 50, 100, 250], 'learning_rate' : [0.01, 0.05, 0.1, 0.5, 1]},
    "XGB" : {'n_estimators': [10, 50, 100, 250], "max_depth": [5, 10, 15], 'eta': [0.01, 0.05, 0.1, 0.5, 1]}
    }

def cross_validate(model_name, X, y, train):
    logging.info("Cross-validation " + model_name + "...")
    classifier = val_models[model_name]
    parameters = params[model_name]

    # -1 for the training set, 0 for the validation set
    fold_arr = np.zeros(train.shape)
    fold_arr[train] = -1
    ps = PredefinedSplit(test_fold=fold_arr)

    clf = GridSearchCV(classifier, parameters, scoring='accuracy', verbose=3, cv=ps)
    clf.fit(X, y.ravel())

    export_dict(clf.cv_results_['mean_fit_time'], clf.cv_results_['std_fit_time'], clf.cv_results_['mean_score_time'],
                clf.cv_results_['std_score_time'], clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score'], \
                clf.cv_results_['params'], file_name=model_name+'_CrossValidation_results.csv',
                first_row=('mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                           'mean_test_score', 'std_test_score', 'params'))

    best_columns = ('BEST ESTIMATOR', 'BEST SCORE', 'BEST PARAMS')
    export_dict([clf.best_estimator_], [clf.best_score_], [clf.best_params_], file_name=model_name+'_best_results.csv', first_row=best_columns)

def validate_classifiers(X, y, train, scale=False):
    if(scale):
        logging.info('Standard Scaling')
        scaler = StandardScaler()
        scaler.fit(X[train])
        X = scaler.transform(X)
    cross_validate('KNN', X, y, train)
    cross_validate('GaussianNB', X, y, train)
    cross_validate('LinearSVC', X, y, train)
    cross_validate('RBF_SVC', X, y, train)
    cross_validate('DTC', X, y, train) 
    cross_validate('RFC', X, y, train)
    cross_validate('GBC', X, y, train)
    cross_validate('ABC', X, y, train)
    cross_validate('XGB', X, y, train)

def export_dict(*columns, first_row, file_name):
    rows = zip(*columns)
    file = config['model_dir'] + '/' + file_name
    with open(file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(first_row)
        for row in rows:
            writer.writerow(row)