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
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

# default models for cross_validation
val_models = {
    "KNN" : KNeighborsRegressor(weights='uniform', algorithm='auto', n_jobs=-1),
    "LinearReg" : LinearRegression(n_jobs=-1),
    "Ridge" : Ridge(tol=1e-2, max_iter=500, random_state=42),
    "Lasso" : Lasso(tol=1e-2, max_iter=500, random_state=42),
    "ElasticNet" : ElasticNet(tol=1e-2, max_iter=500, random_state=42),
    "RBF_SVR" : SVR(tol=1e-2, max_iter=500),
    "DTR" : DecisionTreeRegressor(random_state=42),
    "RFR" : RandomForestRegressor(max_features='auto', random_state=42, n_jobs=-1),
    "GBR" : GradientBoostingRegressor(max_features='auto', random_state=42),
    "ABR" : AdaBoostRegressor(random_state=42),
    "XGB" : XGBRegressor(random_state=42)
    }

# parameters for cross validation
params = {
    "KNN" : {'n_neighbors': [5, 10, 15, 25, 50, 75, 100], 'leaf_size': [10, 20, 35, 50]},
    "LinearReg" : {},
    "Ridge" : {'alpha' : [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
    "Lasso" : {'alpha' : [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
    "ElasticNet" : {'alpha' : [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'l1_ratio' : [0.01, 0.05, 0.1, 0.3, 0.6, 0.9]},
    "RBF_SVR" : {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10, 100]},
    "DTR" : {'max_depth' : [3, 5, 7, 10, 15, 20, 30, 50]},
    "RFR" : {'n_estimators': [10, 50, 100, 250], 'max_depth': [5, 10, 50, 100, 500]},
    "GBR" : {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 15], 'learning_rate' : [0.01, 0.05, 0.1, 0.5]},
    "ABR" : {'n_estimators': [10, 50, 100, 250], 'learning_rate' : [0.01, 0.05, 0.1, 0.5, 1]},
    "XGB" : {'n_estimators': [10, 50, 100, 250], "max_depth": [5, 10, 15], 'eta': [0.01, 0.05, 0.1, 0.5, 1]}
    }

def cross_validate(model_name, X, y, train):
    logging.info("Cross-validation " + model_name + "...")
    classifier = val_models[model_name]
    parameters = params[model_name]

    # -1 for the train set, 0 for the validation set
    fold_arr = np.zeros(train.shape)
    fold_arr[train] = -1
    ps = PredefinedSplit(test_fold=fold_arr)

    clf = GridSearchCV(classifier, parameters, verbose=3, cv=ps)
    clf.fit(X, y.ravel())

    export_dict(clf.cv_results_['mean_fit_time'], clf.cv_results_['std_fit_time'], clf.cv_results_['mean_score_time'],
                clf.cv_results_['std_score_time'], clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score'], \
                clf.cv_results_['params'], file_name=model_name+'_CrossValidation_results.csv',
                first_row=('mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                           'mean_test_score', 'std_test_score', 'params'))

    best_columns = ('BEST ESTIMATOR', 'BEST SCORE', 'BEST PARAMS')
    export_dict([clf.best_estimator_], [clf.best_score_], [clf.best_params_], file_name=model_name+'_best_results.csv', first_row=best_columns)

def validate_regressors(X, y, train, scale=False):
    if(scale):
        logging.info('Standard Scaling')
        scaler = StandardScaler()
        scaler.fit(X[train])
        X = scaler.transform(X)
    cross_validate("KNN", X, y, train)
    cross_validate("LinearReg", X, y, train)
    cross_validate("Ridge", X, y, train)
    cross_validate("Lasso", X, y, train)
    cross_validate("ElasticNet", X, y, train)
    cross_validate("RBF_SVR", X, y, train)
    #cross_validate("DTR", X, y, train)
    cross_validate("RFR", X, y, train)
    cross_validate("GBR", X, y, train)
    cross_validate("ABR", X, y, train)
    cross_validate("XGB", X, y, train)

def export_dict(*columns, first_row, file_name):
    rows = zip(*columns)
    file = config['model_dir'] + '/' + file_name
    with open(file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(first_row)
        for row in rows:
            writer.writerow(row)