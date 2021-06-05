from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

our_models = {
    'LR_task' : {
        'antisaccade' : {
            'max' : {
                'KNN' : [KNeighborsClassifier, {'leaf_size': 10, 'n_neighbors': 100, 'n_jobs' : -1}],
                'GaussianNB' : [GaussianNB, {'var_smoothing': 0.0011513953993264468}],
                'LinearSVC' : [LinearSVC, {'C': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                #'RBF SVC' : [SVC, {'C': 0.1, 'gamma': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                #'DecisionTree' : [DecisionTreeClassifier, {'max_depth': 7}],
                #'RandomForest' : [RandomForestClassifier, {'max_depth': 10, 'n_estimators': 100, 'n_jobs' : -1}],
                #'GradientBoost' : [GradientBoostingClassifier, {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 50}],
                #'AdaBoost' : [AdaBoostClassifier, {'learning_rate': 0.5, 'n_estimators': 250}],
                #'XGBoost' : [XGBClassifier, {'objective' : 'binary:logistic', 'eval_metric' : 'logloss', 'eta': 0.1, 'max_depth': 15, 'n_estimators': 250, 'use_label_encoder' : False}]
            },
            'min' : {
                'KNN' : [KNeighborsClassifier, {'leaf_size': 10, 'n_neighbors': 10, 'n_jobs' : -1}],
                'GaussianNB' : [GaussianNB, {'var_smoothing': 0.0004941713361323833}],
                'LinearSVC' : [LinearSVC, {'C': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                #'RBF SVC' : [SVC, {'C': 1, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                #'DecisionTree' : [DecisionTreeClassifier, {'max_depth': 5}],
                #'RandomForest' : [RandomForestClassifier, {'max_depth': 10, 'n_estimators': 250, 'n_jobs' : -1}],
                #'GradientBoost' : [GradientBoostingClassifier, {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 50}],
                #'AdaBoost' : [AdaBoostClassifier, {'learning_rate': 0.5, 'n_estimators': 100}],
                #'XGBoost' : [XGBClassifier, {'objective' : 'binary:logistic', 'eval_metric' : 'logloss', 'eta': 0.1, 'max_depth': 5, 'n_estimators': 250, 'use_label_encoder' : False}]
            }
        }
    },
    'Direction_task' : {
        'dots' : {
            'max' : {
                'amplitude' : {
                    'KNN' : [KNeighborsRegressor, {'leaf_size': 10, 'n_neighbors': 100, 'n_jobs' : -1}],
                    'LinearReg' : [LinearRegression, {'n_jobs' : -1}],
                    'Ridge' : [Ridge, {'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [Lasso, {'alpha': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [ElasticNet, {'alpha': 1, 'l1_ratio' : 0.9, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [SVR, {'C': 100, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [RandomForestRegressor, {'max_depth': 50, 'n_estimators': 100, 'n_jobs' : -1}],
                    'GradientBoost' : [GradientBoostingRegressor, {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}],
                    'AdaBoost' : [AdaBoostRegressor, {'learning_rate': 0.01, 'n_estimators': 250}],
                    'XGBoost' : [XGBRegressor, {'eta': 0.1, 'max_depth': 5, 'n_estimators': 100}]
                },
                'angle' : {
                    'KNN' : [KNeighborsRegressor, {'leaf_size': 10, 'n_neighbors': 50, 'n_jobs' : -1}],
                    'LinearReg' : [LinearRegression, {'n_jobs' : -1}],
                    'Ridge' : [Ridge, {'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [Lasso, {'alpha': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [ElasticNet, {'alpha': 0.1, 'l1_ratio' : 0.3, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [SVR, {'C': 10, 'gamma': 0.1, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [RandomForestRegressor, {'max_depth': 10, 'n_estimators': 10, 'n_jobs' : -1}],
                    'GradientBoost' : [GradientBoostingRegressor, {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}],
                    'AdaBoost' : [AdaBoostRegressor, {'learning_rate': 0.01, 'n_estimators': 50}],
                    'XGBoost' : [XGBRegressor, {'eta': 0.1, 'max_depth': 5, 'n_estimators': 250}]
                }
            },
            'min' : {
                'amplitude' : {
                    'KNN' : [KNeighborsRegressor, {'leaf_size': 10, 'n_neighbors': 10, 'n_jobs' : -1}],
                    'LinearReg' : [LinearRegression, {'n_jobs' : -1}],
                    'Ridge' : [Ridge, {'alpha': 1000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [Lasso, {'alpha': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [ElasticNet, {'alpha': 1, 'l1_ratio' : 0.9, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [SVR, {'C': 100, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [RandomForestRegressor, {'max_depth': 50, 'n_estimators': 50, 'n_jobs' : -1}],
                    'GradientBoost' : [GradientBoostingRegressor, {'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 100}],
                    'AdaBoost' : [AdaBoostRegressor, {'learning_rate': 0.1, 'n_estimators': 50}],
                    'XGBoost' : [XGBRegressor, {'eta': 0.1, 'max_depth': 5, 'n_estimators': 250}]
                },
                'angle' : {
                    'KNN' : [KNeighborsRegressor, {'leaf_size': 10, 'n_neighbors': 25, 'n_jobs' : -1}],
                    'LinearReg' : [LinearRegression, {'n_jobs' : -1}],
                    'Ridge' : [Ridge, {'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [Lasso, {'alpha': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [ElasticNet, {'alpha': 0.1, 'l1_ratio' : 0.3, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [SVR, {'C': 10, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [RandomForestRegressor, {'max_depth': 10, 'n_estimators': 100, 'n_jobs' : -1}],
                    'GradientBoost' : [GradientBoostingRegressor, {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}],
                    'AdaBoost' : [AdaBoostRegressor, {'learning_rate': 0.5, 'n_estimators': 50}],
                    'XGBoost' : [XGBRegressor, {'eta': 0.05, 'max_depth': 5, 'n_estimators': 250}]
                }
            }
        },
        'processing_speed' : {
            'max' : {
                'amplitude' : {
                    #TODO
                },
                'angle' : {
                    #TODO
                }
            },
            'min' : {
                'amplitude' : {
                    #TODO
                },
                'angle' : {
                    #TODO
                }
            }
        } 
    },
    'Position_task' : {
        'dots' : {
            'max' : {
                'x' : {
                    'KNN' : [KNeighborsRegressor, {'leaf_size': 10, 'n_neighbors': 75, 'n_jobs' : -1}],
                    'LinearReg' : [LinearRegression, {'n_jobs' : -1}],
                    'Ridge' : [Ridge, {'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [Lasso, {'alpha': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [ElasticNet, {'alpha': 1, 'l1_ratio' : 0.3, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [SVR, {'C': 10, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [RandomForestRegressor, {'max_depth': 500, 'n_estimators': 50, 'n_jobs' : -1}],
                    'GradientBoost' : [GradientBoostingRegressor, {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}],
                    'AdaBoost' : [AdaBoostRegressor, {'learning_rate': 1, 'n_estimators': 10}],
                    'XGBoost' : [XGBRegressor, {'eta': 0.1, 'max_depth': 5, 'n_estimators': 100}]
                },
                'y' : {
                    'KNN' : [KNeighborsRegressor, {'leaf_size': 10, 'n_neighbors': 100, 'n_jobs' : -1}],
                    'LinearReg' : [LinearRegression, {'n_jobs' : -1}],
                    'Ridge' : [Ridge, {'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [Lasso, {'alpha': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [ElasticNet, {'alpha': 0.1, 'l1_ratio' : 0.9, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [SVR, {'C': 100, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [RandomForestRegressor, {'max_depth': 50, 'n_estimators': 100, 'n_jobs' : -1}],
                    'GradientBoost' : [GradientBoostingRegressor, {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 50}],
                    'AdaBoost' : [AdaBoostRegressor, {'learning_rate': 0.1, 'n_estimators': 50}],
                    'XGBoost' : [XGBRegressor, {'eta': 0.05, 'max_depth': 10, 'n_estimators': 100}]
                }
            },
            'min' : {
                'x' : {
                    'KNN' : [KNeighborsRegressor, {'leaf_size': 10, 'n_neighbors': 50, 'n_jobs' : -1}],
                    'LinearReg' : [LinearRegression, {'n_jobs' : -1}],
                    'Ridge' : [Ridge, {'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [Lasso, {'alpha': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [ElasticNet, {'alpha': 1, 'l1_ratio' : 0.6, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [SVR, {'C': 100, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [RandomForestRegressor, {'max_depth': 50, 'n_estimators': 250, 'n_jobs' : -1}],
                    'GradientBoost' : [GradientBoostingRegressor, {'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 100}],
                    'AdaBoost' : [AdaBoostRegressor, {'learning_rate': 0.01, 'n_estimators': 50}],
                    'XGBoost' : [XGBRegressor, {'eta': 0.1, 'max_depth': 10, 'n_estimators': 100}]
                },
                'y' : {
                    'KNN' : [KNeighborsRegressor, {'leaf_size': 10, 'n_neighbors': 100, 'n_jobs' : -1}],
                    'LinearReg' : [LinearRegression, {'n_jobs' : -1}],
                    'Ridge' : [Ridge, {'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [Lasso, {'alpha': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [ElasticNet, {'alpha': 1, 'l1_ratio' : 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [SVR, {'C': 100, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [RandomForestRegressor, {'max_depth': 50, 'n_estimators': 250, 'n_jobs' : -1}],
                    'GradientBoost' : [GradientBoostingRegressor, {'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 100}],
                    'AdaBoost' : [AdaBoostRegressor, {'learning_rate': 0.01, 'n_estimators': 250}],
                    'XGBoost' : [XGBRegressor, {'eta': 0.1, 'max_depth': 10, 'n_estimators': 250}]
                }
            }
        }
    }
}