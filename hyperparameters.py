from DL_Models import Ensemble
from StandardML_Models.StandardRegressor_1D import StandardRegressor_1D
from StandardML_Models.StandardRegressor_2D import StandardRegressor_2D
from StandardML_Models.StandardClassifier_1D import StandardClassifier_1D
from DL_Models.Ensemble import Ensemble
from config import config

# Use the following formats to add your own models (see hyperparameters.py for examples)
# 'NAME' : [MODEL, {'param1' : value1, 'param2' : value2, ...}]
# the model should be callable with MODEL(param1=value1, param2=value2, ...)

your_models = {
    'LR_task': {

    },
    'Direction_task': {
        'amplitude': {

        },
        'angle': {

        }
    },
    'Position_task': {

    }
}


our_ML_models = {
    'LR_task' : {
        'antisaccade' : {
            'max' : {
                'KNN' : [StandardClassifier_1D, {'model_name':'KNN', 'leaf_size': 10, 'n_neighbors': 100, 'n_jobs' : -1}],
                'GaussianNB' : [StandardClassifier_1D, {'model_name':'GaussianNB', 'var_smoothing': 0.0011513953993264468}],
                'LinearSVC' : [StandardClassifier_1D, {'model_name':'LinearSVC', 'C': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                'RBF SVC' : [StandardClassifier_1D, {'model_name':'RBF SVC', 'C': 0.1, 'gamma': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                'DecisionTree' : [StandardClassifier_1D, {'model_name':'DecisionTree', 'max_depth': 7}],
                'RandomForest' : [StandardClassifier_1D, {'model_name':'RandomForest', 'max_depth': 10, 'n_estimators': 100, 'n_jobs' : -1}],
                'GradientBoost' : [StandardClassifier_1D, {'model_name':'GradientBoost', 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 50}],
                'AdaBoost' : [StandardClassifier_1D, {'model_name':'AdaBoost', 'learning_rate': 0.5, 'n_estimators': 250}],
                'XGBoost' : [StandardClassifier_1D, {'model_name':'XGBoost', 'objective' : 'binary:logistic', 'eval_metric' : 'logloss', 'eta': 0.1, 'max_depth': 15, 'n_estimators': 250, 'use_label_encoder' : False}]
            },

            'min' : {
                'KNN' : [StandardClassifier_1D, {'model_name':'KNN', 'leaf_size': 10, 'n_neighbors': 10, 'n_jobs' : -1}],
                'GaussianNB' : [StandardClassifier_1D, {'model_name':'GaussianNB', 'var_smoothing': 0.0004941713361323833}],
                'LinearSVC' : [StandardClassifier_1D, {'model_name':'LinearSVC', 'C': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                'RBF SVC' : [StandardClassifier_1D, {'model_name':'RBF SVC', 'C': 1, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                'DecisionTree' : [StandardClassifier_1D, {'model_name':'DecisionTree', 'max_depth': 5}],
                'RandomForest' : [StandardClassifier_1D, {'model_name':'RandomForest', 'max_depth': 10, 'n_estimators': 250, 'n_jobs' : -1}],
                'GradientBoost' : [StandardClassifier_1D, {'model_name':'GradientBoost', 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 50}],
                'AdaBoost' : [StandardClassifier_1D, {'model_name':'AdaBoost', 'learning_rate': 0.5, 'n_estimators': 100}],
                'XGBoost' : [StandardClassifier_1D, {'model_name':'XGBoost', 'objective' : 'binary:logistic', 'eval_metric' : 'logloss', 'eta': 0.1, 'max_depth': 5, 'n_estimators': 250, 'use_label_encoder' : False}]
            }
        }
    },
    'Direction_task' : {
        'dots' : {
            'max' : {
                'amplitude' : {
                    'KNN' : [StandardRegressor_1D, {'model_name':'KNN', 'leaf_size': 10, 'n_neighbors': 100, 'n_jobs' : -1}],
                    'LinearReg' : [StandardRegressor_1D, {'model_name':'LinearReg', 'n_jobs' : -1}],
                    'Ridge' : [StandardRegressor_1D, {'model_name':'Ridge', 'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [StandardRegressor_1D, {'model_name':'Lasso', 'alpha': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [StandardRegressor_1D, {'model_name':'ElasticNet', 'alpha': 1, 'l1_ratio' : 0.9, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [StandardRegressor_1D, {'model_name':'RBF SVR', 'C': 100, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [StandardRegressor_1D, {'model_name':'RandomForest', 'max_depth': 50, 'n_estimators': 100, 'n_jobs' : -1}],
                    'GradientBoost' : [StandardRegressor_1D, {'model_name':'GradientBoost', 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}],
                    'AdaBoost' : [StandardRegressor_1D, {'model_name':'AdaBoost', 'learning_rate': 0.01, 'n_estimators': 250}],
                    'XGBoost' : [StandardRegressor_1D, {'model_name':'XGBoost', 'eta': 0.1, 'max_depth': 5, 'n_estimators': 100}]
                },
                'angle' : {
                    'KNN' : [StandardRegressor_1D, {'model_name':'KNN', 'leaf_size': 10, 'n_neighbors': 50, 'n_jobs' : -1}],
                    'LinearReg' : [StandardRegressor_1D, {'model_name':'LinearReg', 'n_jobs' : -1}],
                    'Ridge' : [StandardRegressor_1D, {'model_name':'Ridge', 'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [StandardRegressor_1D, {'model_name':'Lasso', 'alpha': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [StandardRegressor_1D, {'model_name':'ElasticNet', 'alpha': 0.1, 'l1_ratio' : 0.3, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [StandardRegressor_1D, {'model_name':'RBF SVR', 'C': 10, 'gamma': 0.1, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [StandardRegressor_1D, {'model_name':'RandomForest', 'max_depth': 10, 'n_estimators': 10, 'n_jobs' : -1}],
                    'GradientBoost' : [StandardRegressor_1D, {'model_name':'GradientBoost', 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}],
                    'AdaBoost' : [StandardRegressor_1D, {'model_name':'AdaBoost', 'learning_rate': 0.01, 'n_estimators': 50}],
                    'XGBoost' : [StandardRegressor_1D, {'model_name':'XGBoost', 'eta': 0.1, 'max_depth': 5, 'n_estimators': 250}]
                }
            },
            'min' : {
                'amplitude' : {
                    'KNN' : [StandardRegressor_1D, {'model_name':'KNN', 'leaf_size': 10, 'n_neighbors': 10, 'n_jobs' : -1}],
                    'LinearReg' : [StandardRegressor_1D, {'model_name':'LinearReg', 'n_jobs' : -1}],
                    'Ridge' : [StandardRegressor_1D, {'model_name':'Ridge', 'alpha': 1000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [StandardRegressor_1D, {'model_name':'Lasso', 'alpha': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [StandardRegressor_1D, {'model_name':'ElasticNet', 'alpha': 1, 'l1_ratio' : 0.9, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [StandardRegressor_1D, {'model_name':'RBF SVR', 'C': 100, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [StandardRegressor_1D, {'model_name':'RandomForest', 'max_depth': 50, 'n_estimators': 50, 'n_jobs' : -1}],
                    'GradientBoost' : [StandardRegressor_1D, {'model_name':'GradientBoost', 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 100}],
                    'AdaBoost' : [StandardRegressor_1D, {'model_name':'AdaBoost', 'learning_rate': 0.1, 'n_estimators': 50}],
                    'XGBoost' : [StandardRegressor_1D, {'model_name':'XGBoost', 'eta': 0.1, 'max_depth': 5, 'n_estimators': 250}]
                },
                'angle' : {
                    'KNN' : [StandardRegressor_1D, {'model_name':'KNN', 'leaf_size': 10, 'n_neighbors': 25, 'n_jobs' : -1}],
                    'LinearReg' : [StandardRegressor_1D, {'model_name':'LinearReg', 'n_jobs' : -1}],
                    'Ridge' : [StandardRegressor_1D, {'model_name':'Ridge', 'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'Lasso' : [StandardRegressor_1D, {'model_name':'Lasso', 'alpha': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'ElasticNet' : [StandardRegressor_1D, {'model_name':'ElasticNet', 'alpha': 0.1, 'l1_ratio' : 0.3, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RBF SVR' : [StandardRegressor_1D, {'model_name':'RBF SVR', 'C': 10, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                    'RandomForest' : [StandardRegressor_1D, {'model_name':'RandomForest', 'max_depth': 10, 'n_estimators': 100, 'n_jobs' : -1}],
                    'GradientBoost' : [StandardRegressor_1D, {'model_name':'GradientBoost', 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}],
                    'AdaBoost' : [StandardRegressor_1D, {'model_name':'AdaBoost', 'learning_rate': 0.5, 'n_estimators': 50}],
                    'XGBoost' : [StandardRegressor_1D, {'model_name':'XGBoost', 'eta': 0.05, 'max_depth': 5, 'n_estimators': 250}]
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
                'KNN' : [StandardRegressor_2D, {'model_name':'KNN', 'leaf_size': 10, 'n_neighbors': 75, 'n_jobs' : -1}],
                'LinearReg' : [StandardRegressor_2D, {'model_name':'LinearReg', 'n_jobs' : -1}],
                'Ridge' : [StandardRegressor_2D, {'model_name':'Ridge', 'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                'Lasso' : [StandardRegressor_2D, {'model_name':'Lasso', 'alpha': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                'ElasticNet' : [StandardRegressor_2D, {'model_name':'ElasticNet', 'alpha': 1, 'l1_ratio' : 0.3, 'tol' : 1e-5, 'max_iter' : 1200}],
                'RBF SVR' : [StandardRegressor_2D, {'model_name':'RBF SVR', 'C': 10, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                'RandomForest' : [StandardRegressor_2D, {'model_name':'RandomForest', 'max_depth': 500, 'n_estimators': 50, 'n_jobs' : -1}],
                'GradientBoost' : [StandardRegressor_2D, {'model_name':'GradientBoost', 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}],
                'AdaBoost' : [StandardRegressor_2D, {'model_name':'AdaBoost', 'learning_rate': 1, 'n_estimators': 10}],
                'XGBoost' : [StandardRegressor_2D, {'model_name':'XGBoost', 'eta': 0.1, 'max_depth': 5, 'n_estimators': 100}]
            },
            'min' : {
                'KNN' : [StandardRegressor_2D, {'model_name':'KNN', 'leaf_size': 10, 'n_neighbors': 50, 'n_jobs' : -1}],
                'LinearReg' : [StandardRegressor_2D, {'model_name':'LinearReg', 'n_jobs' : -1}],
                'Ridge' : [StandardRegressor_2D, {'model_name':'Ridge', 'alpha': 10000, 'tol' : 1e-5, 'max_iter' : 1200}],
                'Lasso' : [StandardRegressor_2D, {'model_name':'Lasso', 'alpha': 1, 'tol' : 1e-5, 'max_iter' : 1200}],
                'ElasticNet' : [StandardRegressor_2D, {'model_name':'ElasticNet', 'alpha': 1, 'l1_ratio' : 0.6, 'tol' : 1e-5, 'max_iter' : 1200}],
                'RBF SVR' : [StandardRegressor_2D, {'model_name':'RBF SVR', 'C': 100, 'gamma': 0.01, 'tol' : 1e-5, 'max_iter' : 1200}],
                'RandomForest' : [StandardRegressor_2D, {'model_name':'RandomForest', 'max_depth': 50, 'n_estimators': 250, 'n_jobs' : -1}],
                'GradientBoost' : [StandardRegressor_2D, {'model_name':'GradientBoost', 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 100}],
                'AdaBoost' : [StandardRegressor_2D, {'model_name':'AdaBoost', 'learning_rate': 0.01, 'n_estimators': 50}],
                'XGBoost' : [StandardRegressor_2D, {'model_name':'XGBoost', 'eta': 0.1, 'max_depth': 10, 'n_estimators': 100}]
            }
        }
    }
}

our_ML_dummy_models = {
    'LR_task' : {
        'antisaccade' : {
            'max' : {
                "Stratified" : [StandardClassifier_1D, {'model_name':'Stratified', 'strategy' : 'stratified'}],
                "MostFrequent" : [StandardClassifier_1D, {'model_name':'MostFrequent', 'strategy' : 'most_frequent'}],
                "Prior" : [StandardClassifier_1D, {'model_name':'Prior', 'strategy' : 'prior'}],
                "Uniform" : [StandardClassifier_1D, {'model_name': 'Uniform', 'strategy' : 'uniform'}]
            },
            'min' : {
                "Stratified" : [StandardClassifier_1D, {'model_name':'Stratified', 'strategy' : 'stratified'}],
                "MostFrequent" : [StandardClassifier_1D, {'model_name':'MostFrequent', 'strategy' : 'most_frequent'}],
                "Prior" : [StandardClassifier_1D, {'model_name':'Prior', 'strategy' : 'prior'}],
                "Uniform" : [StandardClassifier_1D, {'model_name': 'Uniform', 'strategy' : 'uniform'}]
            }
        }
    },
    'Direction_task' : {
        'dots' : {
            'max' : {
                'amplitude' : {
                "Mean" : [StandardRegressor_1D, {'model_name':'Mean', 'strategy' : 'mean'}],
                "Median" : [StandardRegressor_1D, {'model_name':'Median', 'strategy' : 'median'}]
                },
                'angle' : {
                "Mean" : [StandardRegressor_1D, {'model_name':'Mean', 'strategy' : 'mean'}],
                "Median" : [StandardRegressor_1D, {'model_name':'Median', 'strategy' : 'median'}]
                }
            },
            'min' : {
                'amplitude' : {
                "Mean" : [StandardRegressor_1D, {'model_name':'Mean', 'strategy' : 'mean'}],
                "Median" : [StandardRegressor_1D, {'model_name':'Median', 'strategy' : 'median'}]
                },
                'angle' : {
                "Mean" : [StandardRegressor_1D, {'model_name':'Mean', 'strategy' : 'mean'}],
                "Median" : [StandardRegressor_1D, {'model_name':'Median', 'strategy' : 'median'}]
                }
            }
        },
        'processing_speed' : {
            'max' : {
                'amplitude' : {

                },
                'angle' : {

                }
            },
            'min' : {
                'amplitude' : {

                },
                'angle' : {

                }
            }
        }
    },
    'Position_task' : {
        'dots' : {
            'max' : {
                "Mean" : [StandardRegressor_2D, {'model_name':'Mean', 'strategy' : 'mean'}],
                "Median" : [StandardRegressor_2D, {'model_name':'Median', 'strategy' : 'median'}],
                "Constant" : [StandardRegressor_2D, {'model_name':'Constant', 'strategy' : 'constant'}]
            },
            'min' : {
                "Mean" : [StandardRegressor_2D, {'model_name':'Mean', 'strategy' : 'mean'}],
                "Median" : [StandardRegressor_2D, {'model_name':'Median', 'strategy' : 'median'}],
                "Constant" : [StandardRegressor_2D, {'model_name':'Constant', 'strategy' : 'constant'}]
            }
        }
    }
}

nb_models = 1
batch_size = 64
input_shape = (1, 258) if config['feature_extraction'] else (500, 129)
depth = 12
epochs = 50
verbose = True

our_DL_models = {
    'LR_task' : {
        'antisaccade' : {
            'max' : {
                'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'bce', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'bce', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'bce', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'bce', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'bce', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]
             },
            'min' : {
                'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'bce', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'bce', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'bce', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'bce', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'bce', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]
            }
        }
    },
    'Direction_task' : {
        'dots' : {
            'max' : {
                'amplitude' : {
                    'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                    'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                    'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]

                },
                'angle' : {
                    'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                    'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                    'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                             'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]
                }
            },
            'min' : {
                'amplitude' : {
                    'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                    'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                    'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                             'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]
                },
                'angle' : {
                    'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                    'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                    'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]
                }
            }
        },
        'processing_speed' : {
            'max' : {
                'amplitude' : {
                    'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                    'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                    'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]

                },
                'angle' : {
                    'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                    'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                    'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                             'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]
                }
            },
            'min' : {
                'amplitude' : {
                    'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                    'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                    'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                             'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]
                },
                'angle' : {
                    'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                    'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                    'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                    'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'angle-loss', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 1,
                                              'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]
                }
            }
        }
    },
    'Position_task' : {
        'dots' : {
            'max' : {
                'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 2,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 2,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 2,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 2,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 2,
                                              'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]
            },
            'min' : {
                'CNN' : [Ensemble, {'model_name': 'CNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 2,
                                    'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'EEGNet' : [Ensemble, {'model_name' : 'EEGNet', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 2,
                                       'epochs' : epochs, 'F1' : 16, 'F2' : 256, 'verbose' : verbose, 'D' : 4, 'kernel_size' : 256, 'dropout_rate' : 0.5}],
                'InceptionTime' : [Ensemble, {'model_name': 'InceptionTime', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 2,
                                              'kernel_size': 64, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : True, 'depth' : depth}],
                'PyramidalCNN' : [Ensemble, {'model_name': 'PyramidalCNN', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 2,
                                              'kernel_size': 16, 'epochs' : epochs, 'nb_filters' : 16, 'verbose' : verbose, 'use_residual' : False, 'depth' : 6}],
                'Xception' : [Ensemble, {'model_name': 'Xception', 'nb_models' : nb_models, 'loss':'mse', 'batch_size': batch_size, 'input_shape': input_shape, 'output_shape' : 2,
                                              'kernel_size': 40, 'epochs' : epochs, 'nb_filters' : 64, 'verbose' : verbose, 'use_residual' : True, 'depth' : 18}]
            }
        }
    }
}
# merge two dict, new_dict overrides base_dict in case of incompatibility
def merge_models(base_dict, new_dict):
    result = dict()
    keys = base_dict.keys() | new_dict.keys()
    for k in keys:
        if k in base_dict and k in new_dict:
            if type(base_dict[k]) == dict and type(new_dict[k]) == dict:
                result[k] = merge_models(base_dict[k], new_dict[k])
            else:
                # overriding
                result[k] = new_dict[k]
        elif k in base_dict:
            result[k] = base_dict[k]
        else:
            result[k] = new_dict[k]
    return result

all_models = dict()

if config['include_ML_models']:
    all_models = merge_models(all_models, our_ML_models)
if config['include_DL_models']:
    all_models = merge_models(all_models, our_DL_models)
if config['include_dummy_models']:
    all_models = merge_models(all_models, our_ML_dummy_models)
if config['include_your_models']:
    all_models = merge_models(all_models, your_models)