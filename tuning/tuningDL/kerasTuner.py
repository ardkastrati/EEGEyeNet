from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from config import config
import logging

from CNN.CNN import Classifier_CNN
from DeepEye.deepeye import Classifier_DEEPEYE
from DeepEyeRNN.deepeyeRNN import Classifier_DEEPEYE_RNN
from Xception.Xception import Classifier_XCEPTION
from InceptionTime.Inception import Classifier_INCEPTION
from EEGNet.eegNet import Classifier_EEGNet


def build_model(hp):
    classifier = None
    logging.info('Starting tuning ' + config['model'])
    if config['model'] == 'deepeye':
        classifier = Classifier_DEEPEYE(input_shape=config['deepeye']['input_shape'],
                                        epochs=15, verbose=True, batch_size=64, use_residual=True,
                                        kernel_size=hp.Choice('kernel_size', values=[32, 40, 64]),
                                        nb_filters=hp.Choice('nb_filters', values=[32, 64]),
                                        depth=hp.Int('depth',min_value=6,max_value=14, step=4),
                                        bottleneck_size=hp.Choice('bottleneck_size', values=[32, 64]),
                                        use_simple_convolution= hp.Choice('use_simple_convolution', values=[True, False]),
                                        use_separable_convolution= hp.Choice('use_separable_convolution', values = [True, False]),
                                        preprocessing=False)

                                        # use_separable_convolution=hp.Choice('use_separable_convolution', values=[True, False]),
                                        # preprocessing_F1 = hp.Choice('preprocessing_F1', values=[8, 16, 32, 64]),
                                        # preprocessing_D = hp.Choice('preprocessing_F1', values=[2, 4, 6, 8]),
                                        # preprocessing_kernLength = hp.Choice('preprocessing_kernlength', values=[64, 125, 250]),

    elif config['model'] == 'cnn':
        classifier = Classifier_CNN(input_shape=config['cnn']['input_shape'],
                                    epochs=2, verbose=True, batch_size=64,
                                    use_residual=hp.Choice('use_residual', values=[True, False]),
                                    kernel_size=hp.Choice('kernel_size', values=[40, 32, 64]),
                                    nb_filters=hp.Choice('nb_filters', values=[16, 32, 64]),
                                    depth=hp.Int('depth', min_value=6, max_value=20, step=3),
                                    preprocessing=False
                                    )
    elif config['model'] == 'eegnet':
        classifier = Classifier_EEGNet(dropoutRate = 0.5, 
                                       kernLength = hp.Choice('kernelLength', values=[64, 125, 250]), 
                                       F1 = hp.Choice('F1', values=[16, 32, 64]),
                                       D = hp.Choice('D', values=[2, 4, 8]), 
                                       F2 = hp.Choice('F2', values=[32, 64, 128, 256, 512]), 
                                       norm_rate = 0.5, dropoutType = 'Dropout', epochs = 50)

    elif config['model'] == 'inception':
        classifier = Classifier_INCEPTION(input_shape=config['inception']['input_shape'],
                                          epochs=15, verbose=True, batch_size=64,
                                          use_residual=hp.Choice('use_residual', values=[True, False]),
                                          kernel_size=hp.Choice('kernel_size', values=[40, 32, 64]),
                                          nb_filters=hp.Choice('nb_filters', values=[16, 32, 64]),
                                          depth=hp.Int('depth', min_value=6, max_value=20, step=3),
                                          bottleneck_size=hp.Choice('bottleneck_size', values=[16, 32, 64])
                                          )

    elif config['model'] == 'xception':
        classifier = Classifier_XCEPTION(input_shape=config['inception']['input_shape'],
                                         epochs=2, verbose=True, batch_size=64,
                                         use_residual=hp.Choice('use_residual', values=[True, False]),
                                         kernel_size=hp.Choice('kernel_size', values=[40, 32, 64]),
                                         nb_filters=hp.Choice('nb_filters', values=[16, 32, 64]),
                                         depth=hp.Int('depth', min_value=6, max_value=20, step=3)
                                         )
    elif config['model'] == 'deepeye-rnn':
        classifier = Classifier_DEEPEYE_RNN(input_shape=config['deepeye-rnn']['input_shape'])
    else:
        logging.info('Cannot start the program. Please choose one model in the config.py file')

    return classifier.get_model()


def tune(trainX, trainY):
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=32,
        executions_per_trial=1,
        directory='kerasTunerResults',
        project_name='KerasTuner')

    print(trainX.shape)
    tuner.search_space_summary()
    X_train, X_val, y_train, y_val = train_test_split(trainX, trainY, test_size=0.2, random_state=42)
    tuner.search(X_train, y_train, epochs=15, validation_data=(X_val, y_val), verbose=2)
    tuner.results_summary()
