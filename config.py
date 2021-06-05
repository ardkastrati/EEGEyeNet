# configuration used by the training and evaluation methods
# let's keep it here to have a clean code on other methods that we try
import time
import os
config = dict()

##################################################################
# Please note that the following fields will be set by our scripts to re-train and re-evaluate our model.
# Where experiment results are stored.
config['log_dir'] = './runs/'
# Path to training, validation and test data folders.
# PS: Note that we have to upload the data to the server!!!
# config['data_dir'] = '/cluster/home/your_username/data/'
# config['data_dir'] = './data/'
config['data_dir'] = '../data/'
# Path of root
config['root_dir'] = '.'
config['framework'] = 'pytorch'
config['learning_rate'] = 1e-4
config['early_stopping'] = True
config['patience'] = 10
config['save_models'] = True

##################################################################
# You can modify the rest or add new fields as you need.
# Hyper-parameters and training configuration.
#config['learning_rate'] = 1e-4

#"""
#Parameters that can be chosen:

#cnn: The simple CNN architecture
#inception: The InceptionTime architecture
#eegnet: The EEGNet architecture
#deepeye: The Deep Eye architecture
#xception: The Xception architecture
#deepeye-rnn: The DeepEye RNN architecture

#If downsampled set to true, the downsampled data (which have length 125) will be used. Otherwise, full data with length 500 is used.
#If split set to true, the data will be clustered and fed each to a separate architecture. The extracted features are concatenated
#finally used for classification.
#Cluster can be set to clustering(), clustering2() or clustering3(), where different clusters based on literature are used.
#"""

# Choosing model
config['model'] = ''
#config['downsampled'] = False

#config['ensemble'] = 1 #number of models in the ensemble method

#config['trainX_file'] = 'noweEEG.mat' if config['downsampled'] else 'all_EEGprocuesan.mat'
#config['trainY_file'] = 'all_trialinfoprosan.mat'
#config['trainX_variable'] = 'noweEEG' if config['downsampled'] else 'all_EEGprocuesan'
#config['trainY_variable'] = 'all_trialinfoprosan'


# 'LR_task' (dataset: 'antisaccade'):
# 'Direction_task' (dataset: 'dots' or 'processing_speed'):
# 'Position_task' (dataset: 'dots'):
config['task'] = 'LR_task'
config['dataset'] = 'antisaccade'
config['preprocessing'] = 'max'  # or min
config['feature_extraction'] = True

# format for .npz data
config['preprocessing_path'] = 'synchronised_' + config['preprocessing']
config['all_EEG_file'] = config['task'] + '_with_' + config['dataset']
config['all_EEG_file'] = config['all_EEG_file'] + '_' + config['preprocessing_path']
config['all_EEG_file'] = config['all_EEG_file'] + ('_hilbert.npz' if config['feature_extraction'] else '.npz')
config['trainX_variable'] = 'EEG'
config['trainY_variable'] = 'labels'

# OLD CROSS VALIDATION IMPLMENTATION
# MAKE SURE TO HAVE PARAMETRS COHERENT WITH DATASET
# c for classification, r for regression
config['ml'] = 'c'
# label to predict for ml
# lr : 1 for label
# dir : 1 for amplitude, 2 for angle
# pos : 1 for pos_x, 2 for pos_y
config['label'] = 1

# CNN
config['cnn'] = {}
# CNN
config['pyramidal_cnn'] = {}
# InceptionTime
config['inception'] = {}
# DeepEye
config['deepeye'] = {}
# Xception
config['xception'] = {}
# EEGNet
config['eegnet'] = {}
# DeepEye-RNN
config['deepeye-rnn'] = {}

config['cnn']['input_shape'] = (500, 129)
config['pyramidal_cnn']['input_shape'] = (500, 129)
config['inception']['input_shape'] =  (500, 129)
config['deepeye']['input_shape'] = (500, 129)
config['deepeye-rnn']['input_shape'] = (500, 129)
config['xception']['input_shape'] = (500, 129)

config['eegnet']['channels'] = 129
config['eegnet']['samples'] = 500


timestamp = str(int(time.time()))
model_folder_name = None
model_folder_name = 'timestamp' if config['model'] == '' else timestamp + "_" + config['model']

#if config['ensemble']>1:
#    model_folder_name += '_ensemble'

config['model_dir'] = os.path.abspath(os.path.join(config['log_dir'], model_folder_name))
config['checkpoint_dir'] = config['model_dir'] + '/checkpoint/'
if not os.path.exists(config['model_dir']):
    os.makedirs(config['model_dir'])

if not os.path.exists(config['checkpoint_dir']):
    os.makedirs(config['checkpoint_dir'])

config['info_log'] = config['model_dir'] + '/' + 'info.log'
config['batches_log'] = config['model_dir'] + '/' + 'batches.log'
