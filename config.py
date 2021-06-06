# configuration used by the training and evaluation methods
# let's keep it here to have a clean code on other methods that we try
import time
import os
config = dict()

##################################################################
##################################################################
############### BENCHMARK CONFIGURATIONS #########################
##################################################################
##################################################################
# 'LR_task' (dataset: 'antisaccade'):
# 'Direction_task' (dataset: 'dots' or 'processing_speed'):
# 'Position_task' (dataset: 'dots'):
config['task'] = 'Position_task'
config['dataset'] = 'dots'
config['preprocessing'] = 'max'  # or min
config['feature_extraction'] = False 

##################################################################
##################################################################
############### PATH CONFIGURATIONS ##############################
##################################################################
##################################################################
# Where experiment results are stored.
config['log_dir'] = './runs/'
# Path to training, validation and test data folders.
config['data_dir'] = './data/'
# Path of root
config['root_dir'] = '.'
# Retrain or load already trained
config['retrain'] = True
config['save_models'] = True
# If retrain is false we need to provide where to load the experiment files
config['load_experiment_dir'] = 'Amplitude_max/'
# all_EEG_file should specify the name of the file where the prepared data is located (if emp
def build_file_name():
    all_EEG_file = config['task'] + '_with_' + config['dataset']
    all_EEG_file = all_EEG_file + '_' + 'synchronised_' + config['preprocessing']
    all_EEG_file = all_EEG_file + ('_hilbert.npz' if config['feature_extraction'] else '.npz')
    return all_EEG_file
config['all_EEG_file'] = build_file_name() # or use your own specified file name


##################################################################
##################################################################
############### MODELS CONFIGURATIONS ############################
##################################################################
##################################################################
# Specific to models now
config['framework'] = 'pytorch'
config['learning_rate'] = 1e-4
config['early_stopping'] = True
config['patience'] = 10


##################################################################
##################################################################
############### HELPER VARIABLES #################################
##################################################################
##################################################################
config['trainX_variable'] = 'EEG'
config['trainY_variable'] = 'labels'


def create_folder():
    if config['retrain']:
        model_folder_name = str(int(time.time()))
        config['model_dir'] = os.path.abspath(os.path.join(config['log_dir'], model_folder_name))
        config['checkpoint_dir'] = config['model_dir'] + '/checkpoint/'
        if not os.path.exists(config['model_dir']):
            os.makedirs(config['model_dir'])

        if not os.path.exists(config['checkpoint_dir']):
            os.makedirs(config['checkpoint_dir'])

        config['info_log'] = config['model_dir'] + '/' + 'info.log'
        config['batches_log'] = config['model_dir'] + '/' + 'batches.log'

    else:
        config['model_dir'] = config['log_dir'] + config['load_experiment_dir']
        config['checkpoint_dir'] = config['model_dir'] + 'checkpoint/'
        stamp = str(int(time.time()))
        config['info_log'] = config['model_dir'] + '/' + 'inference_info_' + stamp + '.log'
        config['batches_log'] = config['model_dir'] + '/' + 'inference_batches_' + stamp + '.log'



