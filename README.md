# Introduction EEGEyeNet
EEGEyeNet is benchmark to evaluate ET prediction based on EEG measurements with an increasing level of difficulty

# Overview 
The repository consists of general functionality to run the benchmark and custom implementation of different machine learning models. We offer to run standard ML models (e.g. kNN, SVR, etc.) on the benchmark. The implementation can be found in the StandardML_Models directory. 

Additionally, we implemented a range of different deep learning models. These can be run in both pytorch and tensorflow. The architecture of our DL models is such that it is easy to create own custom modules for convolutional neural networks, or even implement your own model on top of our BaseNet implementation. 

# Installation (Environment)
There are many dependencies in this benchmark and we propose to use anaconda as package manager.

## General Requirements 
Create a new conda environment: \
conda create -n eegeyenet_benchmark python=3.8.5 

First install the general_requirements.txt \
conda install --file general_requirements.txt 

## Pytorch Requirements 
If you want to run the pytorch DL models, first install pytorch in the recommended way. For Linux users with GPU support this is: \
conda install pytorch torchvision torchaudio cudatoolkit=11 -c pytorch \
For other installation types and cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).

## Tensorflow Requirements 
If you want to run the tensorflow DL models, run \
conda install --file tensorflow_requirements.txt 

## Standard ML Requirements 
If you want to run the standard ML models, run \
conda install --file standard_ml_requirements.txt \
This should be installed after installing pytorch to not risk any dependency issues that have to be resolved by conda. 

# Configuration
The model configuration takes place in hyperparameters.py. The training configuration is contained in config.py. 

## config.py
We start by explaining the settings that can be made for running the benchmark: \

config['task']: choose the task that should be run in the benchmark. \
config['dataset']: choose the dataset used for the task. For some tasks we offer multiple datasets/paradigms. \
config['preprocessing']: choose between minimal and maximal preprocessed data \
config['feature_extraction']: set to True for applying Hilbert transformation on the data. Use this for the standard ML models. \
config['include_ML_models']: set to True if you want to run our standard ML models as configured in hyperparameters.py \
config['include_DL_models']: set to True if you want to run our deep learning models as configured in hyperparameters.py \
config['include_your_models']: set to True if you want to include your own models as configured in hyperparameters.py
config['include_dummy_models']: set to True if you want to run dummy models additionally to compare with your results \

You can either choose to train models or use existing ones in /run/ and perform inference with them. Set \
config['retrain'] = True \
config['save_models'] = True \
if you want to train your specified models. Set both to False if you want to load existing models and perform inference. \
In this case specify the path to your existing model directory under \
config['load_experiment_dir'] = path/to/your/model 

In the model configuration section you can specify which framework you want to use. You can run our deep learning models in both pytorch and tensorflow. Just specify it in config.py, make sure you set up the environment as explained above and everything specific to the framework will be handled in the background. 

config.py also allows to configure hyperparameters such as the learning rate, and enable early stopping of models. 

## hyperparameters.py 
Here we define our models. Standard ML models and deep learning models are configured in a dictionary which contains the object of the model and hyperparameters that are passed when the object is instantiated. 

You can add your own models in the your_models dictionary. Specify the models for each task separately. Make sure to enable all the models that you want to run in config.py.

# Running the benchmark 
Create a /run directory to save files while running models on the benchmark. 
## main.py 
To run the benchmark, make sure to uncomment the benchmark() call in main.py 


# Add Custom Models 
## Implementation of custom models 


## Adding custom models to our benchmark pipeline 

