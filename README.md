## Introduction EEGEyeNet
EEGEyeNet is benchmark to evaluate ET prediction based on EEG measurements with an increasing level of difficulty

## Overview 
The repository consists of general functionality to run the benchmark and custom implementation of different machine learning models. We offer to run standard ML models (e.g. kNN, SVR, etc.) on the benchmark. The implementation can be found in the StandardML_Models directory. 

Additionally, we implemented a range of different deep learning models. These can be run in both pytorch and tensorflow. The architecture of our DL models is such that it is easy to create own custom modules for convolutional neural networks, or even implement your own model on top of our BaseNet implementation. 

## Installation (Environment)
There are many dependencies in this benchmark and we propose to use anaconda as package manager.

### General Requirements 
Create a new conda environment: \
```bash 
conda create -n eegeyenet_benchmark python=3.8.5 
```

First install the general_requirements.txt \
```bash
conda install --file general_requirements.txt 
```
### Pytorch Requirements 
If you want to run the pytorch DL models, first install pytorch in the recommended way. For Linux users with GPU support this is: 
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11 -c pytorch 
```
For other installation types and cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).

### Tensorflow Requirements 
If you want to run the tensorflow DL models, run \
```bash
conda install --file tensorflow_requirements.txt 
```

### Standard ML Requirements 
If you want to run the standard ML models, run 
```bash
conda install --file standard_ml_requirements.txt 
```

This should be installed after installing pytorch to not risk any dependency issues that have to be resolved by conda. 

## Configuration
The model configuration takes place in hyperparameters.py. The training configuration is contained in config.py. 

### config.py
We start by explaining the settings that can be made for running the benchmark: 

config['task']: choose the task to run in the benchmark. \
config['dataset']: choose the dataset used for the task. For some tasks we offer data from multiple paradigms. \
config['preprocessing']: choose between minimal and maximal preprocessed data \
config['feature_extraction']: load Hilbert transformation data. Use this for the standard ML models. \
config['include_ML_models']: run our standard ML models as configured in hyperparameters.py \
config['include_DL_models']: run our deep learning models as configured in hyperparameters.py \
config['include_your_models']: include your own models that youconfigured in hyperparameters.py
config['include_dummy_models']: run dummy models additionally to compare with your results \

You can either choose to train models or use existing ones in /run/ and perform inference with them. Set 
```bash
config['retrain'] = True 
config['save_models'] = True 
```
to train your specified models. Set both to False if you want to load existing models and perform inference. \
In this case specify the path to your existing model directory under 
```bash
config['load_experiment_dir'] = path/to/your/model 
```

In the model configuration section you can specify which framework you want to use. You can run our deep learning models in both pytorch and tensorflow. Just specify it in config.py, make sure you set up the environment as explained above and everything specific to the framework will be handled in the background. 

config.py also allows to configure hyperparameters such as the learning rate, and enable early stopping of models. 

### hyperparameters.py 
Here we define our models. Standard ML models and deep learning models are configured in a dictionary which contains the object of the model and hyperparameters that are passed when the object is instantiated. 

You can add your own models in the your_models dictionary. Specify the models for each task separately. Make sure to enable all the models that you want to run in config.py.

## Running the benchmark 
Create a /run directory to save files while running models on the benchmark. 
### main.py 
To run the benchmark, make sure to uncomment the benchmark() call in main.py and enable loading of data. The IOHelper will load the dataset that was specified via the settings in config.py. To start the benchmark, run
```bash
python3 main.py
```
### benchmark.py 
In benchmark.py we load all models specified in hyperparameters.py and try them with the scoring function corresponding to the task that is benchmarked. 

Models will be tried one after another, save checkpoints to the checkpoint directory specified in config.py. Scores of each model are computed and reported. 

## Add Custom Models 
To benchmark models we use a common interface we call trainer. A trainer is an object that implements the following methods: 
```bash
fit() 
predict() 
save() 
load() 
```
### Implementation of custom models 
To implement your own custom model make sure that you create a class that implements the above methods. If you use library models, make sure to wrap them into a class that implements above interface used in our benchmark. 

### Adding custom models to our benchmark pipeline 
In hyperparameters.py add your custom models into the your_models dictionary. You can add objects that implement the above interface. Make sure to enable your custom models in config.py. 
