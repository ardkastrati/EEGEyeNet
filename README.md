# EEGEyeNet
EEGEyeNet is benchmark to evaluate ET prediction based on EEG measurements with an increasing level of difficulty

# Create environment 
There are many dependencies in this benchmark and we propose to use anaconda as package manager.

conda create -n eegeyenet_benchmark python=3.8.5

First install the general_requirements.txt 
pip install -r general_requirements.txt

If you want to run the pytorch DL models, run
pip install -r pytorch_requirements.txt 

If you want to run the tensorflow DL models, run
pip install -r tensorflow_requirements.txt 