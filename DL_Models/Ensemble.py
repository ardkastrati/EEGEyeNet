"""
Common interface to fit and predict with torch and tf ensembles 
"""
from config import config
import numpy as np
class Ensemble:

    def __init__(self, model_name, nb_models, loss, batch_size=64, **model_params):
        """

        :type model_params: dic
        """
        self.type = 'classifier' if loss == 'bce' else 'regressor'
        # create the ensemble 
        if config['framework'] == 'tensorflow':
            from DL_Models.tf_models.Ensemble_tf import Ensemble_tf
            self.ensemble = Ensemble_tf(model_name=model_name, nb_models=nb_models, loss=loss, batch_size=batch_size, **model_params)
        elif config['framework'] == 'pytorch':
            from DL_Models.torch_models.Ensemble_torch import Ensemble_torch
            self.ensemble = Ensemble_torch(model_name=model_name, nb_models=nb_models, loss=loss, batch_size=batch_size, **model_params)
        else:
            raise ValueError("Choose a valid deep learning framework")

    def fit(self, trainX, trainY, validX, validY):
        self.ensemble.fit(trainX, trainY, validX, validY)
        
    def predict(self, testX):
        if self.type == 'classifier':
            return np.round(self.ensemble.predict(testX)) #TODO: Hack
        else:
            return self.ensemble.predict(testX)