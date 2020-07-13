import numpy as np

from typing import List, Tuple, Dict, Union
from functools import partial

from utilities.utils import calc_grad


######################
# utitlity functions #
######################

# TODO: will support other loss functions soon
def mse(Y_pred: np.ndarray, Y: np.ndarray) -> float:
    return np.power(Y - Y_pred, 2)    

#######################
# grad boosting class #
#######################

class grad_boosting:

    def __init__(self,base_learner,T,**kwargs):

        self.model_params = kwargs # dictionary of parameters for the base learner
        self.base_learner = base_learner # base learner class
        self.__T = T # number of base learners
        self.list_of_models = [] # storage of the trained classifiers

    def fit(self, X, Y):
        # initialisation (create first base learner)
        model = self.base_learner(**self.model_params.copy())
        model.fit(X, Y)
        self.list_of_models.append(model)
        for i in range(self.__T-1):
            # calc gradient
            Y_pred = self.predict(X)
            partial_mse = partial(mse, Y=Y)
            grad = calc_grad(partial_mse, Y_pred.squeeze(), delta=0.001)

            # fit base learner to ensemble error gradient
            model = self.base_learner(**self.model_params.copy())
            model.fit(X, grad)
            self.list_of_models.append(model)

    def predict(self, X):
        for i, model in enumerate(self.list_of_models):
            if i == 0:
                resp = model.predict(X)
            else:
                resp -= model.predict(X)
        return resp/len(self.list_of_models)
            
            
        
        

