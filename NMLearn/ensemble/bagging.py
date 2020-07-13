import numpy as np
from numpy.random import permutation

from typing import List, Tuple, Dict, Union

class bagging_ensemble:

    def __init__(self,base_learner,T,percent_data=0.6,percent_feat=1,**kwargs):

        self.model_params = kwargs # dictionary of parameters for the base learner
        self.base_learner = base_learner # base learner class
        self.__T = T # number of base learners
        self.__percent_data = percent_data # percetnage of data to train each learner
        self.__percent_feat = percent_feat # percentage of features to use in each base learner
        self.list_of_models = [] # storage of the trained classifiers

    def fit(self, X, Y):
    
        # train each base-learner
        N = np.floor(self.__percent_data*X.shape[0]).astype(np.int)
        M = np.floor(self.__percent_feat*X.shape[1]).astype(np.int)
        for i in range(self.__T):
            idx = permutation(X.shape[0])[:N]
            X_train = X[idx]
            if M < X.shape[1]:
                X_train = X[:, permutation(X.shape[1])[:M]] # percentage of data/features to keep
            Y_train = Y[idx]
            model = self.base_learner(**self.model_params.copy())
            model.fit(X_train, Y_train)
            self.list_of_models.append(model)

    def predict(self, X):
        for i, model in enumerate(self.list_of_models):
            if i == 0:
                resp = model.predict(X)
            else:
                resp += model.predict(X)
        return resp/len(self.list_of_models)
            
            
        
        

