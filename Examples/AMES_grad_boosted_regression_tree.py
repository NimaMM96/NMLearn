import pandas as pd
import numpy as np
import os

from NMLearn.classifiers.tree.desicion_tree import regression_tree
from NMLearn.ensemble.grad_boosting import grad_boosting
from NMLearn.utilities.metrics import rmse

#####################
# Utility functions #
#####################

def pre_processing(df):
    df = pd.get_dummies(df, dummy_na=True) # for columns with categorical variables, NaN values encoded too.
    df = df.apply(lambda x: x.fillna(x.mean()), axis=0) # for remaining columns NaN values are filled with mean of that column
    return df

##########
# config #
##########
# data parameters
DATA_PATH = "<Path to Dataset>"

# ensemble parameters
NUMBER_OF_TREES = 25
BASE_LEARNER_PARAMS = {"max_depth": 7, "no_trials": 32, "training_alogrithim": "randomize", "obj_func": "mse"}

################
# load in data #
################

path_to_data_train = os.path.join(DATA_PATH, 'train.csv')
path_to_data_test = os.path.join(DATA_PATH, 'test.csv')
df_train = pd.read_csv(path_to_data_train)
df_test = pd.read_csv(path_to_data_test)

##################
# pre-processing #
##################

# feature engineering
df_train = pre_processing(df_train)
df_test = pre_processing(df_test)

# parse dependant variable from data and convert to numpy array type
Y_train, X_train = df_train["SalePrice"].to_numpy(), df_train[[col for col in df_train.columns if col != "SalePrice"]].to_numpy()
Y_test, X_test = df_test["SalePrice"].to_numpy(), df_test[[col for col in df_test.columns if col != "SalePrice"]].to_numpy()

# remove variables with no variance
std = np.std(X_train, axis=0)
X_train = X_train[:, std>1]

#############
# Grow Tree #
#############

model = grad_boosting(regression_tree, NUMBER_OF_TREES, **BASE_LEARNER_PARAMS)
model.fit(X_train, Y_train)
Y_train_prob = model.predict(X_train).squeeze()
train_score = rmse(Y_train_prob, Y_train, apply_log=True)

#########################
# Evaluate on test data #
#########################
Y_test_prob = model.predict(X_test).squeeze()
test_score = rmse(Y_test_prob, Y_test, apply_log=True)

print("Test Performance: {:.3f}".format(test_score))
print("Train Performance: {:.3f}".format(train_score))      
