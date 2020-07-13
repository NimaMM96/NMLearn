import os

from NMLearn.classifiers.tree.desicion_tree import classification_tree
from NMLearn.ensemble.bagging import bagging_ensemble
from NMLearn.utilities.dataset_utils.mnist import load_mnist_data
from NMLearn.utilities.metrics import accuracy

##########
# config #
##########
# data parameters
DATA_PATH = "<Path to Dataset>"

# ensemble parameters
NUMBER_OF_TREES = 10
PERCENT_DATA = 0.6
PERCENT_FEAT = 1

# base learner parameters dictionary
BASE_LEARNER_PARAMS = {"max_depth": 7, "no_trials": 20, "training_alogrithim": "randomize", "obj_func": "gini"}

################
# Load in Data #
################

# load in training data
X_train = load_mnist_data(os.path.join(DATA_PATH, 'train-images-idx3-ubyte.gz'))
Y_train = load_mnist_data(os.path.join(DATA_PATH, 'train-labels-idx1-ubyte.gz'))

# load in test data
X_test = load_mnist_data(os.path.join(DATA_PATH, 't10k-images-idx3-ubyte.gz'))
Y_test = load_mnist_data(os.path.join(DATA_PATH, 't10k-labels-idx1-ubyte.gz'))


#############
# Grow Tree #
#############

model = bagging_ensemble(classification_tree, NUMBER_OF_TREES, PERCENT_DATA, PERCENT_FEAT, **BASE_LEARNER_PARAMS)
model.fit(X_train, Y_train)

Y_train_prob = model.predict(X_train)
train_acc = accuracy(Y_train_prob, Y_train)


#########################
# Evaluate on test data #
#########################

Y_test_prob = model.predict(X_test)
test_acc = accuracy(Y_test_prob, Y_test)

print("Test Performance: {:.3f}".format(test_acc))
print("Train Performance: {:.3f}".format(train_acc))      
