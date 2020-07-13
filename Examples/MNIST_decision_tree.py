import os
import time

from NMLearn.classifiers.tree.desicion_tree import classification_tree
from NMLearn.utilities.dataset_utils.mnist import load_mnist_data
from NMLearn.utilities.metrics import accuracy

##########
# config #
##########
# data parameters
DATA_PATH = "<Path to Dataset>"

# model parameters
MAX_FEATURES = 32
MAX_DEPTH = 7
OBJECTIVE_FCN = "gini"
TRAINING_ALGO = "CART"

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

model = classification_tree(MAX_DEPTH, to_features_to_check=MAX_FEATURES, training_alogrithim=TRAINING_ALGO, obj_func=OBJECTIVE_FCN)
start = time.time()
model.fit(X_train, Y_train)
duration = time.time()-start
Y_train_prob = model.predict(X_train)
train_acc = accuracy(Y_train_prob, Y_train)

#########################
# Evaluate on test data #
#########################

Y_test_prob = model.predict(X_test)
test_acc = accuracy(Y_test_prob, Y_test)

print("Test Performance: {:.3f}".format(test_acc))
print("Train Performance: {:.3f}".format(train_acc))      
