import numpy as np

from typing import List, Tuple, Dict, Union
from graphviz import Digraph

from utilities.utils import histogram
from classifiers.tree.splitter import Splitter, split

###########################################
# utility functions for generating graphs #
###########################################

def label_nodes(node, f):
    if node["leaf_node"]:
        f.node(get_node_label(node))
    else:
        f.node(get_node_label(node))
        label_nodes(node["left_node"],f)
        label_nodes(node["right_node"],f)

def get_node_label(node):
    if node["leaf_node"]:
        if "value" in node:
            label =  "Leaf Node {}\nMean Value {}".format(node["leaf_number"], node["value"])
        else:
            label =  "Leaf Node {}\nMajority Class {}".format(node["leaf_number"], np.argmax(node["histogram"]))
    else:
        label = "Node\nfeature {} <= {}".format(node["feature_index"], node["threshold"])
    return label

def label_edges(node, f):
    if not node["leaf_node"]:
        f.edge(get_node_label(node), get_node_label(node["left_node"]))
        f.edge(get_node_label(node), get_node_label(node["right_node"]))
        label_edges(node["left_node"], f)
        label_edges(node["right_node"], f)


##############
# Node class #
##############

# override repr method in dict class, to avoid showing the nested dictionary objects
class Node(dict): 
    def __repr__(self):
        if self.__getitem__('leaf_node'):
            return f"Leaf{self.__class__.__name__}"
        else:
            return f"{self.__class__.__name__}(threshold: {self.__getitem__('threshold')}, feature_index: {self.__getitem__('feature_index')})"

######################
# base decision tree #
######################

class base_decision_tree:
    C = None # this attribute represents the number of classes for a classification problem and for regression tasks, represents the number of dependant variables
    def __init__(self, max_depth, p=20, training_alogrithim="CART", obj_func='IG', growth_rate=0.05, **kwargs):

        # check if required parameter is passed, if not check kwargs
        if max_depth is None:
            max_depth = kwargs["max_depth"]
            p = kwargs.get("p", 20)
            growth_rate = kwargs.get("growth_rate", 0.05)
            training_alogrithim = kwargs.get("training_alogrithim", "CART")
            obj_func = kwargs.get("obj_func", "IG")


        # splitter object (TODO: implement in Cython)
        self.splitter_obj = Splitter(training_alogrithim.lower(), obj_func.lower(), p)

        # private parameters
        self.__depth = max_depth
        self.__no_leaf_nodes = 0
        self.__is_regression = obj_func.lower() == "mse"
        self.__growth_rate = growth_rate


    def traverse_tree(self, X: np.ndarray, I: np.ndarray, node: Node) -> np.ndarray:
        if not node['leaf_node']:
            X_left, X_right, cond = split(X, node['feature_index'], node['threshold'])
            I_left, I_right = I[cond], I[~cond]
            if len(X_right) == 0:
                return self.traverse_tree(X_left, I_left, node['left_node'])
            elif len(X_left) == 0:
                return self.traverse_tree(X_right, I_right, node['right_node'])
            else:
                return np.concatenate((self.traverse_tree(X_left, I_left, node['left_node']), self.traverse_tree(X_right, I_right, node['right_node'])))
        else:
            if len(X)!=0:
                if self.__is_regression:
                    return np.concatenate((np.tile(np.array([node['value']])[np.newaxis,:], reps=(X.shape[0],1)),I[:, np.newaxis]), axis=1)
                else:
                    return np.concatenate((np.tile(node['histogram'][np.newaxis,:], reps=(X.shape[0],1)),I[:, np.newaxis]), axis=1)    

    def leaf_node(self, X: np.ndarray) -> Node:
        self.__no_leaf_nodes += 1
        if self.__is_regression:
            return Node(leaf_node=True, value=np.mean(X[:,-self.C:]), leaf_number=self.__no_leaf_nodes)
        else:
            return Node(leaf_node=True, histogram=histogram(X[:,-1], self.C), leaf_number=self.__no_leaf_nodes)

    def train_node(self, X: np.array, current_depth: int) -> Node:
        if np.unique(X[:,-1]).shape[0] == 1: # if split pure, node should be leaf node
            return self.leaf_node(X)
        node =  Node(leaf_node=False)
        params, data = self.splitter_obj.get_best_split(X)
        if params == -1:
            return self.leaf_node(X)
        node['threshold'], node['feature_index'] = params
        X_left, X_right = data
        
        if (current_depth < self.__depth):
            node['left_node'] = self.train_node(X_left, current_depth+1) if X_left.shape[0] > 1 else self.leaf_node(X_left)
            node['right_node'] = self.train_node(X_right, current_depth+1) if X_right.shape[0] > 1 else self.leaf_node(X_right)
        elif (X_left.shape[0] > 0) and (X_right.shape[0] > 0):
            node['left_node'] = self.leaf_node(X_left)
            node['right_node'] = self.leaf_node(X_right)
        else:
            node = self.leaf_node(X)
        return node

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        Y = Y[:,np.newaxis] if len(Y.shape) == 1 else Y
        X = np.concatenate((X, Y), axis=1)
        self.C = Y.shape[1] if self.__is_regression else np.unique(Y).shape[0]
        self.splitter_obj.C = self.C
        self.root = self.train_node(X,1)

    def predict(self, X: np.ndarray) -> Dict:
        probs = self.traverse_tree(X, np.arange(X.shape[0]), self.root)
        probs = probs[np.argsort(probs[:,-1])][:,:-1]
        return probs

    def create_graph(self, path_to_write):
        f = Digraph('decision_tree_graph')
        f.attr(rankdir='TB')
        f.attr("node", style="filled", color="lightgrey")

        label_nodes(self.root, f)
        label_edges(self.root, f)

        f.render('dot', 'pdf')



###################
# regression tree #
###################

class regression_tree(base_decision_tree):
    def __init__(self, max_depth, p=20, training_alogrithim="Randomize", obj_func='mse', **kwargs):
        # assertions
        assert training_alogrithim.lower() == "randomize", "For regression only the random selection algorithim is suppourted currently"
        assert obj_func.lower() == "mse", "For regression only the MSE criterion is suppourted, not {}".format(obj_func)
        if max_depth is None: # when used in ensemble
            assert kwargs["obj_func"].lower() == "mse", "For regression only the MSE criterion is suppourted, not {}".format(obj_func)
        super(regression_tree, self).__init__(max_depth, p, training_alogrithim, obj_func, **kwargs)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        super(regression_tree, self).fit(X, Y)

    def predict(self, X: np.ndarray) -> Dict:
        return super(regression_tree, self).predict(X)

#######################
# classification tree #
#######################
        
class classification_tree(base_decision_tree):
    def __init__(self, max_depth, p=20, training_alogrithim="CART", obj_func='gini', growth_rate=0.05, **kwargs):
        # assertions
        if obj_func.lower() == "ig":
            assert training_alogrithim.lower() == "randomize", "For the Information Gain criterion, only the random selection training algorithim suppourted currently"
        super(classification_tree, self).__init__(max_depth, p, training_alogrithim, obj_func, growth_rate, **kwargs)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        super(classification_tree, self).fit(X, Y)

    def predict(self, X: np.ndarray) -> Dict:
        return super(classification_tree, self).predict(X)
