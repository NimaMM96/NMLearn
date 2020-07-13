import numpy as np

from typing import List, Tuple, Dict, Union
import warnings

from utilities.utils import histogram


#####################
# utility functions #
#####################

def split(X: np.ndarray, feat_index: int, th: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    split applied at each node, currently only support single feature splitting
    """
    cond = X[:,feat_index]<=th
    return X[cond], X[~cond], cond    

def gini_increment(score: float, p: float, N: int, inc: int) -> float:
    """
    incrmenent the gini metric in a given node, used in the CART algorithim

    params: score, current gini score in a given node
    params: p, (unormalised) probability which has changed and effects score
    params: N, No. of samples in the node
    """
    score -= ((p-inc)/(N-inc))**2
    score *= ((N-inc)/N)**2
    score += (p/N)**2
    return score


##################
# Splitter Class #
##################

class Splitter:
    list_of_algos = ["randomize", "cart"]
    list_of_fcns = ["ig", "gini", "mse"]
    C = None # this attribute represents the number of classes for a classification problem and for regression tasks, represents the number of dependant variables
    def __init__(self, algo: str, obj_fcn: str, p: int):
        """
        algo: algorithim to find split for node, currently suppourt: random trials and CART
        obj_fcn: criteria to use for calculating distribution impurity currently suppourt (entropy, gini and mse (regression))
        num_feats: for random trials algorithim this is the number of trials, for CART this is num of features to use in search for best split
        """
        # public attributes
        self.algo = algo
        self.obj_fcn = obj_fcn
        self.p = p

        # private attributes
        self.__is_regression = obj_fcn == "mse"

        # checks
        if self.algo not in self.list_of_algos:
            raise ValueError("The splitting algorithim {} is not suppourted, please choose from {}".format(self.algo, self.list_of_algos))

        if self.obj_fcn not in self.list_of_fcns:
            raise ValueError("The objective function {} is not suppourted, please choose from {}".format(self.obj_fcn, self.list_of_fcns))

    @staticmethod
    def entropy(P: np.ndarray) -> float:
        with warnings.catch_warnings(): # ignore divide by zero warnings in log
            warnings.simplefilter("ignore")
            LogP = np.log2(P)
        LogP[~np.isfinite(LogP)] = 0
        return (P*LogP).sum()

    @staticmethod
    def gini(P: np.ndarray) -> float:
        return np.power(P,2).sum()

    @staticmethod
    def mse(P: np.ndarray) -> float:
        return -np.sum(np.power(P-np.mean(P),2))


    def metric(self, Y_left: np.ndarray, Y_right: np.ndarray) -> float:
        """
        function which executes the impuritiy metric specified, only supourt IG and gini
        """
        obj_fcn = self.obj_fcn # don't change value of class attribute
        if obj_fcn == "ig":
            obj_fcn = "entropy" # name of function to execute for information gain calculation        
        fcn = getattr(Splitter, obj_fcn)
        if self.__is_regression:
            return fcn(Y_left), fcn(Y_right)
        else:
            P_left = histogram(Y_left, self.C)
            P_right = histogram(Y_right, self.C)
            return Y_left.shape[0]*fcn(P_left), Y_right.shape[0]*fcn(P_right)


    def get_split_random(self, X: np.ndarray) -> Tuple[Tuple[float,int],Tuple[np.ndarray,np.ndarray]]:
        """
        function to return best split on current node, from a total of p trials.
        This strategy introduces more randomness into the classifer, which could poentially
        be benifical in an ensemble (reduce error correlation between base classifiers).
        """

        best_score = -np.inf
        max_th_per_feat = np.max(X, axis=0)
        min_th_per_feat = np.min(X, axis=0)
        for i in range(self.p):
            feat_index = np.random.randint(X.shape[1]-1) # final dimension is labels
            max_, min_ = max_th_per_feat[feat_index], min_th_per_feat[feat_index]
            if (max_- min_)!=0:
                th = np.random.rand()*(max_- min_) + min_
            else:
                continue # skip this iteration as cannot split on this feature
            X_left, X_right, _ = split(X, feat_index, th)
            if self.__is_regression:
                score_left, score_right = self.metric(X_left[:,-self.C:], X_right[:,-self.C:])
            else:
                score_left, score_right = self.metric(X_left[:,-1], X_right[:,-1])
            score = score_left + score_right
            if score > best_score:
                best_score = score
                best_th = th
                best_feat_index = feat_index
                X_left_ = X_left
                X_right_ = X_right
        if best_score == -np.inf:
            return -1, -1
        else:
            return (best_th, best_feat_index), (X_left_, X_right_)


    def get_split_CART(self, X: np.ndarray) -> Tuple[Tuple[float,np.ndarray],Tuple[int,np.ndarray]]:
        """
        this function implements the CART algorithim, no randomness in generating tree using this training algorithim
        but would likely improve the performance of the individual tree, but in an ensemble may lead to large error correlation
        which could significantly reduce the ensemble gain.
        """

        # calculate variance of features, use variance for selecting most important
        feat_var = np.std(X[:,:-1], axis=0)
        feat_to_search = np.argsort(feat_var)[-self.p:]
        I = np.argsort(X[:,:-1], axis=0)
        for i, feat_index in enumerate(feat_to_search): # last column is the class labels
            x = X[I[:, feat_index]].copy()
            x = x[:, [feat_index, X.shape[1]-1]]
            # calculte initial histograms (unormalised) and impurity metrics
            x_left, x_right = x[:1], x[1:]
            M_left = histogram(x_left[:,-1], self.C) * x_left.shape[0]
            M_right = histogram(x_right[:,-1], self.C) * x_right.shape[0]
            score_left, score_right = self.metric(x_left[:,-1], x_right[:,-1])
            score = score_left + score_right
            if i == 0:
                best_score = score
                best_th = x[0,0]
                best_feat_index = feat_index
                
            # unormalise score makes calculation simpler
            score_left /= x_left.shape[0]
            score_right /= x_right.shape[0]

            # loop over all splits on this feature
            N = x_right.shape[0]
            for j in range(1,N-1):
                x_left, x_right = x[:j+1], x[j+1:]
                new_class_to_left = x[j,-1]
                M_left[new_class_to_left] += 1
                M_right[new_class_to_left] -= 1

                # modifiy scores
                score_left = gini_increment(score_left, M_left[new_class_to_left], x_left.shape[0], 1)
                score_right = gini_increment(score_right, M_right[new_class_to_left], x_right.shape[0], -1)

                score = score_left*x_left.shape[0] + score_right*x_right.shape[0]

                if score > best_score:
                    best_score = score
                    best_th = x[j,0]
                    best_feat_index = feat_index

        # get best split
        X_left, X_right, _ = split(X, best_feat_index, best_th)
        return (best_th, best_feat_index), (X_left, X_right)


    def get_best_split(self, X: np.ndarray):
        if self.algo == "cart":
            return self.get_split_CART(X)
        else:
            return self.get_split_random(X)
        
