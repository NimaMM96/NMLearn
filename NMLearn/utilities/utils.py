import numpy as np

from typing import List, Tuple, Dict, Union, Callable

def histogram(X_: np.ndarray, no_cats: int) -> Union[np.ndarray, None]:
    """
    calculate histogram from a set of samples, at the moment
    only supports categorical rv's
    """
    if X_.shape[0] == 0:
        return None
        assert (X_ == 0).sum()==0, "class labels should start from 0"
    X = X_.copy().astype(np.int)
    Y = np.zeros(no_cats)
    for x in X:
        Y[x] += 1
    return Y/X.shape[0]

def calc_grad(func: Callable, X: np.ndarray, delta: float) -> np.ndarray:
    """
    Currently only suppourt scalar-scalar functions
    """
    return (func(X+delta)-func(X))/delta
