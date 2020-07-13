import numpy as np

def accuracy(Y_prob: np.ndarray, Y_true: np.ndarray) -> float:

    # assertions, only suppourt distribution as input, not predicted class labels
    no_classes = np.unique(Y_true).shape[0]
    assert Y_prob.shape[1] == no_classes, "predictions must be a distribution"
    assert round(Y_prob.sum(), 2) == Y_prob.shape[0], "predictions must be a distribution"

    Y_pred_class = np.argmax(Y_prob, axis=1)
    return np.equal(Y_pred_class, Y_true).mean()

def rmse(Y_pred: np.ndarray, Y_true: np.ndarray, take_root: bool=True, apply_log: bool=False, eps: float=0.0) -> float:
    if apply_log:
        residuals = np.log(Y_pred+eps) - np.log(Y_true+eps)
    else:
        residuals = Y_pred - Y_true
    return np.sqrt(np.mean(np.power(residuals,2))) if take_root else np.mean(np.power(residuals,2))
