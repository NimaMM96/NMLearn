import unittest
import numpy as np
from numpy.testing import assert_array_equal

from functools import partial

from utilities.utils import histogram, calc_grad

#####################
# Utility functions #
#####################

def mse(X, Y):
    return np.power(Y-X,2)

class TestBase(unittest.TestCase):

    def test_histogram(self):

        # test cases
        X_1 = np.array([0,0,0,1,1,1,1,0])
        Y_1 = np.array([0.5, 0.5])

        X_2 = np.array([0,0,0,0,1,1,1,1,2,2])
        Y_2 = np.array([0.4,0.4,0.2])

        # assertions
        actual = histogram(X_1, np.unique(X_1).shape[0])
        self.assertIsInstance(actual,np.ndarray)
        assert_array_equal(actual, Y_1)

        actual = histogram(X_2, np.unique(X_2).shape[0])
        self.assertIsInstance(actual,np.ndarray)
        assert_array_equal(actual, Y_2)

    def test_calc_grad(self):
        # test data
        X_test = np.array([0.6, 0.1, 0.0])
        Y_test = np.array([0.8, 0.3, 0.1])
        expected_grad = np.array([-0.4, -0.4, -0.2])
        
        partial_mse = partial(mse, Y=Y_test)
        actual_grad = calc_grad(partial_mse, X_test, delta=0.001)
        for i in range(expected_grad.shape[0]):
            assert abs(expected_grad[i] - actual_grad[i]) <= 1e-3, "Error is to large, expected gradient {:.3f} vs actual gradient {:.3f}".format(expected_grad[i], actual_grad[i])
        


# run tests
if __name__ == "__main__":
    unittest.main()
