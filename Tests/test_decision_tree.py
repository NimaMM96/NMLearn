import unittest
import numpy as np
from numpy.testing import assert_array_equal

from classifiers.tree.desicion_tree import base_decision_tree as decision_tree
from classifiers.tree.desicion_tree import Node
from classifiers.tree.splitter import Splitter, split, gini_increment
from utilities.utils import histogram

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

    def test_split(self):

        # test case
        X_1 = np.array([[0.5,0.3],[0.5,0.3],[0.5,0.3],[0.5,0.3]])
        X_2 = np.array([[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]])
        X_3 = np.array([[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]])
        X = np.concatenate((X_1, X_2),axis=0)
        X = np.concatenate((X, X_3),axis=0)

        Y_left = np.concatenate((X_1, X_2),axis=0)
        Y_right = X_3.copy()

        # assertions
        X_left, X_right,_ = split(X, 0, 0.5)
        assert_array_equal(X_left, Y_left)
        assert_array_equal(X_right, Y_right)

        X_left, X_right,_ = split(X, 1, 0.3)
        assert_array_equal(X_left, Y_left)
        assert_array_equal(X_right, Y_right)

    def test_entropy(self):

        # test case
        P_1, H_1 = np.array([0.5, 0.5]), -1
        P_2, H_2 = np.array([1.0, 0.0]), 0.0
        P_3, H_3 = np.array([0.133, 0.133, 0.25, 0.484]), -1.7809

        # assertions
        H_actual_1 = Splitter.entropy(P_1)
        self.assertEqual(round(H_actual_1,4), H_1)

        H_actual_2 = Splitter.entropy(P_2)
        self.assertEqual(round(H_actual_2,4), H_2)

        H_actual_3 = Splitter.entropy(P_3)
        self.assertEqual(round(H_actual_3,4), H_3)


    def test_tree_traversal(self):

        # define dummy tree
        root = Node(leaf_node=False, feature_index=1, threshold=0.4)
        root['left_node'] = Node(leaf_node=True, histogram=np.array([0.0, 1.0]))
        root['right_node'] = Node(leaf_node=False, feature_index=0, threshold=0.25)
        root['right_node']['left_node'] = Node(leaf_node=True, histogram=np.array([0.0, 1.0]))
        root['right_node']['right_node'] = Node(leaf_node=True, histogram=np.array([1.0, 0.0]))

        # define classification model
        model = decision_tree(0)
        model.root = root

        X = np.array([[0.5, 0.3], [0.5, 0.5], [0.0, 0.0]])
        Y = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

        # assertions
        actual = model.predict(X)
        assert_array_equal(actual, Y)


    def test_metric(self):

        # create descion_tree obj, to define objective function
        model_ig = Splitter(algo="cart", obj_fcn="ig", p=0)
        model_gini = Splitter(algo="cart", obj_fcn="gini", p=0)
        
        # test cases (1)
        y_left1, y_right1 = np.array([0,0,0,1]), np.array([0,1])
        expected1_ig = -5.25
        expected1_gini = 3.5

        # test cases (2)
        y_left2, y_right2 = np.array([0,1,2,0,1,2]), np.array([0,1,1])
        expected2_ig = -12.26
        expected2_gini = 3.67

        # test cases (3)
        y_left3, y_right3 = np.array([0,0,0,0,0,0]), np.array([1,1,2])
        expected3_ig = -2.75
        expected3_gini = 7.67        

        # assertions
        model_ig.C = 2
        model_gini.C = 2
        self.assertEqual(expected1_ig, round(sum(model_ig.metric(y_left1, y_right1)),2))
        self.assertEqual(expected1_gini, round(sum(model_gini.metric(y_left1, y_right1)),2))

        model_ig.C = 3
        model_gini.C = 3
        self.assertEqual(expected2_ig, round(sum(model_ig.metric(y_left2, y_right2)),2))
        self.assertEqual(expected2_gini, round(sum(model_gini.metric(y_left2, y_right2)),2))
        
        model_ig.C = 3
        model_gini.C = 3        
        self.assertEqual(expected3_ig, round(sum(model_ig.metric(y_left3, y_right3)),2))
        self.assertEqual(expected3_gini, round(sum(model_gini.metric(y_left3, y_right3)),2))


    def test_increment_gini(self):

        # test case (1)
        M_left1, M_right1 = np.array([4, 5, 6]), np.array([11, 21, 33])
        N_left1, N_right1 = 15, 65
        i1 = 1
        score_left1, score_right1 = np.power(M_left1/N_left1,2).sum(), np.power(M_right1/N_right1,2).sum()
        expected_score = 30.656


        # assertion 1
        score_left = gini_increment(score_left1, M_left1[i1]+1, N_left1+1, 1)
        score_right = gini_increment(score_right1, M_right1[i1]-1, N_right1-1, -1)
        actual_score = (N_left1+1)*score_left + (N_right1-1)*score_right
        self.assertEqual(round(actual_score,3), expected_score)


# run tests
if __name__ == "__main__":
    unittest.main()

