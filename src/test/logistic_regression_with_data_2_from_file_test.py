import numpy as np
import pytest
import data_preprocessing
from logistic_regression import *

data = data_preprocessing.load_data('ex2data2.txt')
(X, y) = data_preprocessing.separate_X_and_y(data)
X = data_preprocessing.map_feature(X[:, 0], X[:, 1], 6)

(m, n) = X.shape
initial_theta = np.zeros([n, 1])

def test_cost_with_regularization_theta_zeros_lambda_1():
    cost, _ = cost_function_with_regularization(initial_theta, X, y, 1)
    assert cost == pytest.approx(0.693, 0.01)

def test_grad_with_regularization_theta_zeros_lambda_1():
    _, grad = cost_function_with_regularization(initial_theta, X, y, 1)
    assert grad.shape[0] == n
    # Expected gradients (approx) - first five values only
    assert np.allclose(
        grad[0:5],
        np.array([
            [0.0085],
            [0.0188],
            [0.0001],
            [0.0503],
            [0.0115]]), atol=0.0001)
