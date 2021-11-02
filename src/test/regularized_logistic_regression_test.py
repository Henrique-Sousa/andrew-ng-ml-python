import numpy as np
import pytest
import data_preprocessing
from logistic_regression import *

data = data_preprocessing.load_data('ex2data2.txt')
(X, y) = data_preprocessing.separate_X_and_y(data)
X = data_preprocessing.map_feature(X[:, 0], X[:, 1], 6)

(m, n) = X.shape
initial_theta = np.zeros([n, 1])
test_theta = np.ones([n, 1])

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

def test_grad_with_regularization_test_theta__lambda_10():
    cost, grad = cost_function_with_regularization(test_theta, X, y, 10);
    assert cost == pytest.approx(3.16, 0.01)
    # Expected gradients (approx) - first five values only
    assert np.allclose(
        grad[0:5],
        np.array([
            [0.3460],
            [0.1614],
            [0.1948],
            [0.2269],
            [0.0922]]), atol=0.0001)
