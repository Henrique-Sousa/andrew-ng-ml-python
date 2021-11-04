import pytest
import numpy as np
from logistic_regression import *

def test_sigmoid_function():
    assert sigmoid(0) == 0.5
    assert sigmoid(100) == pytest.approx(1)
    assert sigmoid(-100) == pytest.approx(0)

def test_sigmoid_with_matrix():
    A = np.array([
        [0, 100],
        [-100, 0]])
    S = sigmoid(A)
    T = np.array([
        [0.5, 1],
        [0, 0.5]])
    assert np.allclose(S, T)

def test_cost_and_gradient_with_regularization():
    theta_t = np.array([
        [-2],
        [-1],
        [1],
        [2]])
    X_t = np.block([np.ones([5, 1]), np.arange(1, 16).reshape([3,5]).T/10])
    y_t = (np.array([[1],[0],[1],[0],[1]]) >= 0.5).astype(np.int8)
    lambda_t = 3
    J, grad = cost_function_with_regularization(theta_t, X_t, y_t, lambda_t)
    assert J == pytest.approx(2.534819, 0.000001)
    assert np.allclose(
        grad, 
        np.array([
            [0.146561],
            [-0.548558],
            [0.724722],
            [1.398003]]), atol=0.000001)
