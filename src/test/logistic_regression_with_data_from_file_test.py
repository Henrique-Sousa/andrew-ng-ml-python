import numpy as np
import pytest
import data_preprocessing
from logistic_regression import *

data = data_preprocessing.load_data('ex2data1.txt')
(X, y) = data_preprocessing.separate_X_and_y(data)
X = data_preprocessing.with_leading_ones(X)
(m, n) = X.shape

initial_theta = np.zeros([n, 1])
test_theta = np.array([[-24], [0.2], [0.2]])

def test_cost_with_zeros_theta():
    cost = cost_function(initial_theta, X, y)
    assert cost == pytest.approx(0.693, 0.01)

def test_gradient_with_zeros_theta():
    grad = gradient(initial_theta, X, y)
    assert np.allclose(
        grad,
        np.array([
            [-0.1000],
            [-12.0092],
            [-11.2628]]), atol=0.001)
    
def test_cost_with_test_theta():
    cost = cost_function(test_theta, X, y)
    assert cost == pytest.approx(0.218, 0.01)

def test_gradient_with_test_theta():
    grad = gradient(test_theta, X, y)
    assert np.allclose(
        grad,
        np.array([
        [0.043],
        [2.566],
        [2.647]]), atol=0.001)

def test_fit():
    (theta, cost) = fit(initial_theta, X, y, maxiter = 400)
    assert cost == pytest.approx(0.203, 0.03)
    assert np.allclose(
        theta,
        np.array([
            [-25.161],
            [0.206],
            [0.201]]), atol=0.001)

def test_fminunc():
    (theta, cost) = fminunc(initial_theta, X, y, maxiter = 400)
    assert cost == pytest.approx(0.203, 0.03)
    assert np.allclose(
        theta,
        np.array([
            [-25.161],
            [0.206],
            [0.201]]), atol=0.001)
    prob = sigmoid(np.array([1, 45, 85]) @ theta)
    assert prob == pytest.approx(0.775, 0.002)
