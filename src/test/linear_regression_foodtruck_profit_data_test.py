import numpy as np
from linear_regression import *
import pytest
import data_preprocessing

data = data_preprocessing.load_data('ex1data1.txt')
(X, y) = data_preprocessing.separate_X_and_y(data)
X = data_preprocessing.with_leading_ones(X)

def test_cost_function_with_zeros_as_theta():

    theta = np.zeros([2, 1])
    J = compute_cost(X, y, theta)

    assert round(J, 2) == 32.07

def test_cost_function_with_theta():

    theta = np.array([
        [-1],
        [2]
    ])

    J = compute_cost(X, y, theta)

    assert round(J, 2) == 54.24

def test_gradient_descent():

    initial_theta = np.zeros([2, 1])
    iterations = 1500
    alpha = 0.01
    (theta, _) = gradient_descent(X, y, initial_theta, alpha, iterations)

    assert theta[0] == pytest.approx(-3.6303, 0.001)
    assert theta[1] == pytest.approx(1.1664, 0.001)
