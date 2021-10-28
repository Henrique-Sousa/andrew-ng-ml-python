import numpy as np
import os
from linear_regression import *

X = np.array([
    [1, 2],
    [6, 3],
    [2, 7],
    [8, 3]])

theta = np.array([
    [4],
    [5]])

y = np.array([
    [3],
    [2],
    [5],
    [9]])

def test_h():
    result = h(X, theta)
    assert result.shape == (X.shape[0], 1)
    assert np.array_equal(
        result, 
        np.array([
            [14],
            [39],
            [43],
            [47]]))

def test_residuals():
    result = residuals(X, y, theta)
    assert np.array_equal(
        result,
        np.array([
            [11],
            [37],
            [38],
            [38]]))
    
def test_sum_of_squared_residuals():
    result = sum_of_squared_residuals(X, y, theta)
    assert result == 4378

def test_cost_function():
    result = compute_cost(X, y, theta)
    assert result == 547.25 

def test_gradient():
    result = gradient(X, y, theta)
    assert np.array_equal(
        result,
        np.array([
            [613],
            [513]]))

def test_normal_equation_non_invertible():
    X = np.ones([3, 3])
    y = np.ones([1, 3])
    theta = normal_equation(X, y)
    assert np.array_equal(theta, np.zeros(3))
