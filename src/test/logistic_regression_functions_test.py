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
