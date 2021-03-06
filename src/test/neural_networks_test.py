import numpy as np
import scipy.io
import pytest
from neural_networks import *

input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   

mat = scipy.io.loadmat('./data/ex3data1.mat')
X = mat['X']
y = mat['y']
m = X.shape[0]

parameters = scipy.io.loadmat('./data/ex3weights.mat')
Theta1 = parameters['Theta1']
Theta2 = parameters['Theta2']
nn_params = np.block([Theta1.flatten(order='F'), Theta2.flatten(order='F')])

def test_nn_cost_function():
    lmbda = 0
    J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                       num_labels, X, y, lmbda)
    assert J == pytest.approx(0.287629, 0.000001)

def test_nn_cost_function_with_regularization():
    lmbda = 1
    J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                       num_labels, X, y, lmbda)
    assert J == pytest.approx(0.383770, 0.000001)

def test_sigmoid_gradient():
    assert sigmoid_gradient(0) == 0.25

def test_backpropagation():
    grad, numgrad, diff = check_nn_gradients()

    assert np.allclose(numgrad, grad, atol=0.0001)
    assert diff <= 1e-9 

def test_backpropagation_with_regularization():
    lmbda = 3
    grad, numgrad, diff = check_nn_gradients(lmbda)

    assert np.allclose(numgrad, grad, atol=0.0001)
    assert diff <= 1e-9 
    debug_J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                       num_labels, X, y, lmbda)
    assert debug_J == pytest.approx(0.576051, 0.000001)
