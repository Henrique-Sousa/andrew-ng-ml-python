import numpy as np
import os
from linear_regression import *

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
local_data_path = 'data/ex1data1.txt'
src_dir = os.path.join(current_dir, '..')
data_path = os.path.join(src_dir, local_data_path)

data = np.genfromtxt(data_path, delimiter=',') 

m = data.shape[0]

X = data[:,0]
y = data[:,1]

X = X.reshape([m, 1])
y = y.reshape([m, 1])

ones = np.ones([m, 1])
X = np.concatenate([ones, X], axis=1)


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
