import numpy as np
import os

def load():
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

    return (X, y)


def with_leading_ones(X):
    m = X.shape[0]
    ones = np.ones([m, 1])
    new_X = np.concatenate([ones, X], axis=1)
    return new_X
