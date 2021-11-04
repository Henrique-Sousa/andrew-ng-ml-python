import numpy as np
import os

def load_data(file):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    local_data_path = os.path.join('data', file)
    src_dir = os.path.join(current_dir, '..')
    data_path = os.path.join(src_dir, local_data_path)

    data = np.genfromtxt(data_path, delimiter=',') 

    return data

def separate_X_and_y(data):

    m = data.shape[0]
    n = data.shape[1]

    X = data[:, 0:n-1]
    y = data[:, n-1]

    X = X.reshape([m, n-1])
    y = y.reshape([m, 1])

    return (X, y)

def with_leading_ones(X):
    m = X.shape[0]
    ones = np.ones([m, 1])
    new_X = np.concatenate([ones, X], axis=1)
    return new_X

def feature_normalize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_norm = (X - mu) / sigma
    return (X_norm, mu, sigma)

def map_feature(X1, X2, degree):
    m = X1.shape[0]
    out = np.ones([m, 1])
    X1 = X1.reshape([m, 1])
    X2 = X2.reshape([m, 1])
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            out = np.concatenate([out, (X1**(i-j)) * (X2**j)], axis=1)
    return out
