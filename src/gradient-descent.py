import numpy as np
import matplotlib.pyplot as plt
import os
from linear_regression.compute_cost import compute_cost

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
local_data_path = 'data/ex1data1.txt'
data_path = os.path.join(current_dir, local_data_path)

data = np.genfromtxt(data_path, delimiter=',') 

m = data.shape[0]

X = data[:,0]
y = data[:,1]

X = X.reshape([m, 1])
y = y.reshape([m, 1])

ones = np.ones([m, 1])
X = np.concatenate([X, ones], axis=1)

theta = np.zeros([2, 1])
J = compute_cost(X, y, theta)

print(J)
