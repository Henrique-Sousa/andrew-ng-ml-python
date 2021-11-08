import numpy as np
import scipy.io
from display_data import display_data

mat = scipy.io.loadmat('./data/ex3data1.mat')
X = mat['X']
y = mat['y']

m = X.shape[0]

rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]

display_data(sel)
