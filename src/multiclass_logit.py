import numpy as np
import scipy.io
from display_data import display_data
from logistic_regression import *
from data_preprocessing import *

input_layer_size  = 400
num_labels = 10

mat = scipy.io.loadmat('./data/ex3data1.mat')
X = mat['X']
y = mat['y']

m = X.shape[0]

rand_indices = np.random.permutation(m);
sel = X[rand_indices[0:100], :]

display_data(sel);

lbda = 0.1
all_theta = one_vs_all(X, y, num_labels, lbda)
pred = predict_one_vs_all(all_theta, X)
accuracy = np.mean((pred == y.reshape(m)).astype(np.float16)) * 100
print(f'Training Set Accuracy: {accuracy}')
