import numpy as np
import scipy.io
from logistic_regression import *

num_labels = 10

mat = scipy.io.loadmat('./data/ex3data1.mat')
X = mat['X']
y = mat['y']

m = X.shape[0]

lbda = 0.1
all_theta = one_vs_all(X, y, num_labels, lbda)
pred = predict_one_vs_all(all_theta, X)
accuracy = np.mean((pred == y.reshape(m)).astype(np.float16)) * 100
print(f'Training Set Accuracy: {accuracy}')
