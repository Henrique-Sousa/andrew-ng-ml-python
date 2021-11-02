import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing
from logistic_regression import *
from data_preprocessing import *

data = data_preprocessing.load_data('ex2data1.txt')
(X, y) = data_preprocessing.separate_X_and_y(data)

pos = np.where(y == 1)
neg = np.where(y == 0)

plt.scatter(X[pos, 0], X[pos, 1], c='black', marker='+')
plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o')
plt.legend(['Admited', 'Not admited'])

X = with_leading_ones(X)
(m, n) = X.shape
initial_theta = np.zeros([n, 1])
theta, cost = fminunc(initial_theta, X, y, 400)

xaxis = np.linspace(
        X[:, 1].min(),
        X[:, 1].max(), 100)

line = (-theta[1] * xaxis - theta[0]) / theta[2]
plt.plot(xaxis, line) 
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

plt.show()
