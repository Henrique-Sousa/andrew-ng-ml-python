import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing
from logistic_regression import *
from data_preprocessing import *

data = data_preprocessing.load_data('ex2data2.txt')
(X, y) = data_preprocessing.separate_X_and_y(data)

pos = np.where(y == 1)
neg = np.where(y == 0)

plt.scatter(X[pos, 0], X[pos, 1], c='black', marker='+')
plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o')
plt.legend(['y = 1', 'y = 0'])
plt.xlabel('Microship Test 1')
plt.ylabel('Microship Test 2')

plt.show()
