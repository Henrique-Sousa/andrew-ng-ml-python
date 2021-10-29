import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing

data = data_preprocessing.load_data('ex2data1.txt')
(X, y) = data_preprocessing.separate_X_and_y(data)

pos = np.where(y == 1)
neg = np.where(y == 0)

plt.scatter(X[pos, 0], X[pos, 1], c='black', marker='+')
plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o')
plt.show()
