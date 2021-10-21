import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('./data/ex1data1.txt', delimiter=',') 

X = data[:,0]
y = data[:,1]

plt.scatter(X, y, c='red', marker='x')
plt.show()
