import numpy as np
import matplotlib.pyplot as plt
from linear_regression import * 
import data_preprocessing
from matplotlib.pyplot import figure

(X, y) = data_preprocessing.load()
X = data_preprocessing.with_leading_ones(X)

iterations = 1500
alpha = 0.01

initial_theta = np.zeros([2, 1])
(theta, J_history) = gradient_descent(X, y, initial_theta, alpha, iterations)
print(theta)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), dpi=80)

ax1.scatter(X[:, 1], y, c='red', marker='x')
ax1.plot(X[:, 1], X @ theta, c='blue')
ax1.set_title('Linear Regression')
ax1.set(xlabel='Population of City in 10,000s', ylabel='Profit in $10,000s')
ax1.legend(['Linear regression', 'Training data'])

# plotting cost:
# ignoring first values since they are too high
# and this makes visualization hard
ax2.plot(range(2, len(J_history)), J_history[2:])
ax2.set_title('Cost function')
ax2.set(xlabel='No. of iterations', ylabel='Cost J')

plt.show()

predict1 = np.array([1, 3.5]) @ theta
print(f'For population = 35,000, we predict a profit of {predict1 * 10000}')

predict2 = np.array([1, 7]) @ theta
print(f'For population = 70,000, we predict a profit of {predict2 * 10000}')
