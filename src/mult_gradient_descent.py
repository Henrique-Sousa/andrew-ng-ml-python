import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing
import linear_regression

(X, y) = data_preprocessing.load('ex1data2.txt')
(X, mu, sigma) = data_preprocessing.feature_normalize(X)
X = data_preprocessing.with_leading_ones(X)

alphas = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001] 
iters = 400

plt.title('Cost function')
plt.xlabel('No. of iterations')
plt.ylabel('Cost J')

for alpha in alphas:
    theta = np.zeros([3, 1])
    (theta, J_history) = linear_regression.gradient_descent(X, y, theta, alpha, iters)
    plt.plot(range(0, len(J_history)), J_history[0:])

plt.legend(alphas)
plt.show()

alpha = 1
iters = 40

theta = np.zeros([3, 1])

(theta, J_history) = linear_regression.gradient_descent(X, y, theta, alpha, iters)
sqft = (1650 - mu[0]) / sigma[0]
br = (3 - mu[1]) / sigma[1]
x = np.array([1, sqft, br])
price = (x @ theta).item()
print(f'Predicted price of a 1650 sq-ft, 3 br house using gradient descent at a learning rate of {alpha} with {iters} iterations: {price}')
print(f'Theta: {theta}')
