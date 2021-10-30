import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(X, theta):
    return sigmoid(X @ theta)

def cost_function(theta, X, y):
    m = y.shape[0]
    return (1 / m) * ((-y.T @ np.log(h(X, theta))) - ((1 - y).T @ np.log(1 - h(X, theta))))

def gradient(theta, X, y):
    m = y.shape[0]
    return (1 / m) * (X.T @ (h(X, theta) - y))

def cost_function_and_gradient(theta, X, y):
    J = cost_function(theta, X, y) 
    grad = gradient(theta, X, y) 
    return (J, grad)
