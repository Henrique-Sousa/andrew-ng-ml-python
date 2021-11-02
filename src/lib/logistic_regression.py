import numpy as np
from scipy.optimize import * 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(X, theta):
    return sigmoid(X @ theta)

def cost_function(theta, X, y):
    n = X.shape[1]
    if theta.shape == (n,):
        theta = theta.reshape(n, 1)
    m = y.shape[0]
    cost = (1 / m) * ((-y.T @ np.log(h(X, theta))) - ((1 - y).T @ np.log(1 - h(X, theta))))
    return cost.item()

def gradient(theta, X, y):
    m = y.shape[0]
    return (1 / m) * (X.T @ (h(X, theta) - y))

def cost_function_and_gradient(theta, X, y):
    J = cost_function(theta, X, y) 
    grad = gradient(theta, X, y) 
    return (J, grad)

def fit(initial_theta, X, y, maxiter):
    m, n = X.shape
    y = y.reshape(m)
    initial_theta = initial_theta.reshape(n)
    result = minimize(
        fun = cost_function,
        x0 = initial_theta,
        args = (X, y),
        method = 'BFGS',
        jac = lambda t, X, y: np.ndarray.flatten(gradient(t.reshape(n), X, y)),
        options = {'maxiter': maxiter})
    return (result.x.reshape(n, 1), result.fun)

def fminunc(initial_theta, X, y, maxiter):
    m, n = X.shape
    y = y.reshape(m)
    initial_theta = initial_theta.reshape(n)
    result = minimize(
        fun = cost_function_and_gradient,
        x0 = initial_theta,
        args = (X, y),
        method = 'BFGS',
        jac = True, 
        options = {'maxiter': maxiter})
    return (result.x.reshape(n, 1), result.fun)

def predict(theta, X):
    return (sigmoid(X @ theta) > 0.5).astype(np.int8)

def cost_function_with_regularization(theta, X, y, lbda):
    m = y.shape[0]
    J = cost_function(theta, X, y) + (lbda / (2 * m)) * (theta[1:].T @ theta[1:]).item()
    grad = gradient(theta, X, y)
    grad[1:] += (lbda / m) * theta[1:]
    return J, grad

def fit_with_regularization(initial_theta, X, y, maxiter, lbda):
    m, n = X.shape
    y = y.reshape(m)
    initial_theta = initial_theta.reshape(n)
    result = minimize(
        fun = cost_function_with_regularization,
        x0 = initial_theta,
        args = (X, y, lbda),
        method = 'BFGS',
        jac = True, 
        options = {'maxiter': maxiter})
    return (result.x.reshape(n, 1), result.fun)
