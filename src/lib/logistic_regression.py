import numpy as np
from scipy.optimize import * 
from data_preprocessing import *

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

def one_vs_all(X, y, num_labels, lbda):
    m, n = X.shape
    labels = np.unique(y)
    initial_theta = np.zeros([n + 1, 1])
    X = with_leading_ones(X)
    all_theta = np.zeros([num_labels, n + 1])
    for i in range(0, num_labels):
        y_i = (y == labels[i]).astype(np.int8)
        theta, _ = fit_with_regularization(initial_theta, X, y_i, 400, lbda) 
        print(f'Training regularized logistic regression classifier for label {i}')
        all_theta[i, :] = theta.reshape(n + 1)
    return all_theta

def predict_one_vs_all(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    X = with_leading_ones(X)
    probs = sigmoid(X @ all_theta.T)
    return 1 + np.argmax(probs, axis=1)
