import numpy as np

'''
Parameters
----------
X: [float]
    m x n array
y: [float] 
    m x 1 array of values of the output variable
theta: [float]
    n x 1 array
'''

def h(X, theta):
    return X @ theta 

def residuals(X, y, theta):
    return h(X, theta) - y

def sum_of_squared_residuals(X, y, theta):
    res = residuals(X, y, theta)
    return (res.T @ res)[0][0]

def compute_cost(X, y, theta):
    m = y.shape[0]
    ssr = sum_of_squared_residuals(X, y, theta)
    return (1 / (2 * m)) * ssr

def gradient(X, y, theta):
    return X.T @ residuals(X, y, theta)
    
def gradient_descent(X, y, theta, alpha, iterations):
    m = y.shape[0]
    J_history = []
    while iterations >= 0:
        J_history.append(compute_cost(X, y, theta))
        theta -= (alpha / m) * gradient(X, y, theta)
        iterations -= 1
    return (theta, J_history)

def normal_equation(X, y):
    try:
        inverse = np.linalg.inv(X.T @ X)
        return inverse @ X.T @ y
    except np.linalg.LinAlgError:
        # Matrix is noninvertible
        return np.zeros(X.shape[1])
