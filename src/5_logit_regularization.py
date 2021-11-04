import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing as dpp
import logistic_regression as lr

data = dpp.load_data('ex2data2.txt')
X_initial, y = dpp.separate_X_and_y(data)
pos = np.where(y == 1)
neg = np.where(y == 0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=80)
lbdas = [0, 1, 100]

for lbda, ax in zip(lbdas, axes):

    ax.scatter(X_initial[pos, 0], X_initial[pos, 1], c='k', marker='+')
    ax.scatter(X_initial[neg, 0], X_initial[neg, 1], c='y', marker='o')
    ax.set(xlabel='Microship Test 1')
    ax.set(ylabel='Microship Test 2')

    X = dpp.map_feature(X_initial[:, 0], X_initial[:, 1], 6)
    m, n = X.shape
    initial_theta = np.zeros([n, 1])
    maxiter = 400
    theta, J = lr.fit_with_regularization('BFGS', initial_theta, X, y, maxiter, lbda) 

    u = np.linspace(-1, 1.5, 50).reshape([50, 1]);
    v = np.linspace(-1, 1.5, 50).reshape([50, 1]);

    z = np.zeros([len(u), len(v)]);
    for i in range(0, len(u)):
        for j in range(0, len(v)):
            z[i, j] = dpp.map_feature(u[i], v[j], 6) @ theta;

    z = z.T

    ax.contour(u.reshape(50), v.reshape(50), z, 0)
    ax.legend(['y = 1', 'y = 0', 'Decision Boundary'])
    ax.set_title(f'lambda = {lbda}')

plt.show()
