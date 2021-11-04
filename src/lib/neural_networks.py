from data_preprocessing import *
from logistic_regression import sigmoid

def predict(Theta1, Theta2, X):
    X = with_leading_ones(X)
    a2 = sigmoid(X @ Theta1.T)
    a2 = with_leading_ones(a2)
    a3 = sigmoid(a2 @ Theta2.T)
    return 1 + np.argmax(a3, axis=1)
