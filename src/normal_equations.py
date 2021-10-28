import numpy as np
import data_preprocessing
import linear_regression

(X, y) = data_preprocessing.load('ex1data2.txt')
X = data_preprocessing.with_leading_ones(X)

theta = linear_regression.normal_equation(X, y)
print(f'Theta computed from the normal equations: {theta}')

x = np.array([1, 1650, 3])
price = (x @ theta).item()
print(f'Predicted price of a 1650 sq-ft, 3 br house (using normal equations): {price}')
