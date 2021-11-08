import numpy as np
import scipy.io
from neural_networks import predict

data = scipy.io.loadmat('./data/ex3data1.mat')
X = data['X']
y = data['y']

m = X.shape[0]

parameters = scipy.io.loadmat('./data/ex3weights.mat')
Theta1 = parameters['Theta1']
Theta2 = parameters['Theta2']

pred = predict(Theta1, Theta2, X)
accuracy = np.mean((pred == y.reshape(m)).astype(np.float16)) * 100
print(f'Training Set Accuracy: {accuracy}')
