import numpy as np
import scipy.io
from neural_networks import predict
from display_data import display_data

data = scipy.io.loadmat('./data/ex3data1.mat')
X = data['X']
y = data['y']

m = X.shape[0]

parameters = scipy.io.loadmat('./data/ex3weights.mat')
Theta1 = parameters['Theta1']
Theta2 = parameters['Theta2']

rp = np.random.permutation(m)

for i in range(0, m + 1):
    digit = X[rp[i], :].reshape([1, 400])

    print('\nDisplaying Example Image\n')
    display_data(digit)

    pred = predict(Theta1, Theta2, digit)
    print(f'\nNeural Network Prediction: {pred} (digit {pred % 10})\n')
    print(f'\nActual value: {y[rp[i]] % 10}\n')
    
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
      break
