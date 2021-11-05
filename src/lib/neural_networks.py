import numpy as np
from data_preprocessing import *
from logistic_regression import sigmoid

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def rand_initialize_weights(L_in, L_out):
    epsilon_init = 0.12
    W = np.random.random([L_out, 1 + L_in]) * 2 * epsilon_init - epsilon_init
    return W

def nn_cost_function(nn_params,
                input_layer_size,
                hidden_layer_size,
                num_labels,
                X, y, lmbda):
    '''
    NNCOSTFUNCTION Implements the neural network cost function for a two layer
    neural network which performs classification

       Computes the cost and gradient of the neural network.
       
       The parameters for the neural network are "unrolled" into the vector
       nn_params and need to be converted back into the weight matrices. 
     
       The returned parameter grad is an "unrolled" vector of the
       partial derivatives of the neural network.
    '''
    
    # Reshape nn_params back into the parameters Theta1 and Theta2,
    # the weight matrices for our 2 layer neural network
    Theta1 = (nn_params[0: hidden_layer_size * (input_layer_size + 1)]
        .reshape(hidden_layer_size, (input_layer_size + 1)))
    Theta2 = (nn_params[(hidden_layer_size * (input_layer_size + 1)):]
        .reshape( num_labels, (hidden_layer_size + 1)))
    
    # Setup some useful variables
    m = X.shape[0]

    # 1: Feedforward the neural network and return the cost in the variable J
    labels = np.arange(1, num_labels + 1)
    y_dummy = (y == labels).astype(np.int8)
    X = with_leading_ones(X)
    # print(f'Theta1.shape: {Theta1.shape}')
    # print(f'X.shape: {X.shape}')
    # print(f'(Theta1 @ X.T).shape: {(Theta1 @ X.T).shape}')
    # print(f'y_dummy.shape: {y_dummy.shape}')
    # print(f'Theta1.T.shape: {Theta1.T.shape}')
    # print(f'X[0].shape: {X[0].shape}')
    # print(f'X[0].reshape([1, input_layer_size + 1]).shape: {X[0].reshape([1, input_layer_size + 1]).shape}')
    # print((X.reshape([m, input_layer_size + 1]) @ Theta1.T).shape)
    z2 = X @ Theta1.T
    # print(f'z2.shape: {z2.shape}')
    a2 = with_leading_ones(sigmoid(z2))
    z3 = a2 @ Theta2.T
    # print(f'a2.shape: {a2.shape}')
    # print(f'Theta2.shape: {Theta2.shape}')
    # print(f'z3.shape: {z3.shape}')
    # print(f'np.log(sigmoid(z3)).shape: {np.log(sigmoid(z3)).shape}')
    # print(f'y_dummy.shape: {y_dummy.shape}')
    a3 = sigmoid(z3)
    J = (1/m) * np.sum(-y_dummy * np.log(a3) - (1 -y_dummy) * np.log(1 - a3))
    J += (lmbda / (2 * m)) * (np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2))

    ## Part 2: Implement the backpropagation algorithm to compute the gradients
    ##         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    ##         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    ##         Theta2_grad, respectively. After implementing Part 2, you can check
    ##         that your implementation is correct by running checkNNGradients
    ##
    ##         Note: The vector y passed into the function is a vector of labels
    ##               containing values from 1..K. You need to map this vector into a 
    ##               binary vector of 1's and 0's to be used with the neural network
    ##               cost function.
    ##
    ##         Hint: We recommend implementing backpropagation using a for-loop
    ##               over the training examples if you are implementing it for the 
    ##               first time.
    ##
    ## Part 3: Implement regularization with the cost function and gradients.
    ##
    ##         Hint: You can implement this around the code for
    ##               backpropagation. That is, you can compute the gradients for
    ##               the regularization separately and then add them to Theta1_grad
    ##               and Theta2_grad from Part 2.
    ##
    #       
    # You need to return the following variables correctly 
    #J = 0;
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    # Unroll gradients
    grad = np.block([Theta1.flatten(), Theta2.flatten()])
    return J, grad

def predict(Theta1, Theta2, X):
    X = with_leading_ones(X)
    a2 = sigmoid(X @ Theta1.T)
    a2 = with_leading_ones(a2)
    a3 = sigmoid(a2 @ Theta2.T)
    return 1 + np.argmax(a3, axis=1)
