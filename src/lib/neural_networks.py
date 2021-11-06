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
        .reshape(hidden_layer_size, (input_layer_size + 1), order='F'))
    Theta2 = (nn_params[(hidden_layer_size * (input_layer_size + 1)):]
        .reshape( num_labels, (hidden_layer_size + 1), order='F'))
    
    # Setup some useful variables
    m = X.shape[0]

    # 1: Feedforward the neural network and return the cost in the variable J
    labels = np.arange(1, num_labels + 1)
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a 
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    y_dummy = (y == labels).astype(np.int8)
    X = with_leading_ones(X)
    a1 = X
    z2 = a1 @ Theta1.T
    a2 = with_leading_ones(sigmoid(z2))
    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)
    J = (1/m) * np.sum(-y_dummy * np.log(a3) - (1 -y_dummy) * np.log(1 - a3))
    J += (lmbda / (2 * m)) * (np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2))

    # 2: Backpropagation algorithm to compute the gradients Theta1_grad and Theta2_grad.
    #    return the partial derivatives of the cost function with respect
    #    to Theta1 and Theta2 in Theta1_grad and Theta2_grad, respectively.
    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)
    for i in range(0, m):
        d3 = (a3[i, :] - y_dummy[i, :]).reshape(num_labels, 1)
        aux = a2[i, :].reshape([hidden_layer_size + 1, 1]).T
        Delta2 += d3 @ aux
        d2 = ((Theta2.T @ d3).T[:, 1:] * sigmoid_gradient(z2[i, :])).T
        Delta1 += d2 @ a1[i, :].reshape([input_layer_size + 1, 1]).T
    Theta1_grad = (1 / m) * Delta1
    Theta2_grad = (1 / m) * Delta2

    # Unroll gradients
    grad = np.block([Theta1_grad.flatten(order='F'), Theta2_grad.flatten(order='F')])

    ## Part 3: Implement regularization with the cost function and gradients.
    ##
    ##         Hint: You can implement this around the code for
    ##               backpropagation. That is, you can compute the gradients for
    ##               the regularization separately and then add them to Theta1_grad
    ##               and Theta2_grad from Part 2.
    ##
    #       
    # You need to return the following variables correctly 
    #J = 0

    return J, grad

def debug_initialize_weights(fan_out, fan_in):
    #debuginitializeweights initialize the weights of a layer with fan_in
    #incoming connections and fan_out outgoing connections using a fixed
    #strategy, this will help you later in debugging
    #   w = debuginitializeweights(fan_in, fan_out) initializes the weights 
    #   of a layer with fan_in incoming connections and fan_out outgoing 
    #   connections using a fix set of values
    #
    #   note that w should be set to a matrix of size(1 + fan_in, fan_out) as
    #   the first row of w handles the "bias" terms
    
    # set w to zeros
    w = np.zeros([fan_out, 1 + fan_in])
    
    # initialize w using "sin", this ensures that w is always of the same
    # values and will be useful for debugging
    w = np.sin(np.arange(1, w.size + 1)).reshape(w.shape, order='F') / 10

    return w

def compute_numerical_gradient(J, theta):
    #COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    #and gives us a numerical estimate of the gradient.
    #   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    #   gradient of the function J around theta. Calling y = J(theta) should
    #   return the function value at theta.
    
    # Notes: The following code implements numerical gradient checking, and 
    #        returns the numerical gradient.It sets numgrad(i) to (a numerical 
    #        approximation of) the partial derivative of J with respect to the 
    #        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
    #        be the (approximately) the partial derivative of J with respect 
    #        to theta(i).)
    
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(0, theta.size):
        # Set perturbation vector
        perturb[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad
    
def check_nn_gradients(lmbda = 0):
    #   Creates a small neural network to check the
    #   backpropagation gradients, it will output the analytical gradients
    #   produced by your backprop code and the numerical gradients (computed
    #   using compute_numerical_gradient). These two gradient computations should
    #   result in very similar values.

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    # Reusing debug_initialize_weights to generate X
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + (np.arange(1, m + 1) % num_labels).T
    y = y.reshape([m, 1])

    # Unroll parameters
    nn_params = np.block([Theta1.flatten(order='F'), Theta2.flatten(order='F')])

    # Short hand for cost function
    cost_func = lambda p: nn_cost_function(p, input_layer_size, hidden_layer_size,
            num_labels, X, y, lmbda)

    _, grad = cost_func(nn_params)
    numgrad = compute_numerical_gradient(cost_func, nn_params)

    # Evaluate the norm of the difference between two solutions.  
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)

    return grad, numgrad, diff

def predict(Theta1, Theta2, X):
    X = with_leading_ones(X)
    a2 = sigmoid(X @ Theta1.T)
    a2 = with_leading_ones(a2)
    a3 = sigmoid(a2 @ Theta2.T)
    return 1 + np.argmax(a3, axis=1)
