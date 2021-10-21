def h(X, theta):
    '''
    Parameters
    ----------
    X: [float]
        m x 2 array
    theta: [float]
        2 x 1 array

    Returns
    -------
    h: [float] 
        the computed function h(theta) of each observation
    '''

    return X @ theta 

def compute_cost(X, y, theta):
    '''
    Parameters
    ----------
    X: [float] 
        m x n array of values of the input variables
    y: [float] 
        m x 1 array of values of the output variable
    theta: [float] 
        array of coeficients

    Returns
    -------
    J: float
        the cost computed
    '''

    m = y.shape[0]
    residuals = h(X, theta) - y
    return (1 / (2 * m))*(residuals.T @ residuals)[0][0]
