import h5py
import numpy as np


def sigmoid(Z):
    """
    Applies sigmoid function element-wise to matrix Z
    """
    A = 1 / (1 + np.exp(-Z))
    return A


def activation_forward(Z, *, activation='relu'):
    """
    Applies various activation functions element-wise to input matrix Z.

    Parameters
    ----------
    Z : np.array(float)
        input matrix
    activation : str
        what activation function to use for the layer (relu, sigmoid, tanh)

    Returns
    -------
    A : np.array(float)
        the "activation matrix"
    """
    A = np.zeros(Z.shape)
    valid_functions = {'relu', 'sigmoid', 'tanh'}
    assert activation in valid_functions, "Not a valid activation function."

    if activation == 'relu':
        A = np.maximum(0, Z)
    elif activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'tanh':
        A = np.tanh(Z)

    return A


def fully_connected_forward(X, *, num_units, activation='relu', params, hyperparams):
    """
    Parameters
    ----------
    X : np.array(float)
        input from previous layer
    num_units : int
        the number of "neurons"/units in the layer
    activation : str
        what activation function to use for the layer (relu, sigmoid, tanh)
    params : dict of (str: np.array(float))
        cache of parameters for backprop calculations

    hyperparams : dict of (str: int or str))
        cache of hyperparameters

    Returns
    -------
    a : np.array(float)
        output of fully connected NN layer
    params : dict of (str: np.array(float))
        cache of parameters for backprop calculations, updated with current
        layer's parameters
    hyperparams : dict of (str: int or str)
        cache of  hyperparameters, updated with current layers' hyperparameters
    """
    # get dimensions of X
    n_x, m = X.shape

    # get current layer number
    if 'l' not in hyperparams:
        hyperparams['l'] = 0
    l = hyperparams['l'] + 1

    Z = np.zeros((num_units, m))
    A = np.zeros((num_units, m))

    if 'W' + str(l) not in params:
        # RANDOMLY initialize the weight matrix
        # (note that initializing to zero will make every neuron perform the same
        # computation, and so every neuron's gradients will be the same; 
        # i.e., this layer would only contain one unique neuron and "units" might as
        # well be always equal to 1)
        W = np.random.randn(num_units, n_x) * 0.01  # <- multiply by this small
                                                    # (arbitrary) number since some
                                                    # activation functions only
                                                    # have meaningful gradients
                                                    # near the origin

        # every neuron's bias can be initialized to zero since the weight for each
        # neuron is virtually guaranteed to be different 
        b = np.zeros((num_units, 1))
    else:
        W = params['W' + str(l)]
        b = params['b' + str(l)]

    Z = np.dot(W, X) + b    # we can add the bias—(units, 1)—to the dot
                            # product—(units, n_x)—thanks to numpy broadcasting

    A = activation_forward(Z, activation=activation)

    # update caches
    params['W' + str(l)] = W
    params['b' + str(l)] = b
    params['Z' + str(l)] = Z
    params['A' + str(l)] = A
    hyperparams['l'] += 1
    hyperparams['l' + str(l) + '_units'] = num_units
    hyperparams['l' + str(l) + '_activation'] = activation

    return A, params, hyperparams


def binary_logistic_loss(Y_true, Y_pred):
    """

    """
    L = - Y_true * np.log(Y_pred) - (1 - Y_true) * np.log(1 - Y_pred)
    return L


def binary_cost(Y_true, Y_pred):
    """
    Computes the binary logistic cost function.
    """
    _, m = Y_true.shape

    L = binary_logistic_loss(Y_true, Y_pred)
    J = (1 / m) * np.sum(L)

    return J


def relu_backward(Z, dA):
    """

    """
    dZ = np.copy(dA)
    dZ[Z < 0] = 0
    return dZ


def sigmoid_backward(Z, dA):
    """

    """
    dZ = np.zeros(dA.shape)
    s = sigmoid(Z)
    dZ = dA * (1 - s) * s
    return dZ


def tanh_backward(Z, dA):
    """

    """
    dZ = np.zeros(dA.shape)
    t = np.tanh(Z)
    dZ = dA * (1 - (t ** 2))
    return dZ


def activation_backward(Z, dA, *, activation='relu'):
    """

    """
    dZ = np.zeros(dA.shape)
    valid_functions = {'relu', 'sigmoid', 'tanh'}
    assert activation in valid_functions, "Not a valid activation function."

    if activation == 'relu':
        dZ = relu_backward(Z, dA)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(Z, dA)
    elif activation == 'tanh':
        dZ = tanh_backward(Z, dA)

    return dZ


def fully_connected_backward(X, Y_true, Y_pred, *, learning_rate, params,
                             hyperparams, grads):
    """
    Chain rule time

    Parameters
    ----------
    X : np.array(float)
    Y_true : np.array(float
    params : dict of (str: np.array(float))
    hyperparams : dict of (str: int or str)
    grads : dict of (str: np.array(float))
    """
    num_layers = len(params) // 4
    _, m = X.shape

    #dAL = - (Y_true / Y_pred) + ((1 - Y_true) / (1 - Y_pred))
    dAL = - np.divide(Y_true , Y_pred) + np.divide((1 - Y_true) , (1 - Y_pred))
    grads['dA' + str(num_layers)] = dAL

    for i in reversed(range(1, num_layers+1)):
        Ai = params['A' + str(i)]
        Zi = params['Z' + str(i)]
        Wi = params['W' + str(i)]
        bi = params['b' + str(i)]
        activation = hyperparams['l' + str(i) + '_activation']

        if i > 1:
            Ai_minus_1 = params['A' + str(i-1)]
        else:
            Ai_minus_1 = np.copy(X)

        dAi = grads['dA' + str(i)]
        dZi = activation_backward(Zi, dAi, activation=activation)
        dWi = (1 / m) * np.dot(dZi, Ai_minus_1.T)
        dbi = (1 / m) * np.sum(dZi, axis=1, keepdims=True)

        dAi_minus_1 = np.dot(Wi.T, dZi)

        grads['dZ' + str(i)] = dZi
        grads['dW' + str(i)] = dWi
        grads['db' + str(i)] = dbi
        grads['dA' + str(i-1)] = dAi_minus_1

        params['W' + str(i)] -= learning_rate * dWi
        params['b' + str(i)] -= learning_rate * dbi

    return grads


def train(X, Y_true, params, hyperparams, grads, learning_rate, num_epochs,
          seed=0):
    """

    """
    np.random.seed(seed)
    learning_rate = 0.001
    num_epochs = 2000

    params = {}
    hyperparams = {}
    grads = {}

    for i in range(num_epochs):
        A1, params, hyperparams = fully_connected_forward(X, num_units=7,
                                                          activation='relu',
                                                          params=params,
                                                          hyperparams=hyperparams)
        Y_pred, params, hyperparams = fully_connected_forward(A1, num_units=1,
                                                              activation='sigmoid',
                                                              params=params,
                                                              hyperparams=hyperparams)
        cost = binary_cost(Y_true, Y_pred)

        grads = fully_connected_backward(X, Y_true, Y_pred, learning_rate=learning_rate,
                                 params=params, hyperparams=hyperparams, grads=grads)

        # TODO: fix this hack-y way of not letting the system think there are
        # more layers when we're just going to the next epoch
        hyperparams = {}

