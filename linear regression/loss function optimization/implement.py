import numpy as np

''' mean square error '''

def linear_regression(x, y):
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((x, one), axis = 1)
    ANS = np.dot(np.linalg.pinv(x), y)
    return ANS

def predict(x, w):
    one = np.ones((x.shape[0], 1))
    x = np.concatenate((x, one), axis = 1)
    return np.dot(x, w)
