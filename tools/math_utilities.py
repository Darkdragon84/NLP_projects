import numpy


def softmax(Z):
    Y = numpy.exp(Z)
    Y /= Y.sum(axis=1).reshape(Y.shape[0], 1)
    return Y
