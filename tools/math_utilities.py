import numpy


def softmax(Z):
    Y = numpy.exp(Z)
    Y_nrm = Y.sum(axis=1)
    Y /= Y_nrm.reshape(len(Y_nrm), 1)
    return Y
