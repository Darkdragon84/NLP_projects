import time
from collections import defaultdict

import numpy
from scipy.sparse import csr_matrix
import timeit


def index_ops(xinds, yinds, binds, W):

    N = len(xinds)
    F, C = W.shape
    F -= 1

    # xinds = numpy.asarray(xinds)
    # yinds = numpy.asarray(yinds)
    # xinds_t = [[] for _ in range(C)]
    # for i, f in enumerate(xinds):
    #     xinds_t[f].append(i)

    ts = time.time()

    xinds_curr = xinds[binds]
    # xinds_t_curr = xinds_t[binds]
    yinds_curr = yinds[binds]

    Y_pred = W[xinds_curr] + W[-1]
    probs = numpy.array([Y_pred[i, j_i] for i, j_i in enumerate(yinds_curr)])

    for i, j_i in enumerate(yinds_curr):
        # Y_pred <- Y_pred - Y
        Y_pred[i, j_i] -= 1

    grad = numpy.zeros_like(W)

    for i, f in enumerate(xinds_curr):
        grad[f, :] += Y_pred[i]

    te = time.time() - ts
    print("index", te)

    return Y_pred, probs, grad


def sparse_ops(xinds, yinds, binds, W):
    N = len(xinds)
    F, C = W.shape
    F -= 1

    XS_full = csr_matrix(([1.]*2*N, (list(range(N)) * 2, xinds.tolist() + [F]*N)), (N, F + 1))
    YS_full = csr_matrix(([1.]*N, (list(range(N)), yinds.tolist())), (N, C))

    # N = len(binds)
    # nrange = list(range(N))
    # nrange2 = nrange*2
    # ones = [1.]* N
    # ones2 = ones*2
    # fs = [F]*N

    ts = time.time()

    # xinds_curr = xinds[binds]
    # yinds_curr = yinds[binds]

    # XS = csr_matrix((ones2, (nrange2, xinds_curr.tolist() + fs)), (N, F + 1))
    # YS = csr_matrix((ones, (nrange, yinds_curr)), (N, C))

    XS = XS_full[binds]
    # yinds_curr = yinds[binds]
    YS = YS_full[binds]

    Y_pred = XS.dot(W)

    probs = YS.multiply(Y_pred).data

    # YScoo = coo_matrix(YS)
    # for i, j_i in zip(YScoo.row, YScoo.col):
    for i, j_i in zip(YS.indptr[:-1], YS.indices):
        # Y_pred <- Y_pred - Y
        Y_pred[i, j_i] -= 1

    grad = XS.T.dot(Y_pred)
    # Ydiff = Y_pred - YS
    # Ydiff = YS._rsub_dense(Y_pred)

    te = time.time() - ts
    print("sparse", te)

    return Y_pred, probs, grad
    # return Ydiff, probs


def main():
    # N = 100000
    # F = 10000
    # C = 10000
    # bs = 1000
    N = 10000
    F = 10
    C = 10
    bs = 100
    W = numpy.random.randn(F + 1, C) / ((F + 1) * C)

    xinds = numpy.random.randint(0, F, N)
    yinds = numpy.random.randint(0, C, N)
    # xinds = numpy.random.randint(0, F, N).tolist()
    # yinds = numpy.random.randint(0, C, N).tolist()
    # assert isinstance(xinds, list)
    # assert isinstance(yinds, list)

    # XS = csr_matrix(([1.]*2*N, (list(range(N)) * 2, xinds + [F]*N)), (N, F + 1))
    # YS = csr_matrix(([1.]*N, (list(range(N)), yinds)), (N, C))

    sinds = numpy.arange(N)
    numpy.random.shuffle(sinds)
    binds = sinds[:bs]

    yp1, p1, grad1 = index_ops(xinds, yinds, binds, W)
    # yp2, p2 = sparse_ops(XS, YS, binds, W)
    yp2, p2, grad2 = sparse_ops(xinds, yinds, binds, W)
    #
    print("ypred", numpy.linalg.norm(yp1 - yp2))
    print("probs", numpy.linalg.norm(p1 - p2))
    print("grad", numpy.linalg.norm(grad1 - grad2))


if __name__ == '__main__':
    main()
