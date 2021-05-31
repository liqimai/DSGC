import numpy as np
from scipy import linalg
# from cvxopt import matrix, solvers

def inter_intra_variance(X, Y, display=True):
    Q,_,_,_,V = QM(X, Y)
    V, Q = np.trace(V), np.trace(Q)
    variance = [V, Q, V-Q, Q/V, (V-Q)/V, Q/(V-Q), (V-Q)/Q,
                np.sqrt(V), np.sqrt(Q), np.sqrt(V-Q), np.sqrt(Q/(V-Q)), np.sqrt((V-Q)/Q)]
    format ="TotalVar\t{}\n"\
            "IntraVar\t{}\n"\
            "InterVar\t{}\n"\
            "Intra/Total\t{}\n"\
            "Inter/Total\t{}\n"\
            "Intra/Inter\t{}\n"\
            "Inter/Intra\t{}\n"\
            "TotalSTD\t{}\n"\
            "IntraSTD\t{}\n"\
            "InterSTD\t{}\n"\
            "IntraSTD/InterSTD\t{}\n"\
            "InterSTD/IntraSTD\t{}\n"
    if display:
        print(format.format(*variance))
    return variance

def QM(X, Y):
    mask = Y.sum(1).astype(np.bool)
    u = X[mask].mean(axis=0, keepdims=True).T
    V = np.cov(X[mask], rowvar=False, bias=True)

    M = X.T.dot(Y) / Y.sum(axis=0, keepdims=True)
    Pi = np.diag(Y.sum(axis=0) / Y.sum())
    Q = V + u.dot(u.T) - M.dot(Pi).dot(M.T)
    return Q, M, Pi, u, V


def cvr(X, Y, lam=0.):
    n, m = X.shape
    n2, k = Y.shape
    assert n2 == n

    Q, M, _, _, V = QM(X, Y)
    inter_intra_variance(X, Y)
    O = np.zeros([k, k])
    A = [[2 * Q + 2*lam*np.ones_like(Q), M],
         [M.T, O]]
    A = np.block(A)

    O = 2*lam*np.ones([m, m])
    B = [[O],
         [M.T]]
    B = np.block(B)
    # try:
    #     # F = cvxopt_solve_qp(Q, q=np.zeros(m), A=M.T, b=M.T)
    #     # F = linalg.solve(A, B, assume_a='sym')
    # except linalg.LinAlgError:
    # print('singular A')
    Ainv = linalg.pinvh(A)
    F = Ainv.dot(B)
    # assert np.allclose(np.dot(A, F), B, atol=1e-5)

    F = F[:m, :m]
    X = X.dot(F)
    inter_intra_variance(X.dot(F), Y)
    return X, F

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


if __name__ == '__main__':
    # from gcn.utils import load_public_split_data
    # adj, X, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_public_split_data('cora')
    # X = X.toarray().astype(np.float64)
    # Y = (y_train + y_val + y_test).astype(np.float64)

    X = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ])
    Y = np.array([
        [1,0],
        [1,0],
        [1,0],
        [0,1],
        [0,1],
    ])
    inter_intra_variance(X,Y)

    G = np.array([
        [0.666666666666667, 0.333333333333333, 0, 0, 0],
        [0.333333333333333, 0.333333333333333, 0.333333333333333, 0, 0],
        [0, 0.333333333333333, 0.666666666666667, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    inter_intra_variance(G.dot(X),Y)

    F = np.array([
        [0.500000000000000, 0.500000000000000, 0, 0],
        [0.500000000000000, 0.500000000000000, 0, 0],
        [0, 0, 0.500000000000000, 0.500000000000000],
        [0, 0, 0.500000000000000, 0.500000000000000],
    ])
    inter_intra_variance(G.dot(X).dot(F),Y)
