import numpy as np
from shrink import shrink
from FastIllinoisSolver import FastIllinoisSolver
import time

from scipy.sparse.linalg import eigsh as largest_eigsh
import scipy.sparse.linalg


def DenseErrorSolver(Y, D, alpha, delta, global_max_iter, lasso_max_iter):
    '''
    Objective:  min_{X,L} ||X||_1 + alpha||L||_* + delta/2||E||_F^2 st: Y = DX + L + E
    Augmented Lagrangian Formulation : ||X||_1 + alpha||L||_* + <Lambda,Y - DX - L - E> + (beta/2)||Y - DX - L - E||_F^2
    Solution via ADMM:
    Initializations: X=0?, L=0?, E=0?, Lambda=ones?, beta=?
    Iterations:
    (1) Solve L_{k+1}= argmin L : alpha||L||_* + (beta/2)||Y - D@X_k - L - E + (1/beta)Lambda_k||_F^2 
                     = argmin L : ||L||_* + beta/(2*alpha) * ||Y - D@X_k - L - E + (1/beta)Lambda_k||_F^2 
        -> L_{k+1}= Shrink_{(alpha/beta)}(Y - D@X_k - E + (1/beta)Lambda_k) 
                  = U @ S_{(alpha/beta)}(Sigma) @ V^t, for the SVD of Y - D@X_k - E - (1/beta)Lambda_k.
    (2) Solve X_{k+1}= argmin X : ||X||_1 + (beta/2)||Y - D@X_k - E - L + (1/beta)Lambda_k||_F^2 
        -> Lasso problem
    (2.5) Solve E_{k+1} = argmin E : delta/2||E||_F^2 + <Lambda,Y - D@X - L - E> + (beta/2)||Y - D@X - L - E||_F^2
    -> E_{k+1} = (Lambda + beta(Y - D@X - L))/(beta + delta)
    (3) Lambda_{k+1}= Lambda_k + beta(Y - D@X_k - L - E)

    Input: 
    Y:                  test sample
    D:                  dictionary consisting of training frames
    alpha:              weighting hyperparameter for ||L||_*
    delta:              weighting hyperparameter for ||E||_F^2
    global_max_iter:    maximum number of iterations for ADMM method
    lasso_max_iter:     maximum number of iterations for solving lasso subproblem

    Output: 
    X:  sparse matrix
    L:  low-rank matrix
    '''
    M, K = Y.shape
    N = D.shape[1]

    X = np.zeros((N, K))
    L = np.zeros((M, K))
    E = np.zeros((M, K))
    Lambda = np.ones((M, K))

    Dt = D.T
    DtD = Dt @ D

    tau, evecs_large_sparse = largest_eigsh(DtD, 1, which='LM')
    tauInv = 1 / tau
    beta = (20 * M * K) / np.sum(np.abs(Y))
    betaInv = 1 / beta
    betaTauInv = betaInv * tauInv

    tolX = 1e-6
    tolL = 1e-6
    for i in range(global_max_iter):
        print('\nGlobal Iteration %d of %d' % (i, global_max_iter))
        X_old = X
        L_old = L
        E_old = E

        # Solve L_{k+1}= argmin L : alpha||L||_* + (beta/2)||Y - D@X_k - L - E + (1/beta)Lambda_k||_F^2
        U, S, V = scipy.sparse.linalg.svds(Y - D @ X - E + (1 / beta) * Lambda)
        S = shrink(S, (alpha / beta))
        L = U @ np.diag(S) @ V

        # (2) Solve X_{k+1}= argmin X : ||X||_1 + (beta/2)||Y - D@X_k - E - L + (1/beta)Lambda_k||_F^2
        # Reduced to x_{k+1}= argmin x : (2/beta)||x||_1 + ||b-Dx||_2^2 per column
        b = (Y - L - E + (1 / beta) * Lambda)
        Dtb = Dt @ b
        print("begin FastIllinois")
        start = time.process_time()
        for c in range(K):
            f = lambda x: (2/beta)* np.linalg.norm(x,ord=1) + np.linalg.norm(b[:, c] - (D@x))**2
            X[:, c] = FastIllinoisSolver(DtD, Dtb[:, c], X[:, c], f, tauInv, betaTauInv, lasso_max_iter)
        elapsed = (time.process_time() - start)
        print("FastIllinois time: ", elapsed)

        #(2.5) Solve E_{k+1} = argmin E : delta/2||E||_F^2 + <Lambda,Y - D@X - L - E> + (beta/2)||Y - D@X - L - E||_F^2
        #-> E_{k+1} = (Lambda + beta(Y - D@X - L))/(beta + delta)
        E = (Lambda + beta*(Y - D@X - L))/(beta + delta)

        # (3) Lambda_{k+1}= Lambda_k+ beta(Y - D@X_k - L - E)
        Lambda = Lambda + beta * (Y - D @ X - L - E)

        # Stopping Criteria
        if (np.linalg.norm(X_old - X) < tolX * np.linalg.norm(X_old)) and (
                np.linalg.norm(L_old - L) < tolL * np.linalg.norm(L_old)):
            break
        print('\nL1 norm of X =', np.sum(np.abs(X)))
        print('Nuclear norm of L =', np.sum(S))
        print('Frobenius norm of E =', np.linalg.norm(E))
        print('Objective funcion', np.sum(np.abs(X)) + alpha * np.sum(S) + (delta/2)*np.linalg.norm(E)**2)
        print('Constraint error', np.linalg.norm(Y - D @ X - L - E))

    return X, L

