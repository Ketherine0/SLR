import numpy as np
from shrink import shrink
from FastIllinoisSolver import FastIllinoisSolver
import time
from scipy.sparse.linalg import eigsh as largest_eigsh
import scipy.sparse.linalg


def FastSolver(Y, D, alpha, global_max_iter, lasso_max_iter):
    '''
    Objective:  min_{X,L} ||X||_1 + alpha||L||_* st: Y = DX + L
    Augmented Lagrangian Formulation : ||X||_1 + alpha||L||_* + <Lambda,Y - DX - L > + (beta/2)||Y - DX - L||_F^2
    Solution via ADMM:
    Initializations: X=0?, L=0?, Lambda=ones?, beta=?
    Iterations:
    (1) Solve L_{k+1}= argmin L : alpha||L||_* + (beta/2)||Y - D@X_k - L +(1/beta)Lambda_k||_F^2 
                     = argmin L : ||L||_* + beta/(2*alpha) * ||Y - D@X_k - L +(1/beta)Lambda_k||_F^2 
        -> L_{k+1}= D_{(alpha/beta)}(Y - D@X_k + (1/beta)Lambda_k) 
                  = U @ S_{(alpha/beta)}(Sigma) @ V^t, for the SVD of Y - D@X_k -(1/beta)Lambda_k.
    (2) Solve X_{k+1}= argmin X : ||X||_1 + (beta/2)||Y - D@X_k - L +(1/beta)Lambda_k||_F^2 
        -> Lasso problem
    (3) Lambda_{k+1}= Lambda_k+ beta(Y - D@X_k - L)

    Input: 
    Y:                  test sample
    D:                  dictionary consisting of training frames
    alpha:              weighting hyperparameter
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
    Lambda = np.ones((M, K))

    Dt = D.T
    DtD = Dt @ D
    tau = np.max(np.abs(np.linalg.eigvals(DtD)))
    DtD = Dt @ D

    # Benchmark the sparse routine
    start = time.process_time()
    tau, evecs_large_sparse = largest_eigsh(DtD, 1, which='LM')
    elapsed = (time.process_time() - start)
    print("eigsh elapsed time: ", elapsed)

    tauInv = 1 / tau
    beta = (20 * M * K) / np.sum(np.abs(Y))
    betaInv = 1 / beta
    betaTauInv = betaInv * tauInv

    tolX = 1e-6
    tolL = 1e-6
    for i in range(global_max_iter):
        print('Global Iteration %d of %d \n' % (i, global_max_iter))
        X_old = X
        L_old = L

        # (1) Solve L_{k+1}= argmin L : ||L||_* + beta/(2*alpha) * ||Y - D@X_k - L + (1/beta)Lambda_k||_F^2 
        start = time.process_time()
        U, S, V = scipy.sparse.linalg.svds(Y - D @ X + (1 / beta) * Lambda)
        elapsed = (time.process_time() - start)
        print("SVD time: ", elapsed)

        S = shrink(S, (alpha / beta))
        L = U @ np.diag(S) @ V

        # (2) Solve X_{k+1}= argmin X : ||X||_1 + (beta/2)||Y - D@X_k - L +(1/beta)Lambda_k||_F^2
        # Reduced to x_{k+1}= argmin x : (2/beta)||x||_1 + ||b-Dx||_2^2 per column
        b = (Y - L + (1 / beta) * Lambda)
        Dtb = Dt @ b
        print("begin FastIllinois")

        start = time.process_time()
        for c in range(K):
            print('col: ', c)
            X[:, c] = FastIllinoisSolver(DtD, Dtb[:, c], X[:, c], tauInv, betaTauInv, lasso_max_iter)
        elapsed = (time.process_time() - start)
        print("eigsh elapsed time: ", elapsed)

        # (3) Lambda_{k+1}= Lambda_k+ beta(Y-D@X_k-L)
        Lambda = Lambda + beta * (Y - D @ X - L)

        # Stopping Criteria
        print(np.linalg.norm(X_old - X))
        print(tolX * np.linalg.norm(X_old))
        if (np.linalg.norm(X_old - X) < tolX * np.linalg.norm(X_old)) and (
                np.linalg.norm(L_old - L) < tolL * np.linalg.norm(L_old)):
            break

