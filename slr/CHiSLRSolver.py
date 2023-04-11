import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigsh as largest_eigsh

def group_sparse_rep(Y, D, y_train, lambdaL, lambdaG, maxIter, eps = 0.05, rho = 1.05):
    '''
    Optimization:  min_{X,L}: ||X||_1 + lambdaG*sum(||X_g||_F) + lambdaL*||L||_* st: Y = DX + L

    Input: 
    Y:          test sample (m by p)
    D:          dictionary consisting of training frames (m by n)
    y_train:    label of each column of D matrix
    lambdaL:    weighting hyperparameter
    lambdaG:    weighting hyperparameter
    eps:        tolerance
    rho:        scaling hyperparameter

    Output: 
    X:  sparse matrix
    L:  low-rank matrix
    '''
    Dt = D.T
    DtD = Dt @ D
    tau, _ = largest_eigsh(DtD, 1, which='LM')
    tau = 5 / tau

    mD,nD = D.shape
    pY = Y.shape[1]

    mu = 1e-4
    mu_max = 1e6

    X = np.zeros((nD,pY))
    Z = np.zeros((mD,pY))
    L = Z
    error = []

    for iter in range(maxIter):
        # Update L
        U, S, V = svds(Y - D @ X + Z / mu)
        r_term = np.sum(S>(lambdaL/mu))
        if r_term >= 1:
            S = S[:r_term]-lambdaL/mu
        else:
            r_term = 1
            S = np.zeros(1)
        L = U[:,:r_term] @ np.diag(S) @ V[:r_term,:]

        # Update X
        temp = Y - L + Z / mu
        G = Dt @ (D @ X - temp)
        R = X - tau*G
        alpha = tau/mu
        beta = lambdaG*tau/mu
        
        for i in np.unique(y_train):
            idx = [index for index, element in enumerate(y_train) if element == i]
            Rg = R[idx,:]
            
            H = np.maximum(Rg - alpha, 0) + np.minimum(Rg + alpha,0)
            nH = np.linalg.norm(H)
            
            if nH == 0:
                X[idx,:] = 0
            else:
                H_new = H / nH * np.maximum(nH - beta, 0)
                for l in range(H_new.shape[0]):
                    X[idx[l],:] =  H[l,:]

        # Update Z 
        Z = Z + mu*(Y - L - D@X)
    
        mu = np.minimum(mu*rho, mu_max)
        stopCriterion = np.linalg.norm(Y - L - D@X)
        error.append(stopCriterion)
        if np.remainder(iter+1,20) == 0:
            print('Iter %d Error %.5f'%(iter+1,stopCriterion))
        if stopCriterion < eps:
            print('Iter %d Error %.5f'%(iter+1,stopCriterion))
            break
    return X, L, error