import numpy as np
from shrink import shrink
from FastIllinoisSolver import FastIllinoisSolver

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
        -> Lasso problem. Should be solved using ADMM
    (3) Lambda_{k+1}= Lambda_k+ beta(Y - D@X_k - L)

    Input: Y, D, alpha, global_max_iter, lasso_max_iter
    Output: X, L
    '''
    M, K = Y.shape
    N = D.shape[1]

    X = np.zeros(N,K)
    L = np.zeros(M,K)
    Lambda = np.ones(M,K)

    Dt = D.T
    DtD = Dt @ D 
    tau = np.max(np.abs(np.linalg.eigvals(DtD)))
    tauInv = 1/tau
    beta = (20*M*K) / np.sum(np.abs(Y))
    betaInv = 1 / beta
    betaTauInv = betaInv * tauInv

    tolX = 1e-6
    tolL = 1e-6
    for i in range(global_max_iter):
        print('Global Iteration %d of %d \n'%(i,global_max_iter))
        X_old = X
        L_old = L   
        
        # (1) Solve L_{k+1}= argmin L : alpha||L||_* + beta/(2*alpha) * ||Y - D@X_k - L + (1/beta)Lambda_k||_F^2 
        U,S,V = np.linalg.svd(Y - D@X + (1/beta)*Lambda, full_matrices=False)
        S = shrink(S,(alpha/beta))
        L = U@np.diag(S)@V
        
        # (2) Solve X_{k+1}= argmin X : ||X||_1 + (beta/2)||Y - D@X_k - L + (1/beta)Lambda_k||_F^2 
        # Reduced to x_{k+1}= argmin x : ||x||_1 + (beta/2)||b-Ax||_2^2 per column
        # Reduced to x_{k+1}= argmin x : (2/beta)||x||_1 + ||b-Ax||_2^2 per column
        b = (Y - L + (1/beta)*Lambda)
        Dtb = Dt@b
        for c in range(K):
            X[:,c] = FastIllinoisSolver(DtD, Dtb[:,c], X[:,c], tauInv, betaTauInv,lasso_max_iter)

        # (3) Lambda_{k+1}= Lambda_k+ beta(Y-D@X_k-L)
        Lambda = Lambda + beta*(Y - D@X - L)
        
        # Stopping Criteria
        if (np.linalg.norm(X_old - X) < tolX *  np.linalg.norm(X_old)) and (np.linalg.norm(L_old - L) < tolL * np.linalg.norm(L_old)):
            break
        
        # Print Error
        energy = np.linalg.norm(X, ord=1) + alpha * np.sum(S)
        constraint_error = np.linalg.norm(Y - D@X - L)**2
        print('L1 norm of X =', np.linalg.norm(X, ord=1))
        print('Nuclear norm of L =', np.sum(S))
        print('EnergyFunction =', energy)
        print('Constraint Error =', constraint_error)
    return X, L